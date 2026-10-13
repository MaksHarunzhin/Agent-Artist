"""Мастерская: основной цикл работы художников с экспертом."""

import json
from dataclasses import asdict
from pathlib import Path

from .config import ArtistConfig, Settings
from .dialogue import Dialogue
from .generators import ImageGenerator, create_image_generators
from .llm import YandexLLM, create_yandex_client
from .prompts import (
    build_expert_answers_system_prompt,
    build_expert_review_system_prompt,
    build_prompt_writer_system_prompt,
    build_questions_system_prompt,
    build_shorten_prompt_message,
)
from .records import IterationRecord, Review, parse_review
from .timelapse import render_run
from .vision import LocalVisionModel


# Сколько раз художник может переписать промпт, не уложившийся в лимит генератора
MAX_SHORTEN_ATTEMPTS = 3


def write_fitting_prompt(artist_llm: YandexLLM, artist: ArtistConfig,
                         generator: ImageGenerator, context: str) -> tuple[str, str]:
    """
    Художник пишет промпт по контексту. Возвращает (промпт, пояснение художника).

    Если промпт не укладывается в ограничения генератора, художник сам
    сокращает его — иначе генератор молча отбросит часть деталей.
    """
    system_prompt = build_prompt_writer_system_prompt(
        artist, generator.prompt_limit_description())

    response = artist_llm.ask_json(system_prompt, context, required_fields=("prompt",))
    prompt = str(response.get("prompt", ""))
    reasoning = str(response.get("reasoning", ""))

    for _ in range(MAX_SHORTEN_ATTEMPTS):
        problem = generator.check_prompt(prompt)
        if problem is None:
            break
        print(f"  ! [Художник {artist.artist_id}] {problem} — прошу сократить")
        response = artist_llm.ask_json(
            system_prompt, context + build_shorten_prompt_message(prompt, problem),
            required_fields=("prompt",))
        prompt = str(response.get("prompt", prompt))

    return generator.fit_prompt(prompt), reasoning


class ArtStudio:
    """Связывает всех участников и проводит цикл работы над картинами."""

    def __init__(self, settings: Settings,
                 image_generators: dict[str, ImageGenerator] | None = None):
        """
        image_generators — уже созданные генераторы по artist_id. Нужны, чтобы серия
        прогонов не загружала SDXL заново каждый раз; по умолчанию создаются по настройкам.
        """
        self.settings = settings
        self.artists = settings.artists
        folder_id = settings.yandex_folder_id
        yandex_client = create_yandex_client(settings.yandex_api_key, folder_id)

        self.artist_llm = YandexLLM(yandex_client, folder_id, settings.artist_llm_name)
        self.expert_llm = YandexLLM(yandex_client, folder_id, settings.expert_llm_name)
        self.expert_vision_llm = YandexLLM(yandex_client, folder_id,
                                           settings.expert_vision_llm_name)
        self.vision_model = LocalVisionModel(settings.ollama_host, settings.vision_model_name)

        if image_generators is None:
            image_generators = create_image_generators(settings, yandex_client)
        self.image_generators = image_generators

        self.dialogue = Dialogue(settings.shared_questions)

        self.history: dict[str, list[IterationRecord]] = {}
        for artist in self.artists:
            self.history[artist.artist_id] = []

    # ----- Контекст для художника -----

    def build_artist_context(self, artist: ArtistConfig) -> str:
        """Собрать всё, что художник знает на текущий момент."""
        visible_entries = self.dialogue.visible_to(artist.artist_id)

        context = (
            f"Идея картины: {self.settings.idea}\n\n"
            f"Согласованные детали (вопросы художников и ответы эксперта):\n"
            f"{Dialogue.to_text(visible_entries)}\n"
        )

        artist_history = self.history[artist.artist_id]
        if artist_history:
            last_record = artist_history[-1]
            problems_text = "; ".join(str(problem) for problem in last_record.review.problems)
            context += (
                f"\nТвой прошлый промпт: {last_record.prompt}\n"
                f"Оценка эксперта: {last_record.review.score}/10\n"
                f"Проблемы: {problems_text}\n"
                f"Инструкции эксперта: {last_record.review.instructions}\n"
            )

        return context

    # ----- Шаг 1. Уточняющие вопросы -----

    def run_question_rounds(self, rounds: int, min_rounds: int = 0) -> None:
        """
        Несколько раундов вопросов. В каждом раунде художники спрашивают по очереди.
        Художник, которому больше нечего спросить, в следующих раундах не участвует.
        """
        active_artists = list(self.artists)

        for round_number in range(1, rounds + 1):
            if not active_artists:
                break

            print(f"--- Раунд вопросов {round_number} ---")
            still_asking = []

            for artist in active_artists:
                asked = self.ask_questions(artist, round_number, min_rounds)
                if asked:
                    still_asking.append(artist)

            active_artists = still_asking

    def ask_questions(self, artist: ArtistConfig, round_number: int, min_rounds: int) -> bool:
        """Один художник задаёт вопросы эксперту. Возвращает True, если вопросы были."""
        response = self.artist_llm.ask_json(
            build_questions_system_prompt(artist, self.settings.shared_questions,
                                          self.settings.max_questions_per_round),
            self.build_artist_context(artist),
        )

        questions = []
        for question in response.get("questions", []):
            if question:
                questions.append(str(question))
        questions = questions[:self.settings.max_questions_per_round]

        artist_is_ready = bool(response.get("ready")) and round_number > min_rounds
        if artist_is_ready or not questions:
            print(f"[Художник {artist.artist_id}] Информации достаточно.")
            return False

        answers = self.get_expert_answers(artist, questions)

        for question, answer in zip(questions, answers):
            self.dialogue.add(artist.artist_id, question, answer)
            print(f"[Художник {artist.artist_id}] {question}")
            print(f"[Эксперт] {answer}")

        return True

    def get_expert_answers(self, artist: ArtistConfig, questions: list[str]) -> list[str]:
        """
        Эксперт отвечает на вопросы художника.
        Эксперт всегда видит весь свой диалог со всеми художниками,
        чтобы не противоречить самому себе.
        """
        numbered_questions = []
        for number, question in enumerate(questions, start=1):
            numbered_questions.append(f"{number}. {question}")

        user_message = (
            f"Идея картины: {self.settings.idea}\n\n"
            f"Все твои прежние ответы художникам:\n"
            f"{Dialogue.to_text(self.dialogue.entries)}\n\n"
            f"Сейчас спрашивает Художник {artist.artist_id}:\n"
            + "\n".join(numbered_questions)
        )

        response = self.expert_llm.ask_json(
            build_expert_answers_system_prompt(self.settings.expert_taste), user_message,
            required_fields=("answers",))

        answers = []
        for answer in response.get("answers", []):
            answers.append(str(answer))
        return answers

    # ----- Шаг 2. Промпт для генерации -----

    def write_prompt(self, artist: ArtistConfig) -> tuple[str, str]:
        """Художник пишет промпт по всему, что знает. Возвращает (промпт, пояснение)."""
        return write_fitting_prompt(self.artist_llm, artist,
                                    self.image_generators[artist.artist_id],
                                    self.build_artist_context(artist))

    # ----- Шаг 3. Генерация изображения -----

    def generate_image(self, artist: ArtistConfig, prompt: str,
                       iteration: int) -> tuple[Path | None, str | None]:
        """Сгенерировать картинку. Возвращает (путь к файлу, текст ошибки)."""
        output_path = self.settings.output_dir / f"artist_{artist.artist_id}_{iteration:02d}.png"
        generator = self.image_generators[artist.artist_id]

        try:
            generator.generate(prompt, output_path)
        except Exception as error:
            print(f"  ! Ошибка генерации у художника {artist.artist_id}: {error}")
            return None, str(error)

        print(f"[Генератор {artist.artist_id}] сохранено: {output_path}")
        return output_path, None

    # ----- Шаг 4. Оценка экспертом -----

    def review_image(self, artist: ArtistConfig, prompt: str, image_path: Path,
                     description: str) -> Review:
        """
        Эксперт оценивает картину: по самому изображению или только по описанию.

        Описание от локальной VLM эксперту, который видит картину, не передаётся:
        проверка показала, что модель верит тексту больше, чем изображению,
        и повторяет ошибки описания (например, путаницу «лево/право»).
        """
        agreed_details = self.dialogue.visible_to(artist.artist_id)
        sees_image = self.settings.expert_sees_image

        user_message = (
            f"Идея картины: {self.settings.idea}\n\n"
            f"Художник: {artist.artist_id}\n"
            f"Детали, согласованные с ним:\n{Dialogue.to_text(agreed_details)}\n\n"
            f"Промпт художника:\n{prompt}"
        )
        if not sees_image:
            user_message += f"\n\nОбъективное описание получившейся картины:\n{description}"

        system_prompt = build_expert_review_system_prompt(self.settings.expert_taste, sees_image)
        if sees_image:
            response = self.expert_vision_llm.ask_json(system_prompt, user_message,
                                                       image_path=image_path,
                                                       required_fields=("score",))
        else:
            response = self.expert_llm.ask_json(system_prompt, user_message,
                                                required_fields=("score",))
        return parse_review(response)

    @staticmethod
    def review_for_failed_generation(error: str) -> Review:
        """Оценка, которую получает художник, если картинку не удалось создать."""
        return Review(
            score=0,
            problems=[f"Генерация не удалась: {error}"],
            instructions=(
                "Перепиши промпт: генератор отклонил запрос или произошла ошибка. "
                "Избегай формулировок, которые могут не пройти модерацию."
            ),
        )

    # ----- Одна итерация целиком -----

    def run_iteration(self, iteration: int) -> None:
        print(f"\n=== Итерация {iteration} ===")

        # 1. После критики художники могут уточнить детали
        if iteration > 1:
            self.run_question_rounds(self.settings.feedback_question_rounds)

        # 2. Каждый художник пишет промпт
        prompts = {}
        reasonings = {}
        for artist in self.artists:
            prompt, reasoning = self.write_prompt(artist)
            prompts[artist.artist_id] = prompt
            reasonings[artist.artist_id] = reasoning
            print(f"[Художник {artist.artist_id}] {reasoning}")
            print(f"  промпт: {prompt}")

        # 3. Генерация картин
        image_paths = {}
        errors = {}
        for artist in self.artists:
            path, error = self.generate_image(artist, prompts[artist.artist_id], iteration)
            image_paths[artist.artist_id] = path
            errors[artist.artist_id] = error

        # 4. VLM описывает картины
        descriptions = {}
        for artist in self.artists:
            path = image_paths[artist.artist_id]
            if path is None:
                descriptions[artist.artist_id] = ""
                continue
            description = self.vision_model.describe(path)
            descriptions[artist.artist_id] = description
            print(f"[VLM → Художник {artist.artist_id}] {description[:300]}...")

        # VLM больше не нужна до следующей итерации — освобождаем видеопамять
        self.vision_model.unload_from_memory()

        # 5. Эксперт оценивает картины
        for artist in self.artists:
            artist_id = artist.artist_id

            if errors[artist_id] is not None:
                review = self.review_for_failed_generation(errors[artist_id])
            else:
                review = self.review_image(artist, prompts[artist_id], image_paths[artist_id],
                                           descriptions[artist_id])

            print(f"[Эксперт → Художник {artist_id}] {review.score}/10: {review.instructions}")

            image_path = image_paths[artist_id]
            record = IterationRecord(
                iteration=iteration,
                prompt=prompts[artist_id],
                artist_reasoning=reasonings[artist_id],
                image_path=str(image_path) if image_path else None,
                description=descriptions[artist_id],
                review=review,
                error=errors[artist_id],
            )
            self.history[artist_id].append(record)

    # ----- Условие остановки -----

    def all_artists_reached_target(self) -> bool:
        """Проверить, что последняя оценка каждого художника не ниже целевой."""
        for artist in self.artists:
            last_record = self.history[artist.artist_id][-1]
            if last_record.review.score < self.settings.target_score:
                return False
        return True

    # ----- Сохранение и итоги -----

    def save_log(self) -> None:
        """Сохранить весь ход работы в log.json."""
        history_as_dicts = {}
        for artist_id, records in self.history.items():
            history_as_dicts[artist_id] = []
            for record in records:
                history_as_dicts[artist_id].append(asdict(record))

        dialogue_as_dicts = []
        for entry in self.dialogue.entries:
            dialogue_as_dicts.append(asdict(entry))

        log = {
            "idea": self.settings.idea,
            "shared_questions": self.settings.shared_questions,
            "seed": self.settings.seed,
            "expert_sees_image": self.settings.expert_sees_image,
            "expert_taste": self.settings.expert_taste,
            "artists": [asdict(artist) for artist in self.artists],
            "dialogue": dialogue_as_dicts,
            "history": history_as_dicts,
        }

        log_path = self.settings.output_dir / "log.json"
        log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    def print_summary(self) -> None:
        print("\n=== Итоги ===")
        for artist in self.artists:
            best_record = self.history[artist.artist_id][0]
            for record in self.history[artist.artist_id]:
                if record.review.score > best_record.review.score:
                    best_record = record
            print(f"Художник {artist.artist_id}: лучшая картина — "
                  f"{best_record.image_path} ({best_record.review.score}/10)")

    # ----- Запуск -----

    def run(self) -> None:
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)

        if self.settings.shared_questions:
            mode_text = "видят вопросы друг друга"
        else:
            mode_text = "НЕ видят вопросы друг друга"
        print(f"Идея: {self.settings.idea}. Художники {mode_text}.\n")

        print("=== Согласование деталей ===")
        self.run_question_rounds(self.settings.start_question_rounds, min_rounds=1)
        self.save_log()

        for iteration in range(1, self.settings.max_iterations + 1):
            self.run_iteration(iteration)
            self.save_log()

            if iteration >= self.settings.min_iterations and self.all_artists_reached_target():
                print("\nВсе художники достигли целевой оценки.")
                break

        self.print_summary()

        try:
            rendered = render_run(self.settings.output_dir)
            if rendered:
                print(f"Таймлапс: {rendered[0]}")
        except Exception as error:
            # Таймлапс — приятное дополнение, из-за него прогон не должен падать
            print(f"  ! Не удалось собрать таймлапс: {error}")

