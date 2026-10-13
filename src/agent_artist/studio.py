"""Мастерская: основной цикл работы художников с экспертом."""

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
from .runlogs import write_json
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
        # Без картинки эксперт оценивает по описанию — тогда описывать нужно всегда
        self.describes_images = settings.describe_images or not settings.expert_sees_image
        self.vision_model = LocalVisionModel(settings.ollama_host, settings.vision_model_name)
        self.image_generators = image_generators or create_image_generators(settings,
                                                                            yandex_client)
        self.dialogue = Dialogue(settings.shared_questions)
        self.history: dict[str, list[IterationRecord]] = {a.artist_id: [] for a in self.artists}

    # ----- Контекст для художника -----

    def build_artist_context(self, artist: ArtistConfig) -> str:
        """Собрать всё, что художник знает на текущий момент."""
        context = (
            f"Идея картины: {self.settings.idea}\n\n"
            f"Согласованные детали (вопросы художников и ответы эксперта):\n"
            f"{Dialogue.to_text(self.dialogue.visible_to(artist.artist_id))}\n"
        )
        if self.history[artist.artist_id]:
            last = self.history[artist.artist_id][-1]
            context += (
                f"\nТвой прошлый промпт: {last.prompt}\n"
                f"Оценка эксперта: {last.review.score}/10\n"
                f"Проблемы: {'; '.join(str(problem) for problem in last.review.problems)}\n"
                f"Инструкции эксперта: {last.review.instructions}\n"
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
            active_artists = [artist for artist in active_artists
                              if self.ask_questions(artist, round_number, min_rounds)]

    def ask_questions(self, artist: ArtistConfig, round_number: int, min_rounds: int) -> bool:
        """Один художник задаёт вопросы эксперту. Возвращает True, если вопросы были."""
        limit = self.settings.max_questions_per_round
        response = self.artist_llm.ask_json(
            build_questions_system_prompt(artist, self.settings.shared_questions, limit),
            self.build_artist_context(artist),
        )
        questions = [str(question) for question in response.get("questions", []) if question][:limit]

        artist_is_ready = bool(response.get("ready")) and round_number > min_rounds
        if artist_is_ready or not questions:
            print(f"[Художник {artist.artist_id}] Информации достаточно.")
            return False

        for question, answer in zip(questions, self.get_expert_answers(artist, questions)):
            self.dialogue.add(artist.artist_id, question, answer)
            print(f"[Художник {artist.artist_id}] {question}")
            print(f"[Эксперт] {answer}")
        return True

    def get_expert_answers(self, artist: ArtistConfig, questions: list[str]) -> list[str]:
        """
        Эксперт отвечает на вопросы художника. Он всегда видит весь свой диалог
        со всеми художниками, чтобы не противоречить самому себе.
        """
        user_message = (
            f"Идея картины: {self.settings.idea}\n\n"
            f"Все твои прежние ответы художникам:\n"
            f"{Dialogue.to_text(self.dialogue.entries)}\n\n"
            f"Сейчас спрашивает Художник {artist.artist_id}:\n"
            + "\n".join(f"{number}. {question}" for number, question in enumerate(questions, 1))
        )
        response = self.expert_llm.ask_json(
            build_expert_answers_system_prompt(self.settings.expert_taste), user_message,
            required_fields=("answers",))
        return [str(answer) for answer in response.get("answers", [])]

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
        try:
            self.image_generators[artist.artist_id].generate(prompt, output_path)
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
        sees_image = self.settings.expert_sees_image
        user_message = (
            f"Идея картины: {self.settings.idea}\n\n"
            f"Художник: {artist.artist_id}\n"
            f"Детали, согласованные с ним:\n"
            f"{Dialogue.to_text(self.dialogue.visible_to(artist.artist_id))}\n\n"
            f"Промпт художника:\n{prompt}"
        )
        if not sees_image:
            user_message += f"\n\nОбъективное описание получившейся картины:\n{description}"

        expert = self.expert_vision_llm if sees_image else self.expert_llm
        response = expert.ask_json(
            build_expert_review_system_prompt(self.settings.expert_taste, sees_image),
            user_message, image_path=image_path if sees_image else None,
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
        written = {}
        for artist in self.artists:
            prompt, reasoning = self.write_prompt(artist)
            written[artist.artist_id] = (prompt, reasoning)
            print(f"[Художник {artist.artist_id}] {reasoning}")
            print(f"  промпт: {prompt}")

        # 3. Генерация картин: artist_id → (путь или None, ошибка или None)
        generated = {artist.artist_id: self.generate_image(artist, written[artist.artist_id][0],
                                                           iteration)
                     for artist in self.artists}

        # 4. VLM описывает картины
        descriptions = {artist_id: "" for artist_id in generated}
        if self.describes_images:
            for artist_id, (path, _) in generated.items():
                if path:
                    descriptions[artist_id] = self.vision_model.describe(path)
                    print(f"[VLM → Художник {artist_id}] {descriptions[artist_id][:300]}...")
            # VLM больше не нужна до следующей итерации — освобождаем видеопамять
            self.vision_model.unload_from_memory()

        # 5. Эксперт оценивает картины
        for artist in self.artists:
            artist_id = artist.artist_id
            prompt, reasoning = written[artist_id]
            image_path, error = generated[artist_id]

            if error is not None:
                review = self.review_for_failed_generation(error)
            else:
                review = self.review_image(artist, prompt, image_path, descriptions[artist_id])
            print(f"[Эксперт → Художник {artist_id}] {review.score}/10: {review.instructions}")

            self.history[artist_id].append(IterationRecord(
                iteration=iteration, prompt=prompt, artist_reasoning=reasoning,
                image_path=str(image_path) if image_path else None,
                description=descriptions[artist_id], review=review, error=error,
            ))

    # ----- Условие остановки -----

    def all_artists_reached_target(self) -> bool:
        """Последняя оценка каждого художника не ниже целевой."""
        return all(records[-1].review.score >= self.settings.target_score
                   for records in self.history.values())

    # ----- Сохранение и итоги -----

    def save_log(self) -> None:
        """Сохранить весь ход работы в log.json."""
        write_json(self.settings.output_dir / "log.json", {
            "idea": self.settings.idea,
            "shared_questions": self.settings.shared_questions,
            "seed": self.settings.seed,
            "expert_sees_image": self.settings.expert_sees_image,
            "expert": self.settings.expert_name,
            "expert_taste": self.settings.expert_taste,
            "artists": [asdict(artist) for artist in self.artists],
            "dialogue": [asdict(entry) for entry in self.dialogue.entries],
            "history": {artist_id: [asdict(record) for record in records]
                        for artist_id, records in self.history.items()},
        })

    def print_summary(self) -> None:
        print("\n=== Итоги ===")
        for artist_id, records in self.history.items():
            best = max(records, key=lambda record: record.review.score)
            print(f"Художник {artist_id}: лучшая картина — "
                  f"{best.image_path} ({best.review.score}/10)")

    # ----- Запуск -----

    def run(self) -> None:
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        mode_text = "видят" if self.settings.shared_questions else "НЕ видят"
        print(f"Идея: {self.settings.idea}. Художники {mode_text} вопросы друг друга.\n")

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
