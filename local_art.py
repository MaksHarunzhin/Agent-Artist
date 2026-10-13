import base64
import json
import re
import time
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

import httpx
import ollama
import torch
from diffusers import AutoPipelineForText2Image
from openai import OpenAI
from PIL import Image
from dotenv import load_dotenv
import os

if load_dotenv(override=True):
    print("Переменные окружения загружены")

IDEA = "Золотистый ретривер"

# True  — художники видят вопросы друг друга и ответы эксперта на них.
# False — каждый художник видит только свой диалог с экспертом.
SHARED_QUESTIONS = False

# --- Продолжительность работы ---
MIN_ITERATIONS = 2     # минимум итераций (у каждого художника столько же картин)
MAX_ITERATIONS = 8     # жёсткий предел итераций
TARGET_SCORE = 8       # цикл завершается, когда у всех художников оценка не ниже этой

# --- Уточняющие вопросы ---
START_QUESTION_ROUNDS = 3       # раундов вопросов до первой картины
FEEDBACK_QUESTION_ROUNDS = 1    # раундов вопросов после каждой критики
MAX_QUESTIONS_PER_ROUND = 4     # максимум вопросов от одного художника за раунд

# --- Yandex AI Studio ---
YANDEX_API_KEY = os.environ.get("YANDEX_API_KEY", "")
YANDEX_FOLDER_ID = os.environ.get("YANDEX_FOLDER_ID", "")
YANDEX_BASE_URL = "https://ai.api.cloud.yandex.net/v1"

ARTIST_LLM_NAME = "deepseek-v4.1-flash"   # модель, которая думает за художников
EXPERT_LLM_NAME = "deepseek-v4.1-flash"   # модель эксперта

# --- Локальная VLM в Ollama ---
OLLAMA_HOST = "http://127.0.0.1:11434"
VISION_MODEL_NAME = "qwen3-vl:8b"

# --- Куда сохранять результаты ---
OUTPUT_DIR = Path("runs") / time.strftime("%Y%m%d_%H%M%S")

# --- Вкусы эксперта: от них зависят все его решения ---
EXPERT_TASTE = """
- Любишь кинематографичные кадры с выразительным светом и глубиной.
- Предпочитаешь живопись и иллюстрацию фотореализму.
- Ценишь, когда в картине есть история и напряжение момента.
- Не любишь перегруженные сцены: один ясный главный сюжет, продуманный фон.
- Предпочитаешь насыщенную, но гармоничную палитру.
"""


@dataclass
class ArtistConfig:
    """Настройки одного художника."""

    artist_id: str          # короткое имя: "A", "B"
    persona: str            # особенность характера художника
    generator: str          # "local" (SDXL-Turbo) или "yandex_art"
    image_model: str        # имя модели генерации изображений
    prompt_language: str    # на каком языке писать промпт
    max_prompt_words: int   # ограничение длины промпта


ARTISTS = [
    ArtistConfig(
        artist_id="A",
        persona="Ты особенно тщательно продумываешь композицию, ракурс и свет.",
        generator="local",
        image_model="stabilityai/sdxl-turbo",
        prompt_language="английском",
        max_prompt_words=60,
    ),
    ArtistConfig(
        artist_id="B",
        persona="Ты особенно тщательно продумываешь композицию, ракурс и свет.",
        generator="yandex_art",
        image_model="aliceai-image-art-3.0",
        prompt_language="русском",
        max_prompt_words=70,
    ),
]


@dataclass
class QuestionAnswer:
    """Один уточняющий вопрос художника и ответ эксперта."""

    artist_id: str
    question: str
    answer: str

@dataclass
class Review:
    """Оценка картины экспертом."""

    score: int
    problems: list
    instructions: str


@dataclass
class IterationRecord:
    """Всё, что произошло с одним художником за одну итерацию."""

    iteration: int
    prompt: str
    artist_reasoning: str
    image_path: str | None
    description: str
    review: Review
    error: str | None = None


JSON_ONLY_INSTRUCTION = "Ответь ТОЛЬКО JSON-объектом, без пояснений и без markdown.\n"


def build_questions_system_prompt(artist: ArtistConfig) -> str:
    """Инструкция для художника: решить, нужны ли ещё уточнения, и задать вопросы."""
    if SHARED_QUESTIONS:
        visibility_note = (
            "Ты также видишь вопросы другого художника и ответы эксперта ему. "
            "Не дублируй эти вопросы и считай те ответы согласованными деталями.\n"
        )
    else:
        visibility_note = ""

    return (
        f"Ты художник-иллюстратор по имени Художник {artist.artist_id}. {artist.persona}\n"
        "Тебе заказали картину, но все решения о деталях принимает арт-эксперт (заказчик).\n"
        "Тебе дают идею, уже согласованные детали и, возможно, критику твоей прошлой картины.\n"
        + visibility_note +
        "Реши, хватает ли информации, чтобы написать максимально точный и подробный промпт.\n"
        "Если не хватает — задай эксперту конкретные вопросы о том, что ещё не согласовано: "
        "сюжет, персонажи и их количество, внешность, одежда, позы, эмоции, место, "
        "время суток, погода, стиль, палитра, освещение, ракурс, композиция, детали фона.\n"
        "Если есть критика — уточни то, что в ней неясно или требует выбора.\n"
        f"Не повторяй вопросы, на которые уже есть ответ. Не больше {MAX_QUESTIONS_PER_ROUND} вопросов.\n"
        + JSON_ONLY_INSTRUCTION +
        'Если нужны уточнения: {"ready": false, "questions": ["вопрос 1", "вопрос 2"]}\n'
        'Если информации достаточно: {"ready": true, "questions": []}'
    )


def build_prompt_writer_system_prompt(artist: ArtistConfig) -> str:
    """Инструкция для художника: написать промпт для модели генерации."""
    return (
        f"Ты художник-иллюстратор по имени Художник {artist.artist_id}. {artist.persona}\n"
        "Ты пишешь промпт для модели генерации изображений.\n"
        "Тебе дают идею, все согласованные с экспертом детали и, возможно, "
        "его критику твоей прошлой картины.\n"
        "Учти КАЖДУЮ согласованную деталь. Если есть критика — исправь каждое замечание.\n"
        f"Промпт пиши на {artist.prompt_language} языке, не длиннее "
        f"{artist.max_prompt_words} слов. Самое важное (сюжет, персонажи, стиль) — в начале.\n"
        + JSON_ONLY_INSTRUCTION +
        '{"reasoning": "по-русски: что учёл и что изменил", "prompt": "текст промпта"}'
    )


EXPERT_ANSWERS_SYSTEM_PROMPT = (
    "Ты арт-эксперт и заказчик картины. Твои вкусы:\n"
    + EXPERT_TASTE +
    "С тобой работают несколько художников. Сейчас один из них задаёт уточняющие вопросы.\n"
    "Отвечай на каждый вопрос конкретно и решительно, исходя из своих вкусов и идеи.\n"
    "Никогда не отвечай «на ваше усмотрение» — принимай решение сам.\n"
    "Не противоречь своим прежним ответам.\n"
    + JSON_ONLY_INSTRUCTION +
    'Ответы в том же порядке, что и вопросы: {"answers": ["ответ 1", "ответ 2"]}'
)

EXPERT_REVIEW_SYSTEM_PROMPT = (
    "Ты строгий арт-эксперт и заказчик картины. Твои вкусы:\n"
    + EXPERT_TASTE +
    "Тебе дают идею, детали, которые ты согласовал с художником, его промпт "
    "и объективное описание получившейся картины.\n"
    "Саму картину ты не видишь — опирайся только на описание.\n"
    "Сравни задуманное с полученным и оцени картину по шкале от 1 до 10.\n"
    "Дай художнику конкретные инструкции, что изменить в следующей версии.\n"
    + JSON_ONLY_INSTRUCTION +
    '{"score": 7, "problems": ["проблема 1", "проблема 2"], "instructions": "что исправить"}'
)

VISION_PROMPT = (
    "Опиши изображение объективно и подробно: какие объекты есть, сколько их, "
    "где они расположены, цвета, стиль, освещение, фон, любой текст.\n"
    "Отдельно перечисли дефекты: искажения анатомии, лишние или слитые предметы, "
    "размытость, артефакты.\n"
    "Не додумывай того, чего на изображении нет."
)


def remove_thinking_block(text: str) -> str:
    """Убрать блок рассуждений <think>...</think>, если модель его выдала."""
    without_thinking = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return without_thinking.strip()


def extract_json(text: str) -> dict:
    """
    Достать JSON-объект из ответа модели.

    Модели иногда оборачивают JSON в markdown или добавляют текст вокруг,
    поэтому берём всё от первой открывающей до последней закрывающей скобки.
    """
    clean_text = remove_thinking_block(text)
    start = clean_text.find("{")
    end = clean_text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("В ответе нет JSON-объекта", clean_text, 0)

    json_text = clean_text[start:end + 1]
    return json.loads(json_text)


def parse_review(data: dict) -> Review:
    """Превратить ответ эксперта в объект Review, даже если каких-то полей нет."""
    try:
        score = int(data.get("score", 0))
    except (TypeError, ValueError):
        score = 0

    problems = data.get("problems", [])
    if not isinstance(problems, list):
        problems = [str(problems)]

    instructions = str(data.get("instructions", ""))
    return Review(score=score, problems=problems, instructions=instructions)


def check_settings() -> None:
    """Проверить, что ключи Яндекса заданы, — до того, как что-то запускать."""
    if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
        raise RuntimeError(
            "Не заданы переменные окружения YANDEX_API_KEY и YANDEX_FOLDER_ID. "
            "Пропишите их в файле .env в корне проекта."
        )


def create_yandex_client() -> OpenAI:
    """Клиент Yandex AI Studio: через него работают LLM и Yandex ART."""
    return OpenAI(
        api_key=YANDEX_API_KEY,
        base_url=YANDEX_BASE_URL,
        project=YANDEX_FOLDER_ID,
        timeout=300,
    )


class YandexLLM:
    """Языковая модель в Yandex AI Studio, вызываемая через Responses API."""

    def __init__(self, client: OpenAI, model_name: str):
        self.client = client
        self.model_uri = f"gpt://{YANDEX_FOLDER_ID}/{model_name}"

    def ask_json(self, system_prompt: str, user_message: str, attempts: int = 3) -> dict:
        """Отправить запрос и получить ответ в виде словаря из JSON."""
        for attempt in range(1, attempts + 1):
            response = self.client.responses.create(
                model=self.model_uri,
                instructions=system_prompt,
                input=user_message,
                temperature=0.7,
            )
            try:
                return extract_json(response.output_text)
            except json.JSONDecodeError:
                print(f"  ! Модель вернула невалидный JSON (попытка {attempt} из {attempts})")

        raise RuntimeError(f"Модель {self.model_uri} не вернула валидный JSON")


class LocalVisionModel:
    """
    Локальная VLM в Ollama, которая описывает картинки.

    Сначала пробуем Responses API. Если Ollama не принимает через него
    изображение, переходим на нативный API Ollama.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

        # trust_env=False — не отправлять локальные запросы через системный прокси
        self.responses_client = OpenAI(
            base_url=f"{OLLAMA_HOST}/v1/",
            api_key="ollama",  # клиенту нужен ключ, Ollama его игнорирует
            http_client=httpx.Client(trust_env=False, timeout=600),
        )
        self.native_client = ollama.Client(host=OLLAMA_HOST, trust_env=False)
        self.use_native_api = False

    def describe(self, image_path: Path) -> str:
        """Получить объективное описание картинки."""
        if not self.use_native_api:
            try:
                return self._describe_with_responses_api(image_path)
            except Exception as error:
                print(f"  ! Responses API не принял изображение ({error}). "
                      "Дальше VLM работает через нативный API Ollama.")
                self.use_native_api = True

        return self._describe_with_native_api(image_path)

    def _describe_with_responses_api(self, image_path: Path) -> str:
        image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        # /no_think отключает режим рассуждений у моделей семейства qwen3
        prompt_text = VISION_PROMPT + "\n\n/no_think"

        response = self.responses_client.responses.create(
            model=self.model_name,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt_text},
                        {"type": "input_image",
                         "image_url": f"data:image/png;base64,{image_base64}"},
                    ],
                }
            ],
            temperature=0.2,
        )
        description = remove_thinking_block(response.output_text)
        if not description:
            raise ValueError("пустой ответ")
        return description

    def _describe_with_native_api(self, image_path: Path) -> str:
        response = self.native_client.chat(
            model=self.model_name,
            messages=[
                {"role": "user", "content": VISION_PROMPT, "images": [str(image_path)]}
            ],
            think=False,
            options={"temperature": 0.2},
        )
        return remove_thinking_block(response.message.content)

    def unload_from_memory(self) -> None:
        """Выгрузить VLM из видеопамяти, чтобы освободить место для SDXL."""
        try:
            self.native_client.generate(model=self.model_name, prompt="", keep_alive=0)
        except Exception:
            pass


class LocalImageGenerator:
    """Локальная генерация изображений через diffusers (SDXL-Turbo)."""

    def __init__(self, model_name: str):
        if torch.cuda.is_available():
            device = "cuda"
            print(f"[{model_name}] работает на GPU: {torch.cuda.get_device_name(0)}")
            dtype = torch.float16
        elif torch.backends.mps.is_available():
            device = "mps"
            print(f"[{model_name}] работает на GPU Apple (MPS)")
            dtype = torch.float16
        else:
            device = "cpu"
            print(f"[{model_name}] видеокарта не найдена, работает на CPU (будет медленно)")
            dtype = torch.float32

        self.pipeline = self._load_pipeline(model_name, dtype)

        if device == "cuda":
            # Части модели переносятся на видеокарту только на время работы
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to(device)

    @staticmethod
    def _load_pipeline(model_name: str, dtype: torch.dtype):
        """Загрузить модель из локального кэша, а если её там нет — скачать."""
        try:
            return AutoPipelineForText2Image.from_pretrained(
                model_name, dtype=dtype, variant="fp16", local_files_only=True,
            )
        except OSError:
            print(f"[{model_name}] модели нет в кэше, скачиваю (около 7 ГБ)...")
            return AutoPipelineForText2Image.from_pretrained(
                model_name, dtype=dtype, variant="fp16",
            )

    def generate(self, prompt: str, output_path: Path) -> None:
        result = self.pipeline(
            prompt=prompt,
            num_inference_steps=4,   # SDXL-Turbo рассчитана на 1–4 шага
            guidance_scale=0.0,      # turbo-модели работают без guidance
            width=512,
            height=512,
        )
        image = result.images[0]
        image.save(output_path)


class YandexArtGenerator:
    """Облачная генерация изображений через Yandex ART."""

    MAX_PROMPT_CHARACTERS = 500

    def __init__(self, client: OpenAI, model_name: str):
        self.client = client
        self.model_uri = f"art://{YANDEX_FOLDER_ID}/{model_name}"

    def generate(self, prompt: str, output_path: Path) -> None:
        if len(prompt) > self.MAX_PROMPT_CHARACTERS:
            print(f"  ! Промпт длиннее {self.MAX_PROMPT_CHARACTERS} символов, обрезаю")
            prompt = prompt[:self.MAX_PROMPT_CHARACTERS]

        response = self.client.images.generate(
            model=self.model_uri,
            prompt=prompt,
            size="1024x1024",
        )
        image_bytes = base64.b64decode(response.data[0].b64_json)
        image = Image.open(BytesIO(image_bytes))
        image.save(output_path)


def create_image_generator(artist: ArtistConfig, yandex_client: OpenAI):
    """Создать генератор изображений, указанный в настройках художника."""
    if artist.generator == "local":
        return LocalImageGenerator(artist.image_model)
    if artist.generator == "yandex_art":
        return YandexArtGenerator(yandex_client, artist.image_model)
    raise ValueError(f"Неизвестный тип генератора: {artist.generator}")


# =============================================================================
# 6. ДИАЛОГ ХУДОЖНИКОВ С ЭКСПЕРТОМ
# =============================================================================

class Dialogue:
    """Все вопросы художников и ответы эксперта за время работы."""

    def __init__(self):
        self.entries: list[QuestionAnswer] = []

    def add(self, artist_id: str, question: str, answer: str) -> None:
        self.entries.append(QuestionAnswer(artist_id, question, answer))

    def visible_to(self, artist_id: str) -> list[QuestionAnswer]:
        """Что видит конкретный художник — зависит от SHARED_QUESTIONS."""
        if SHARED_QUESTIONS:
            return list(self.entries)

        own_entries = []
        for entry in self.entries:
            if entry.artist_id == artist_id:
                own_entries.append(entry)
        return own_entries

    @staticmethod
    def to_text(entries: list[QuestionAnswer]) -> str:
        """Превратить записи диалога в читаемый текст для промпта."""
        if not entries:
            return "Пока ничего не согласовано."

        lines = []
        for entry in entries:
            lines.append(f"Художник {entry.artist_id}: {entry.question}")
            lines.append(f"Эксперт: {entry.answer}")
        return "\n".join(lines)


# =============================================================================
# 7. МАСТЕРСКАЯ: ОСНОВНОЙ ЦИКЛ РАБОТЫ
# =============================================================================

class ArtStudio:
    """Связывает всех участников и проводит цикл работы над картинами."""

    def __init__(self):
        yandex_client = create_yandex_client()

        self.artist_llm = YandexLLM(yandex_client, ARTIST_LLM_NAME)
        self.expert_llm = YandexLLM(yandex_client, EXPERT_LLM_NAME)
        self.vision_model = LocalVisionModel(VISION_MODEL_NAME)

        self.image_generators = {}
        for artist in ARTISTS:
            self.image_generators[artist.artist_id] = create_image_generator(artist, yandex_client)

        self.dialogue = Dialogue()

        self.history: dict[str, list[IterationRecord]] = {}
        for artist in ARTISTS:
            self.history[artist.artist_id] = []

    # ----- Контекст для художника -----

    def build_artist_context(self, artist: ArtistConfig) -> str:
        """Собрать всё, что художник знает на текущий момент."""
        visible_entries = self.dialogue.visible_to(artist.artist_id)

        context = (
            f"Идея картины: {IDEA}\n\n"
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
        active_artists = list(ARTISTS)

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
            build_questions_system_prompt(artist),
            self.build_artist_context(artist),
        )

        questions = []
        for question in response.get("questions", []):
            if question:
                questions.append(str(question))
        questions = questions[:MAX_QUESTIONS_PER_ROUND]

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
            f"Идея картины: {IDEA}\n\n"
            f"Все твои прежние ответы художникам:\n"
            f"{Dialogue.to_text(self.dialogue.entries)}\n\n"
            f"Сейчас спрашивает Художник {artist.artist_id}:\n"
            + "\n".join(numbered_questions)
        )

        response = self.expert_llm.ask_json(EXPERT_ANSWERS_SYSTEM_PROMPT, user_message)

        answers = []
        for answer in response.get("answers", []):
            answers.append(str(answer))
        return answers

    # ----- Шаг 2. Промпт для генерации -----

    def write_prompt(self, artist: ArtistConfig) -> tuple[str, str]:
        """Художник пишет промпт. Возвращает (промпт, пояснение художника)."""
        response = self.artist_llm.ask_json(
            build_prompt_writer_system_prompt(artist),
            self.build_artist_context(artist),
        )
        prompt = str(response.get("prompt", ""))
        reasoning = str(response.get("reasoning", ""))
        return prompt, reasoning

    # ----- Шаг 3. Генерация изображения -----

    def generate_image(self, artist: ArtistConfig, prompt: str,
                       iteration: int) -> tuple[Path | None, str | None]:
        """Сгенерировать картинку. Возвращает (путь к файлу, текст ошибки)."""
        output_path = OUTPUT_DIR / f"artist_{artist.artist_id}_{iteration:02d}.png"
        generator = self.image_generators[artist.artist_id]

        try:
            generator.generate(prompt, output_path)
        except Exception as error:
            print(f"  ! Ошибка генерации у художника {artist.artist_id}: {error}")
            return None, str(error)

        print(f"[Генератор {artist.artist_id}] сохранено: {output_path}")
        return output_path, None

    # ----- Шаг 4. Оценка экспертом -----

    def review_image(self, artist: ArtistConfig, prompt: str, description: str) -> Review:
        """Эксперт оценивает картину по её описанию."""
        agreed_details = self.dialogue.visible_to(artist.artist_id)

        user_message = (
            f"Идея картины: {IDEA}\n\n"
            f"Художник: {artist.artist_id}\n"
            f"Детали, согласованные с ним:\n{Dialogue.to_text(agreed_details)}\n\n"
            f"Промпт художника:\n{prompt}\n\n"
            f"Объективное описание получившейся картины:\n{description}"
        )

        response = self.expert_llm.ask_json(EXPERT_REVIEW_SYSTEM_PROMPT, user_message)
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
            self.run_question_rounds(FEEDBACK_QUESTION_ROUNDS)

        # 2. Каждый художник пишет промпт
        prompts = {}
        reasonings = {}
        for artist in ARTISTS:
            prompt, reasoning = self.write_prompt(artist)
            prompts[artist.artist_id] = prompt
            reasonings[artist.artist_id] = reasoning
            print(f"[Художник {artist.artist_id}] {reasoning}")
            print(f"  промпт: {prompt}")

        # 3. Генерация картин
        image_paths = {}
        errors = {}
        for artist in ARTISTS:
            path, error = self.generate_image(artist, prompts[artist.artist_id], iteration)
            image_paths[artist.artist_id] = path
            errors[artist.artist_id] = error

        # 4. VLM описывает картины
        descriptions = {}
        for artist in ARTISTS:
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
        for artist in ARTISTS:
            artist_id = artist.artist_id

            if errors[artist_id] is not None:
                review = self.review_for_failed_generation(errors[artist_id])
            else:
                review = self.review_image(artist, prompts[artist_id], descriptions[artist_id])

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
        """Проверить, что последняя оценка каждого художника не ниже TARGET_SCORE."""
        for artist in ARTISTS:
            last_record = self.history[artist.artist_id][-1]
            if last_record.review.score < TARGET_SCORE:
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
            "idea": IDEA,
            "shared_questions": SHARED_QUESTIONS,
            "expert_taste": EXPERT_TASTE,
            "artists": [asdict(artist) for artist in ARTISTS],
            "dialogue": dialogue_as_dicts,
            "history": history_as_dicts,
        }

        log_path = OUTPUT_DIR / "log.json"
        log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    def print_summary(self) -> None:
        print("\n=== Итоги ===")
        for artist in ARTISTS:
            best_record = self.history[artist.artist_id][0]
            for record in self.history[artist.artist_id]:
                if record.review.score > best_record.review.score:
                    best_record = record
            print(f"Художник {artist.artist_id}: лучшая картина — "
                  f"{best_record.image_path} ({best_record.review.score}/10)")

    # ----- Запуск -----

    def run(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if SHARED_QUESTIONS:
            mode_text = "видят вопросы друг друга"
        else:
            mode_text = "НЕ видят вопросы друг друга"
        print(f"Идея: {IDEA}. Художники {mode_text}.\n")

        print("=== Согласование деталей ===")
        self.run_question_rounds(START_QUESTION_ROUNDS, min_rounds=1)
        self.save_log()

        for iteration in range(1, MAX_ITERATIONS + 1):
            self.run_iteration(iteration)
            self.save_log()

            if iteration >= MIN_ITERATIONS and self.all_artists_reached_target():
                print("\nВсе художники достигли целевой оценки.")
                break

        self.print_summary()


# =============================================================================
# 8. ТОЧКА ВХОДА
# =============================================================================

def main() -> None:
    check_settings()
    studio = ArtStudio()
    studio.run()


if __name__ == "__main__":
    main()