"""Настройки эксперимента: художники, модели, вкусы эксперта, продолжительность работы."""

import time
from dataclasses import dataclass, field, replace
from pathlib import Path

YANDEX_BASE_URL = "https://ai.api.cloud.yandex.net/v1"

# Вкусы эксперта: от них зависят все его решения
DEFAULT_EXPERT_TASTE = """
- Любишь кинематографичные кадры с выразительным светом и глубиной.
- Предпочитаешь живопись и иллюстрацию фотореализму.
- Ценишь, когда в картине есть история и напряжение момента.
- Не любишь перегруженные сцены: один ясный главный сюжет, продуманный фон.
- Предпочитаешь насыщенную, но гармоничную палитру.
"""

# Разные заказчики: одна идея у разных экспертов должна дать разные картины
EXPERT_TASTES = {
    "cinematic": DEFAULT_EXPERT_TASTE,
    "minimalist": """
- Ты минималист: любишь пустое пространство, один главный предмет, 2–3 цвета.
- Ценишь графичность, чистые силуэты и геометрию.
- Не любишь детали, шум, фактуру и «красивости».
""",
    "baroque": """
- Ты любитель барокко: пышность, золото, бархат, драматичный свет из темноты.
- Любишь множество деталей, аллегории, торжественные позы.
- Скромность и пустота кажутся тебе бедностью.
""",
    "child": """
- Тебе шесть лет. Ты любишь яркие цвета, смешных зверей и мультики.
- Хочешь, чтобы на картине было весело, добро и что-нибудь неожиданное.
- Не любишь страшное, тёмное и скучное.
""",
    "peredvizhnik": """
- Ты ворчливый художник-передвижник XIX века: правда жизни, простые люди, социальный смысл.
- Любишь приглушённую, землистую палитру и честный реализм.
- Терпеть не можешь «красивость», фэнтези и глянец.
""",
}


@dataclass
class ArtistConfig:
    """Настройки одного художника."""

    artist_id: str          # короткое имя: "A", "B"
    persona: str            # особенность характера художника
    generator: str          # "local" (SDXL-Turbo) или "yandex_art"
    image_model: str        # имя модели генерации изображений
    prompt_language: str    # на каком языке писать промпт
    max_prompt_words: int   # ограничение длины промпта


ARTIST_PERSONA = "Ты особенно тщательно продумываешь композицию, ракурс и свет."

# Все художники с одинаковым характером: различаются только генератором и языком промпта.
# A и C — один язык, разные генераторы; B и C — один генератор, разные языки.
ARTIST_PRESETS = {
    "A": ArtistConfig(
        artist_id="A",
        persona=ARTIST_PERSONA,
        generator="local",
        image_model="stabilityai/sdxl-turbo",
        prompt_language="английском",
        max_prompt_words=40,  # больше не влезет в 77 токенов CLIP
    ),
    "B": ArtistConfig(
        artist_id="B",
        persona=ARTIST_PERSONA,
        generator="yandex_art",
        image_model="aliceai-image-art-3.0",
        prompt_language="русском",
        max_prompt_words=60,  # больше не влезет в 500 символов Yandex ART
    ),
    "C": ArtistConfig(
        artist_id="C",
        persona=ARTIST_PERSONA,
        generator="yandex_art",
        image_model="aliceai-image-art-3.0",
        prompt_language="английском",
        max_prompt_words=60,
    ),
}

DEFAULT_ARTIST_IDS = ("A", "B")


def artists_by_ids(artist_ids) -> list[ArtistConfig]:
    """Художники из ARTIST_PRESETS по их именам, например ["A", "C"]."""
    unknown = [artist_id for artist_id in artist_ids if artist_id not in ARTIST_PRESETS]
    if unknown:
        raise ValueError(f"Неизвестные художники {unknown}, есть: {list(ARTIST_PRESETS)}")
    return [replace(ARTIST_PRESETS[artist_id]) for artist_id in artist_ids]


def default_artists() -> list[ArtistConfig]:
    return artists_by_ids(DEFAULT_ARTIST_IDS)


def default_output_dir() -> Path:
    return Path("runs") / time.strftime("%Y%m%d_%H%M%S")


@dataclass
class Settings:
    """Все параметры одного прогона."""

    idea: str = "Золотистый ретривер"

    # True  — художники видят вопросы друг друга и ответы эксперта на них.
    # False — каждый художник видит только свой диалог с экспертом.
    shared_questions: bool = False

    # --- Продолжительность работы ---
    min_iterations: int = 2     # минимум итераций (у каждого художника столько же картин)
    max_iterations: int = 8     # жёсткий предел итераций
    target_score: int = 8       # цикл завершается, когда у всех художников оценка не ниже этой

    # Seed локального генератора: при одном и том же промпте картинка будет той же,
    # так что разница между итерациями и прогонами зависит только от промпта.
    seed: int = 42

    # --- Уточняющие вопросы ---
    start_question_rounds: int = 3       # раундов вопросов до первой картины
    feedback_question_rounds: int = 1    # раундов вопросов после каждой критики
    max_questions_per_round: int = 4     # максимум вопросов от одного художника за раунд

    # --- Yandex AI Studio ---
    yandex_api_key: str = ""
    yandex_folder_id: str = ""
    artist_llm_name: str = "deepseek-v4.1-flash"   # модель, которая думает за художников
    expert_llm_name: str = "deepseek-v4.1-flash"   # модель эксперта

    # True  — эксперт оценивает саму картинку (мультимодальная модель в Яндексе),
    #         описание от локальной VLM получает как подсказку.
    # False — эксперт видит только описание от локальной VLM.
    expert_sees_image: bool = True
    expert_vision_llm_name: str = "qwen3.6-35b-a3b"

    # Описывать ли картины локальной VLM. Эксперту, который видит картину, описание
    # не нужно; без него прогон быстрее (~40 с на картину) и не занимает Ollama.
    describe_images: bool = True

    # Сколько раз художник переписывает промпт, если модерация генератора его отклонила
    rephrase_attempts: int = 0

    # «Опытные» художники знают типичные промахи своего генератора (из карты непосильных деталей)
    experienced_artists: bool = False

    # Художник правит лучшую попытку, а не последнюю (элитизм)
    keep_best: bool = False

    # --- Локальная VLM в Ollama ---
    ollama_host: str = "http://127.0.0.1:11434"
    vision_model_name: str = "qwen3-vl:8b"

    output_dir: Path = field(default_factory=default_output_dir)
    expert_name: str = "cinematic"
    expert_taste: str = DEFAULT_EXPERT_TASTE
    artists: list[ArtistConfig] = field(default_factory=default_artists)

    def check(self) -> None:
        """Проверить, что ключи Яндекса заданы, — до того, как что-то запускать."""
        if not self.yandex_api_key or not self.yandex_folder_id:
            raise RuntimeError(
                "Не заданы переменные окружения YANDEX_API_KEY и YANDEX_FOLDER_ID. "
                "Пропишите их в файле .env в корне проекта."
            )
