"""Настройки эксперимента: художники, модели, вкусы эксперта, продолжительность работы."""

import time
from dataclasses import dataclass, field
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


@dataclass
class ArtistConfig:
    """Настройки одного художника."""

    artist_id: str          # короткое имя: "A", "B"
    persona: str            # особенность характера художника
    generator: str          # "local" (SDXL-Turbo) или "yandex_art"
    image_model: str        # имя модели генерации изображений
    prompt_language: str    # на каком языке писать промпт
    max_prompt_words: int   # ограничение длины промпта


def default_artists() -> list[ArtistConfig]:
    return [
        ArtistConfig(
            artist_id="A",
            persona="Ты особенно тщательно продумываешь композицию, ракурс и свет.",
            generator="local",
            image_model="stabilityai/sdxl-turbo",
            prompt_language="английском",
            max_prompt_words=40,  # больше не влезет в 77 токенов CLIP
        ),
        ArtistConfig(
            artist_id="B",
            persona="Ты особенно тщательно продумываешь композицию, ракурс и свет.",
            generator="yandex_art",
            image_model="aliceai-image-art-3.0",
            prompt_language="русском",
            max_prompt_words=60,  # больше не влезет в 500 символов Yandex ART
        ),
    ]


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

    # --- Локальная VLM в Ollama ---
    ollama_host: str = "http://127.0.0.1:11434"
    vision_model_name: str = "qwen3-vl:8b"

    output_dir: Path = field(default_factory=default_output_dir)
    expert_taste: str = DEFAULT_EXPERT_TASTE
    artists: list[ArtistConfig] = field(default_factory=default_artists)

    def check(self) -> None:
        """Проверить, что ключи Яндекса заданы, — до того, как что-то запускать."""
        if not self.yandex_api_key or not self.yandex_folder_id:
            raise RuntimeError(
                "Не заданы переменные окружения YANDEX_API_KEY и YANDEX_FOLDER_ID. "
                "Пропишите их в файле .env в корне проекта."
            )
