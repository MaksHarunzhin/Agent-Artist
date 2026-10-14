"""Облачная генерация изображений через Yandex ART."""

import base64
from io import BytesIO
from pathlib import Path

from openai import OpenAI
from PIL import Image

from ..llm import call_with_retries
from .base import ImageGenerator


class YandexArtGenerator(ImageGenerator):
    """Облачная генерация изображений через Yandex ART."""

    MAX_PROMPT_CHARACTERS = 500

    def __init__(self, client: OpenAI, folder_id: str, model_name: str):
        self.client = client
        self.model_uri = f"art://{folder_id}/{model_name}"

    def prompt_limit_description(self) -> str:
        return (
            f"промпт не длиннее {self.MAX_PROMPT_CHARACTERS} символов с пробелами — "
            "это примерно 60 слов. Более длинный промпт будет обрезан."
        )

    def prompt_tips(self) -> str:
        # Из карты непосильных деталей: Yandex ART чаще всего промахивается с расположением и светом
        return (
            "Эта модель хорошо рисует объекты, но чаще всего ошибается в расположении и свете. "
            "Для каждого главного объекта явно укажи, где он в кадре (левая треть, справа на "
            "переднем плане) и куда смотрит персонаж; отдельной фразой — откуда падает свет "
            "и какое время суток."
        )

    def check_prompt(self, prompt: str) -> str | None:
        characters = len(prompt)
        if characters <= self.MAX_PROMPT_CHARACTERS:
            return None

        # Символы LLM считает плохо — переводим лимит в слова с запасом 10%
        words = len(prompt.split())
        target_words = int(words * self.MAX_PROMPT_CHARACTERS / characters * 0.9)
        return (f"промпт длиной {characters} символов при лимите {self.MAX_PROMPT_CHARACTERS}, "
                f"сейчас в нём {words} слов — сократи до {target_words} слов")

    def generate(self, prompt: str, output_path: Path) -> None:
        # Страховка на случай прямого вызова: API не примет длинный промпт
        prompt = self.fit_prompt(prompt)

        response = call_with_retries(
            self.client.images.generate,
            model=self.model_uri,
            prompt=prompt,
            size="1024x1024",
        )
        image_bytes = base64.b64decode(response.data[0].b64_json)
        image = Image.open(BytesIO(image_bytes))
        image.save(output_path)
