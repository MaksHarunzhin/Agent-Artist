"""Облачная генерация изображений через Yandex ART."""

import base64
from io import BytesIO
from pathlib import Path

from openai import OpenAI
from PIL import Image


class YandexArtGenerator:
    """Облачная генерация изображений через Yandex ART."""

    MAX_PROMPT_CHARACTERS = 500

    def __init__(self, client: OpenAI, folder_id: str, model_name: str):
        self.client = client
        self.model_uri = f"art://{folder_id}/{model_name}"

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
