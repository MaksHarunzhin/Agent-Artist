"""Языковые модели в Yandex AI Studio и разбор их JSON-ответов."""

import base64
import json
import re
from io import BytesIO
from pathlib import Path

from openai import OpenAI
from PIL import Image

from .config import YANDEX_BASE_URL


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


def image_to_data_url(image_path: Path) -> str:
    """Картинка в виде data URL. JPEG в несколько раз легче PNG, а для оценки его хватает."""
    buffer = BytesIO()
    Image.open(image_path).convert("RGB").save(buffer, format="JPEG", quality=90)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{image_base64}"


def create_yandex_client(api_key: str, folder_id: str) -> OpenAI:
    """Клиент Yandex AI Studio: через него работают LLM и Yandex ART."""
    return OpenAI(
        api_key=api_key,
        base_url=YANDEX_BASE_URL,
        project=folder_id,
        timeout=300,
    )


class YandexLLM:
    """Языковая модель в Yandex AI Studio, вызываемая через Responses API."""

    def __init__(self, client: OpenAI, folder_id: str, model_name: str):
        self.client = client
        self.model_uri = f"gpt://{folder_id}/{model_name}"

    def ask_json(self, system_prompt: str, user_message: str, attempts: int = 3,
                 image_path: Path | None = None) -> dict:
        """
        Отправить запрос и получить ответ в виде словаря из JSON.
        С image_path к сообщению прикладывается картинка (нужна мультимодальная модель).
        """
        if image_path is None:
            model_input = user_message
        else:
            model_input = [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_message},
                    {"type": "input_image", "image_url": image_to_data_url(image_path)},
                ],
            }]

        for attempt in range(1, attempts + 1):
            response = self.client.responses.create(
                model=self.model_uri,
                instructions=system_prompt,
                input=model_input,
                temperature=0.7,
            )
            try:
                return extract_json(response.output_text)
            except json.JSONDecodeError:
                print(f"  ! Модель вернула невалидный JSON (попытка {attempt} из {attempts})")

        raise RuntimeError(f"Модель {self.model_uri} не вернула валидный JSON")
