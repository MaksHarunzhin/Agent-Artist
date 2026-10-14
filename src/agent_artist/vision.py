"""Локальная VLM, которая описывает картинки для эксперта."""

import base64
from pathlib import Path

import httpx
import ollama
from openai import OpenAI

from .llm import remove_thinking_block
from .prompts import VISION_PROMPT


class LocalVisionModel:
    """
    Локальная VLM в Ollama, которая описывает картинки.

    Сначала пробуем Responses API. Если Ollama не принимает через него
    изображение, переходим на нативный API Ollama.
    """

    def __init__(self, host: str, model_name: str):
        self.model_name = model_name

        # trust_env=False — не отправлять локальные запросы через системный прокси
        self.responses_client = OpenAI(
            base_url=f"{host}/v1/",
            api_key="ollama",  # клиенту нужен ключ, Ollama его игнорирует
            http_client=httpx.Client(trust_env=False, timeout=600),
        )
        self.native_client = ollama.Client(host=host, trust_env=False)
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
