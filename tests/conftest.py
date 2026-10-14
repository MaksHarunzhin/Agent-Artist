"""Общие подставные объекты: тесты не вызывают ни LLM, ни генераторы."""

from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, BadRequestError

from agent_artist.generators.base import ImageGenerator
from agent_artist.llm import YandexLLM

REQUEST = httpx.Request("POST", "https://example.invalid")


def connection_error() -> APIConnectionError:
    return APIConnectionError(request=REQUEST)


def moderation_refusal() -> BadRequestError:
    return BadRequestError("Я не могу сгенерировать это изображение",
                           response=httpx.Response(400, request=REQUEST), body=None)


class ScriptedClient:
    """Клиент OpenAI, который отдаёт заготовленные ответы модели по очереди."""

    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.calls = []

    @property
    def responses(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.outputs.pop(0))


@pytest.fixture
def scripted_llm():
    def make(outputs: list[str]) -> tuple[YandexLLM, ScriptedClient]:
        client = ScriptedClient(outputs)
        return YandexLLM(client, "folder", "model"), client
    return make


class LengthLimitedGenerator(ImageGenerator):
    """Генератор с лимитом промпта в символах — для проверки подгонки промпта."""

    def __init__(self, limit: int):
        self.limit = limit

    def generate(self, prompt, output_path):
        pass

    def check_prompt(self, prompt):
        return "длинно" if len(prompt) > self.limit else None
