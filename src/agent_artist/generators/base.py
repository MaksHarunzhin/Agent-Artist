"""Общий интерфейс генераторов изображений."""

import re
from abc import ABC, abstractmethod
from pathlib import Path


class ImageGenerator(ABC):
    """Генератор изображений, которым рисует художник."""

    @abstractmethod
    def generate(self, prompt: str, output_path: Path) -> None:
        """Нарисовать картинку по промпту и сохранить её в output_path."""

    def prompt_limit_description(self) -> str:
        """Ограничение на промпт — его сообщают художнику, когда он пишет промпт."""
        return ""

    def check_prompt(self, prompt: str) -> str | None:
        """Описание проблемы, если промпт не подходит генератору, иначе None."""
        return None

    def fit_prompt(self, prompt: str) -> str:
        """
        Последнее средство, если художник не уложился в ограничение:
        отбрасывать фразы с конца, пока промпт не подойдёт. Самое важное
        художник пишет в начале, поэтому теряется второстепенное, а фразы
        не обрываются на полуслове.
        """
        if self.check_prompt(prompt) is None:
            return prompt

        phrases = re.split(r"(?<=[,.;])\s+", prompt.strip())
        while len(phrases) > 1 and self.check_prompt(" ".join(phrases)) is not None:
            phrases.pop()
        fitted = " ".join(phrases).rstrip(",;")
        if fitted != prompt:
            print(f"  ! Промпт не уложился в ограничение генератора, обрезан до: {fitted}")
        return fitted
