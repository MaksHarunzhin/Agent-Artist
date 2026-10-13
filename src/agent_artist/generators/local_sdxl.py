"""Локальная генерация изображений через diffusers (SDXL-Turbo)."""

from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image

from .base import ImageGenerator


class LocalImageGenerator(ImageGenerator):
    """Локальная генерация изображений через diffusers (SDXL-Turbo)."""

    # Текстовый энкодер CLIP читает не больше 77 токенов, включая служебные
    # токены начала и конца. Всё, что дальше, модель просто не видит.
    MAX_PROMPT_TOKENS = 77

    def __init__(self, model_name: str, seed: int):
        self.seed = seed

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

    def count_prompt_tokens(self, prompt: str) -> int:
        return len(self.pipeline.tokenizer(prompt).input_ids)

    def prompt_limit_description(self) -> str:
        return (
            f"Модель читает не больше {self.MAX_PROMPT_TOKENS} токенов CLIP — это примерно "
            "40 английских слов. Всё, что дальше, она не увидит. Пиши короткими фразами "
            "через запятую, без вводных слов."
        )

    def check_prompt(self, prompt: str) -> str | None:
        tokens = self.count_prompt_tokens(prompt)
        if tokens <= self.MAX_PROMPT_TOKENS:
            return None

        # Токены LLM считать не умеет, а слова умеет: переводим лимит в слова
        # по фактическому соотношению для этого промпта, с запасом 10%
        words = len(prompt.split())
        target_words = int(words * self.MAX_PROMPT_TOKENS / tokens * 0.9)
        return (f"промпт занимает {tokens} токенов CLIP при лимите {self.MAX_PROMPT_TOKENS}, "
                f"сейчас в нём {words} слов — сократи до {target_words} слов")

    def generate(self, prompt: str, output_path: Path) -> None:
        # Генератор на CPU даёт одинаковый шум на любом устройстве
        random_generator = torch.Generator("cpu").manual_seed(self.seed)
        result = self.pipeline(
            prompt=prompt,
            generator=random_generator,
            num_inference_steps=4,   # SDXL-Turbo рассчитана на 1–4 шага
            guidance_scale=0.0,      # turbo-модели работают без guidance
            width=512,
            height=512,
        )
        image = result.images[0]
        image.save(output_path)
