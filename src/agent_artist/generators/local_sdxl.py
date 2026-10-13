"""Локальная генерация изображений через diffusers (SDXL-Turbo)."""

from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image


class LocalImageGenerator:
    """Локальная генерация изображений через diffusers (SDXL-Turbo)."""

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
