"""Генераторы изображений, которыми рисуют художники."""

from openai import OpenAI

from ..config import ArtistConfig, Settings
from .base import ImageGenerator


def create_image_generator(artist: ArtistConfig, settings: Settings,
                           yandex_client: OpenAI) -> ImageGenerator:
    """Создать генератор изображений, указанный в настройках художника."""
    if artist.generator == "local":
        # torch и diffusers тяжёлые — импортируем, только если локальный генератор нужен
        from .local_sdxl import LocalImageGenerator
        return LocalImageGenerator(artist.image_model, settings.seed)
    if artist.generator == "yandex_art":
        from .yandex_art import YandexArtGenerator
        return YandexArtGenerator(yandex_client, settings.yandex_folder_id, artist.image_model)
    raise ValueError(f"Неизвестный тип генератора: {artist.generator}")


def create_image_generators(settings: Settings,
                            yandex_client: OpenAI) -> dict[str, ImageGenerator]:
    """Генераторы всех художников из настроек, по artist_id."""
    generators = {}
    for artist in settings.artists:
        generators[artist.artist_id] = create_image_generator(artist, settings, yandex_client)
    return generators
