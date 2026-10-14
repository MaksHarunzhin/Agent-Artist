"""
Шумовой пол: насколько оценки эксперта меняются без всякой причины.

Два источника случайности: сам эксперт (одна и та же картина с тем же контекстом
получает разные оценки) и генератор (Yandex ART по тому же промпту рисует разные
картины). Разница между условиями эксперимента меньше этого разброса — шум.
Кеш — noise.json рядом с логом.
"""

import random
import statistics
from pathlib import Path

from .config import Settings
from .dialogue import Dialogue
from .generators.yandex_art import YandexArtGenerator
from .llm import YandexLLM, create_yandex_client
from .records import QuestionAnswer
from .runlogs import JsonCache, find_logs, has_scores, markdown_table, read_json
from .studio import review_picture


def sample_records(paths: list[Path], count: int, seed: int) -> list[tuple[Path, dict, str, dict]]:
    """Случайные нарисованные картины Yandex ART из прогонов, где эксперт видит картину."""
    candidates = []
    for log_path in find_logs(paths):
        log = read_json(log_path)
        if not has_scores(log) or not log.get("expert_sees_image"):
            continue
        generators = {artist["artist_id"]: artist["generator"] for artist in log["artists"]}
        for artist_id, records in log["history"].items():
            if generators.get(artist_id) != "yandex_art":
                continue
            candidates += [(log_path, log, artist_id, record) for record in records
                           if record.get("image_path")]
    random.Random(seed).shuffle(candidates)
    return candidates[:count]


def measure(paths: list[Path], settings: Settings, count: int, repeats: int) -> str:
    client = create_yandex_client(settings.yandex_api_key, settings.yandex_folder_id)
    expert = YandexLLM(client, settings.yandex_folder_id, settings.expert_vision_llm_name)
    generator = YandexArtGenerator(client, settings.yandex_folder_id, "aliceai-image-art-3.0")

    judge_spreads, generator_spreads = [], []
    for log_path, log, artist_id, record in sample_records(paths, count, settings.seed):
        run_dir = log_path.parent
        cache = JsonCache(run_dir / "noise.json")
        dialogue = Dialogue(bool(log.get("shared_questions")))
        dialogue.entries = [QuestionAnswer(**entry) for entry in log.get("dialogue", [])]
        details = Dialogue.to_text(dialogue.visible_to(artist_id))

        def score(image_path: Path) -> int:
            return review_picture(expert, log["expert_taste"], True, log["idea"], artist_id,
                                  details, record["prompt"], image_path, "").score

        key = f"{artist_id}_{record['iteration']:02d}"
        original = run_dir / Path(record["image_path"]).name
        print(f"  {run_dir.name}/{original.name}...")
        # Шум эксперта: та же картина, тот же контекст, несколько оценок
        judge = cache.get(f"{key}_judge", lambda: {
            "scores": [score(original) for _ in range(repeats)]})["scores"]

        # Шум генератора: тот же промпт, новые картины, по одной оценке на каждую
        def regenerate():
            scores = []
            for attempt in range(repeats):
                path = run_dir / f"noise_{key}_{attempt}.png"
                try:
                    generator.generate(record["prompt"], path)
                except Exception:
                    continue   # отказ модерации — эту попытку пропускаем
                scores.append(score(path))
            return {"scores": scores}
        regenerated = cache.get(f"{key}_generator", regenerate)["scores"]

        judge_spreads.append(statistics.pstdev(judge))
        if len(regenerated) >= 2:
            generator_spreads.append(statistics.pstdev(regenerated))

    def mean(values: list[float]) -> str:
        return f"{statistics.mean(values):.2f}" if values else "—"

    return (
        "## Шумовой пол\n\n"
        f"{count} случайных картин Yandex ART, по {repeats} повтора. Среднее стандартное "
        "отклонение оценки эксперта (0–10):\n\n"
        + markdown_table(["Источник случайности", "Картин", "Разброс оценки"], [
            ["эксперт: та же картина, тот же контекст", len(judge_spreads), mean(judge_spreads)],
            ["генератор: тот же промпт, новые картины", len(generator_spreads),
             mean(generator_spreads)],
        ]) + "\n"
    )
