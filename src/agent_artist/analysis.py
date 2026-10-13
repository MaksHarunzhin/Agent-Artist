"""
Прогоны → таблицы pandas для ноутбука.

Модуль только читает готовые файлы: log.json, а также кеши анализа recovery.json,
translation.json и chain.json. Моделей он не вызывает, поэтому ноутбук можно
перезапускать бесплатно; посчитать недостающее — командами agent-artist-experiment.
"""

from pathlib import Path

import pandas as pd

from .runlogs import artist_labels, find_logs, has_scores, mode_name, read_json

# Колонки задаются явно, чтобы и пустая таблица (эксперимент ещё не запускался)
# поддерживала фильтры вида frame[frame.run.str.startswith(...)]
SCORE_COLUMNS = ["run", "idea", "mode", "expert_sees_image", "artist_id", "artist", "iteration",
                 "score", "refused", "prompt"]
RECOVERY_COLUMNS = ["run", "idea", "mode", "artist", "artist_id", "iteration", "image", "title",
                    "description", "idea_recognized", "idea_named", "details",
                    "details_conveyed"]
TRANSLATION_COLUMNS = ["idea", "artist_id", "iteration", "strategy", "english_equivalent",
                       "prompt"]
TELEPHONE_COLUMNS = ["idea", "artist_id", "artist", "step", "similarity", "seen_title",
                     "refused"]


def scores_frame(paths: list[Path]) -> pd.DataFrame:
    """Строка на картину: идея, режим, художник, итерация, оценка эксперта, отказ генератора."""
    rows = []
    for log_path in find_logs(paths):
        log = read_json(log_path)
        if not has_scores(log):
            continue
        labels = artist_labels(log)
        for artist_id, records in log["history"].items():
            for record in records:
                rows.append({
                    "run": str(log_path.parent), "idea": log["idea"], "mode": mode_name(log),
                    "expert_sees_image": log.get("expert_sees_image", False),
                    "artist_id": artist_id, "artist": labels.get(artist_id, artist_id),
                    "iteration": record["iteration"], "score": record["review"]["score"],
                    "refused": record.get("error") is not None, "prompt": record["prompt"],
                })
    return pd.DataFrame(rows, columns=SCORE_COLUMNS)


def recovery_frame(paths: list[Path]) -> pd.DataFrame:
    """Результаты «угадай заказ» из кешей recovery.json."""
    rows = []
    for log_path in find_logs(paths):
        cache_path = log_path.parent / "recovery.json"
        if not cache_path.exists():
            continue
        log = read_json(log_path)
        labels = artist_labels(log)
        for result in read_json(cache_path).values():
            rows.append({"run": str(log_path.parent), "idea": log["idea"],
                         "mode": mode_name(log), "artist": labels.get(result["artist_id"]),
                         **result})
    return pd.DataFrame(rows, columns=RECOVERY_COLUMNS)


def translation_frame(paths: list[Path]) -> pd.DataFrame:
    """Стратегии перевода английских промптов из кешей translation.json."""
    rows = []
    for log_path in find_logs(paths):
        cache_path = log_path.parent / "translation.json"
        if not cache_path.exists():
            continue
        idea = read_json(log_path)["idea"]
        for key, entry in read_json(cache_path).items():
            artist_id, iteration = key.split("_")
            rows.append({"idea": idea, "artist_id": artist_id, "iteration": int(iteration),
                         **entry})
    return pd.DataFrame(rows, columns=TRANSLATION_COLUMNS)


def telephone_frame(paths: list[Path]) -> pd.DataFrame:
    """Шаги «испорченного телефона» из chain.json: сходство с исходной идеей по шагам."""
    rows = []
    for path in paths:
        for chain_path in sorted(path.rglob("chain.json")):
            chain = read_json(chain_path)
            artist = chain["artist"]
            label = f"{artist['artist_id']} ({artist['generator']}, {artist['prompt_language'][:3]})"
            for step in chain["steps"]:
                rows.append({"idea": chain["idea"], "artist_id": artist["artist_id"],
                             "artist": label, "step": step["step"],
                             "similarity": step["similarity"], "seen_title": step["seen_title"],
                             "refused": step.get("error") is not None})
    return pd.DataFrame(rows, columns=TELEPHONE_COLUMNS)
