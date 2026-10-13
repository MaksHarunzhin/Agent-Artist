"""Общее для анализа прогонов: поиск и чтение log.json, подписи художников, кеши."""

import json
from pathlib import Path
from typing import Callable


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def find_logs(paths: list[Path]) -> list[Path]:
    """Все log.json в указанных папках (рекурсивно); указанный файл берётся как есть."""
    found = []
    for path in paths:
        found += [path] if path.is_file() else sorted(path.rglob("log.json"))
    return found


def has_scores(log: dict) -> bool:
    """Есть ли в логе хоть одна оценённая итерация (старый формат лога — без review)."""
    records = [record for records in log.get("history", {}).values() for record in records]
    return bool(records) and "review" in records[0]


def mode_name(log: dict) -> str:
    return "shared" if log.get("shared_questions") else "isolated"


def artist_label(artist: dict) -> str:
    """Короткая подпись для таблиц: «B (yandex_art, рус)»."""
    return f"{artist['artist_id']} ({artist['generator']}, {artist.get('prompt_language', '')[:3]})"


def artist_labels(log: dict) -> dict[str, str]:
    return {artist["artist_id"]: artist_label(artist) for artist in log.get("artists", [])}


class JsonCache:
    """Словарь в JSON-файле рядом с прогоном: дорогие вызовы моделей считаются один раз."""

    def __init__(self, path: Path):
        self.path = path
        self.data = read_json(path) if path.exists() else {}

    def get(self, key: str, compute: Callable[[], dict]) -> dict:
        if key not in self.data:
            self.data[key] = compute()
            write_json(self.path, self.data)   # сразу на диск: прерванный анализ не теряется
        return self.data[key]


def markdown_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)
