"""
Как художники с английским промптом переносят русскую идею.

Для поговорок это главный вопрос культурной асимметрии: художник рисует дословные
образы («медведь, шкура»), образы английского аналога («цыплята») или просто смысл.
Судья размечает каждый английский промпт; результаты кешируются в translation.json.
"""

import json
from collections import Counter
from pathlib import Path

from .config import Settings
from .llm import YandexLLM, create_yandex_client
from .prompts import TRANSLATION_JUDGE_SYSTEM_PROMPT

CACHE_FILE_NAME = "translation.json"
STRATEGIES = ("literal", "equivalent", "meaning", "none")


def classify_run(log_path: Path, judge: YandexLLM) -> list[tuple[str, str]]:
    """Разметить английские промпты прогона. Возвращает [(artist_id, стратегия)]."""
    run_dir = log_path.parent
    cache_path = run_dir / CACHE_FILE_NAME
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    log = json.loads(log_path.read_text(encoding="utf-8"))
    english_artists = {artist["artist_id"] for artist in log.get("artists", [])
                       if artist.get("prompt_language") == "английском"}

    results = []
    for artist_id, records in log.get("history", {}).items():
        if artist_id not in english_artists:
            continue
        for record in records:
            key = f"{artist_id}_{record['iteration']:02d}"
            if key not in cache:
                verdict = judge.ask_json(
                    TRANSLATION_JUDGE_SYSTEM_PROMPT,
                    f"Идея: {log['idea']}\n\nАнглийский промпт художника:\n{record['prompt']}",
                    required_fields=("strategy",), temperature=0.2)
                strategy = str(verdict["strategy"]).lower()
                cache[key] = {
                    "strategy": strategy if strategy in STRATEGIES else "none",
                    "english_equivalent": str(verdict.get("english_equivalent", "")),
                    "prompt": record["prompt"],
                }
                cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
            results.append((artist_id, cache[key]["strategy"]))
    return results


def classify_and_summarize(paths: list[Path], settings: Settings) -> str:
    """Таблица «идея × художник»: сколько английских промптов какой стратегии."""
    client = create_yandex_client(settings.yandex_api_key, settings.yandex_folder_id)
    judge = YandexLLM(client, settings.yandex_folder_id, settings.expert_llm_name)

    counts: dict[tuple[str, str], Counter] = {}
    for path in paths:
        for log_path in sorted(path.rglob("log.json")):
            idea = json.loads(log_path.read_text(encoding="utf-8"))["idea"]
            for artist_id, strategy in classify_run(log_path, judge):
                counts.setdefault((idea, artist_id), Counter())[strategy] += 1

    if not counts:
        return "Нет английских промптов."

    lines = ["## Как переведена идея (английские промпты)\n",
             "| Идея | Художник | " + " | ".join(STRATEGIES) + " |",
             "|---|---|" + "---|" * len(STRATEGIES)]
    for (idea, artist_id), counter in sorted(counts.items()):
        lines.append(f"| {idea} | {artist_id} | "
                     + " | ".join(str(counter[strategy]) for strategy in STRATEGIES) + " |")
    return "\n".join(lines) + "\n"
