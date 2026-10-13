"""
Карта непосильных деталей: что генераторы не могут нарисовать.

Каждое замечание эксперта к нарисованной картине относится к одной категории
(поза, лево/право, количество объектов, свет, стиль…). Доля категорий по генераторам
показывает, в чём каждый из них систематически промахивается. Отказы генератора
не учитываются: там замечание о том, что картины нет. Кеш — limits.json рядом с логом.
"""

from collections import Counter
from pathlib import Path

from .config import Settings
from .llm import YandexLLM, create_yandex_client
from .prompts import LIMIT_CATEGORIES, LIMITS_JUDGE_SYSTEM_PROMPT
from .runlogs import JsonCache, find_logs, has_scores, markdown_table, read_json

GENERATOR_NAMES = {"local": "SDXL-Turbo", "yandex_art": "Yandex ART"}


def classify_problems(problems: list[str], judge: YandexLLM) -> list[str]:
    """Категории замечаний одной картины (один вызов LLM на картину)."""
    numbered = "\n".join(f"{number}. {problem}" for number, problem in enumerate(problems, 1))
    verdict = judge.ask_json(LIMITS_JUDGE_SYSTEM_PROMPT, numbered,
                             required_fields=("categories",), temperature=0.2)
    categories = [str(category).lower() for category in verdict["categories"]][:len(problems)]
    categories += ["other"] * (len(problems) - len(categories))
    return [category if category in LIMIT_CATEGORIES else "other" for category in categories]


def classify_run(log_path: Path, judge: YandexLLM) -> list[tuple[str, str]]:
    """[(генератор, категория)] для всех замечаний к нарисованным картинам прогона."""
    log = read_json(log_path)
    if not has_scores(log):
        return []
    cache = JsonCache(log_path.parent / "limits.json")
    generators = {artist["artist_id"]: GENERATOR_NAMES.get(artist["generator"], artist["generator"])
                  for artist in log["artists"]}

    results = []
    for artist_id, records in log["history"].items():
        for record in records:
            problems = [str(problem) for problem in record["review"]["problems"]]
            if record.get("error") or not problems:
                continue
            entry = cache.get(f"{artist_id}_{record['iteration']:02d}", lambda: {
                "problems": problems, "categories": classify_problems(problems, judge)})
            results += [(generators[artist_id], category) for category in entry["categories"]]
    return results


def classify_and_summarize(paths: list[Path], settings: Settings) -> str:
    """Таблица «категория × генератор»: доля замечаний каждой категории."""
    client = create_yandex_client(settings.yandex_api_key, settings.yandex_folder_id)
    judge = YandexLLM(client, settings.yandex_folder_id, settings.expert_llm_name)

    counts: dict[str, Counter] = {}
    for log_path in find_logs(paths):
        for generator, category in classify_run(log_path, judge):
            counts.setdefault(generator, Counter())[category] += 1
    if not counts:
        return "Нет замечаний к нарисованным картинам."

    generators = sorted(counts)
    totals = {generator: sum(counts[generator].values()) for generator in generators}
    rows = [[f"{LIMIT_CATEGORIES[category]}"]
            + [f"{100 * counts[generator][category] / totals[generator]:.0f}%"
               for generator in generators]
            for category in sorted(LIMIT_CATEGORIES,
                                   key=lambda c: -sum(counts[g][c] / totals[g] for g in generators))]
    header = ["Категория замечаний"] + [f"{g} (n={totals[g]})" for g in generators]
    return ("## Карта непосильных деталей\n\n"
            "Доля замечаний эксперта каждой категории — от всех замечаний к картинам генератора.\n\n"
            + markdown_table(header, rows) + "\n")
