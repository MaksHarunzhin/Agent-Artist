"""
«Движущаяся мишень»: становится ли эксперт строже по ходу прогона.

Цикл критики в среднем не улучшает оценку. Это может значить, что художник не
прогрессирует, а может — что прогресс съедают новые требования: на каждую критику
эксперт согласует новые детали и судит по ним. Проверка: первую и последнюю картину
прогона эксперт оценивает заново с одним и тем же, финальным контекстом. Если
последняя выигрывает, художник прогрессировал, а рост маскировало ужесточение.
Кеш — drift.json рядом с логом.
"""

import statistics
from pathlib import Path

from .config import Settings
from .dialogue import Dialogue
from .llm import YandexLLM, create_yandex_client
from .records import QuestionAnswer
from .runlogs import JsonCache, artist_labels, find_logs, has_scores, markdown_table, read_json
from .studio import review_picture


def rereview_run(log_path: Path, expert: YandexLLM) -> list[dict]:
    """Первая и последняя нарисованные картины каждого художника — под финальным контекстом."""
    log = read_json(log_path)
    if not has_scores(log) or not log.get("expert_sees_image"):
        return []
    cache = JsonCache(log_path.parent / "drift.json")
    dialogue = Dialogue(bool(log.get("shared_questions")))
    dialogue.entries = [QuestionAnswer(**entry) for entry in log.get("dialogue", [])]
    labels = artist_labels(log)

    results = []
    for artist_id, records in log["history"].items():
        drawn = [record for record in records if record.get("image_path")]
        if len(drawn) < 2:
            continue
        details = Dialogue.to_text(dialogue.visible_to(artist_id))
        row = {"artist": labels.get(artist_id, artist_id), "idea": log["idea"]}
        for which, record in (("first", drawn[0]), ("last", drawn[-1])):
            image_path = log_path.parent / Path(record["image_path"]).name
            rescored = cache.get(f"{artist_id}_{which}", lambda: {
                "iteration": record["iteration"],
                "original": record["review"]["score"],
                "final_context": review_picture(
                    expert, log["expert_taste"], True, log["idea"], artist_id, details,
                    record["prompt"], image_path, "").score,
            })
            row[f"{which}_original"] = rescored["original"]
            row[f"{which}_final_context"] = rescored["final_context"]
        results.append(row)
    return results


def rereview_and_summarize(paths: list[Path], settings: Settings) -> str:
    client = create_yandex_client(settings.yandex_api_key, settings.yandex_folder_id)
    expert = YandexLLM(client, settings.yandex_folder_id, settings.expert_vision_llm_name)

    rows = [row for log_path in find_logs(paths) for row in rereview_run(log_path, expert)]
    if not rows:
        return "Нет прогонов с двумя и более картинами и экспертом, который видит картину."

    def mean(artist: str | None, key: str) -> str:
        values = [row[key] for row in rows if artist is None or row["artist"] == artist]
        return f"{statistics.mean(values):.2f}"

    table = []
    for artist in sorted({row["artist"] for row in rows}) + [None]:
        n = sum(1 for row in rows if artist is None or row["artist"] == artist)
        table.append([artist or "все", n, mean(artist, "first_original"), mean(artist, "last_original"),
                      mean(artist, "first_final_context"), mean(artist, "last_final_context")])
    last_wins = sum(row["last_final_context"] > row["first_final_context"] for row in rows)
    first_wins = sum(row["last_final_context"] < row["first_final_context"] for row in rows)
    return (
        "## Движущаяся мишень\n\n"
        "Первая и последняя картина прогона: исходные оценки и оценки заново, обе с финальным "
        "контекстом (всеми пунктами, согласованными к концу прогона).\n\n"
        + markdown_table(["Художник", "Прогонов", "Первая, исходно", "Последняя, исходно",
                          "Первая, финальный контекст", "Последняя, финальный контекст"], table)
        + f"\n\nС финальным контекстом последняя картина лучше первой в {last_wins} прогонах, "
          f"хуже — в {first_wins}, наравне — в {len(rows) - last_wins - first_wins}.\n"
    )
