"""
«Угадай заказ»: сколько из заказа действительно дошло до картины.

Слепой зритель видит только картину и не знает ни идеи, ни промпта, поэтому
не может «додумать» то, чего на картине нет. Он называет тему и описывает
картину, а судья сравнивает это с настоящим заказом: идеей и пунктами,
согласованными с экспертом. Получаются метрики, независимые от вкуса эксперта:
узнана ли идея и какая доля согласованных пунктов передана.
"""

import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Settings
from .llm import YandexLLM, create_yandex_client
from .prompts import BLIND_VIEWER_SYSTEM_PROMPT, BRIEF_JUDGE_SYSTEM_PROMPT
from .runlogs import JsonCache, artist_labels, find_logs, markdown_table, mode_name, read_json

DETAIL_SCORES = {"yes": 1.0, "partial": 0.5, "no": 0.0}


@dataclass
class Recovery:
    """Результат «угадай заказ» для одной картины."""

    artist_id: str
    iteration: int
    image: str
    title: str
    description: str
    idea_recognized: bool
    details: list[str]          # yes / partial / no для каждого пункта заказа
    details_conveyed: float     # доля переданных пунктов, partial считается за половину
    # Назвал ли зритель идею почти дословно. None — в кешах, посчитанных до этого поля
    idea_named: bool | None = None


class BriefRecovery:
    """Слепой зритель и судья."""

    def __init__(self, settings: Settings):
        client = create_yandex_client(settings.yandex_api_key, settings.yandex_folder_id)
        self.viewer = YandexLLM(client, settings.yandex_folder_id,
                                settings.expert_vision_llm_name)
        self.judge = YandexLLM(client, settings.yandex_folder_id, settings.expert_llm_name)

    def describe(self, image_path: Path) -> dict:
        """Что видит зритель, не знающий заказа: {"title", "description"}."""
        return self.viewer.ask_json(
            BLIND_VIEWER_SYSTEM_PROMPT, "Опиши эту картину.", image_path=image_path,
            required_fields=("title", "description"), temperature=0.2)

    def recover(self, artist_id: str, iteration: int, image_path: Path, idea: str,
                brief: list[str]) -> Recovery:
        seen = self.describe(image_path)
        numbered_brief = "\n".join(f"{number}. {item}" for number, item in enumerate(brief, 1))
        verdict = self.judge.ask_json(
            BRIEF_JUDGE_SYSTEM_PROMPT,
            f"Идея заказа: {idea}\n\n"
            f"Пункты заказа:\n{numbered_brief or 'нет согласованных пунктов'}\n\n"
            f"Зритель назвал тему: {seen['title']}\n"
            f"Описание зрителя:\n{seen['description']}",
            required_fields=("idea_recognized",), temperature=0.2)

        details = [str(mark).lower() for mark in verdict.get("details", [])][:len(brief)]
        conveyed = sum(DETAIL_SCORES.get(mark, 0.0) for mark in details)
        return Recovery(
            artist_id=artist_id, iteration=iteration, image=image_path.name,
            title=str(seen["title"]), description=str(seen["description"]),
            idea_recognized=bool(verdict["idea_recognized"]),
            idea_named=bool(verdict.get("idea_named")),
            details=details, details_conveyed=conveyed / len(brief) if brief else 0.0,
        )


def brief_for_artist(log: dict, artist_id: str) -> list[str]:
    """Пункты заказа, которые знал художник: его вопросы и ответы эксперта (или все — в режиме shared)."""
    return [f"{entry['question']} — {entry['answer']}" for entry in log.get("dialogue", [])
            if log.get("shared_questions") or entry["artist_id"] == artist_id]


def recover_run(log_path: Path, recovery: BriefRecovery) -> list[Recovery]:
    """«Угадай заказ» по всем картинам прогона; кеш — recovery.json рядом с логом."""
    run_dir = log_path.parent
    cache = JsonCache(run_dir / "recovery.json")
    log = read_json(log_path)

    results = []
    for artist_id, records in log.get("history", {}).items():
        brief = brief_for_artist(log, artist_id)
        for record in records:
            if not record.get("image_path"):
                continue
            # Путь в логе мог быть записан относительно другой папки — берём имя файла
            image_path = run_dir / Path(record["image_path"]).name

            def compute():
                print(f"  {run_dir.name}/{image_path.name}...")
                return asdict(recovery.recover(artist_id, record["iteration"], image_path,
                                               log["idea"], brief))

            results.append(Recovery(**cache.get(image_path.name, compute)))
    return results


def percent(values: list[float]) -> str:
    return f"{100 * statistics.mean(values):.0f}%" if values else "—"


def recover_and_summarize(paths: list[Path], settings: Settings) -> str:
    """Прогнать «угадай заказ» по всем log.json в папках и собрать таблицу «идея × художник»."""
    recovery = BriefRecovery(settings)

    # (идея, режим) → artist_id → результаты
    table: dict[tuple[str, str], dict[str, list[Recovery]]] = {}
    labels = {}
    for log_path in find_logs(paths):
        log = read_json(log_path)
        if not log.get("history"):
            continue
        labels.update(artist_labels(log))
        for result in recover_run(log_path, recovery):
            row = table.setdefault((log["idea"], mode_name(log)), {})
            row.setdefault(result.artist_id, []).append(result)

    if not table:
        return "Нет прогонов с картинами."

    def cell(results: list[Recovery]) -> str:
        if not results:
            return "—"
        recognized = percent([float(r.idea_recognized) for r in results])
        named = percent([float(r.idea_named) for r in results if r.idea_named is not None])
        conveyed = percent([r.details_conveyed for r in results])
        return f"{recognized} ({named}) · {conveyed} (n={len(results)})"

    artist_ids = sorted({artist_id for row in table.values() for artist_id in row})
    rows = [[idea, mode] + [cell(row.get(artist_id, [])) for artist_id in artist_ids]
            for (idea, mode), row in sorted(table.items())]
    return (
        "## Угадай заказ\n\n"
        "Слепой зритель видит только картину. В ячейке: доля картин, по которым он узнал идею "
        "(в скобках — назвал почти дословно) · средняя доля переданных пунктов заказа.\n\n"
        + markdown_table(["Идея", "Режим"] + [labels.get(a, a) for a in artist_ids], rows)
        + "\n"
    )
