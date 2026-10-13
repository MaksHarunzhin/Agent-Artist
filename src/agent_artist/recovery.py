"""
«Угадай заказ»: сколько из заказа действительно дошло до картины.

Слепой зритель видит только картину и не знает ни идеи, ни промпта, поэтому
не может «додумать» то, чего на картине нет. Он называет тему и описывает
картину, а судья сравнивает это с настоящим заказом: идеей и пунктами,
согласованными с экспертом. Получаются две метрики, независимые от вкуса эксперта:
узнана ли идея и какая доля согласованных пунктов передана.
"""

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Settings
from .llm import YandexLLM, create_yandex_client
from .prompts import BLIND_VIEWER_SYSTEM_PROMPT, BRIEF_JUDGE_SYSTEM_PROMPT

CACHE_FILE_NAME = "recovery.json"
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

    def recover(self, image_path: Path, idea: str, brief: list[str]) -> tuple[dict, dict]:
        """Вернуть (что увидел зритель, вердикт судьи)."""
        seen = self.viewer.ask_json(
            BLIND_VIEWER_SYSTEM_PROMPT, "Опиши эту картину.", image_path=image_path,
            required_fields=("title", "description"), temperature=0.2)

        numbered_brief = "\n".join(f"{number}. {item}" for number, item in enumerate(brief, 1))
        verdict = self.judge.ask_json(
            BRIEF_JUDGE_SYSTEM_PROMPT,
            f"Идея заказа: {idea}\n\n"
            f"Пункты заказа:\n{numbered_brief or 'нет согласованных пунктов'}\n\n"
            f"Зритель назвал тему: {seen['title']}\n"
            f"Описание зрителя:\n{seen['description']}",
            required_fields=("idea_recognized",), temperature=0.2)
        return seen, verdict


def brief_for_artist(log: dict, artist_id: str) -> list[str]:
    """Пункты заказа, которые знал художник: его вопросы и ответы эксперта (или все — в режиме shared)."""
    items = []
    for entry in log.get("dialogue", []):
        if log.get("shared_questions") or entry["artist_id"] == artist_id:
            items.append(f"{entry['question']} — {entry['answer']}")
    return items


def recover_run(log_path: Path, recovery: BriefRecovery) -> list[Recovery]:
    """
    Прогнать «угадай заказ» по всем картинам прогона.
    Результаты кешируются в recovery.json рядом с логом: повторный запуск ничего не тратит.
    """
    run_dir = log_path.parent
    cache_path = run_dir / CACHE_FILE_NAME
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    log = json.loads(log_path.read_text(encoding="utf-8"))
    results = []
    for artist_id, records in log.get("history", {}).items():
        brief = brief_for_artist(log, artist_id)
        for record in records:
            if not record.get("image_path"):
                continue
            # Путь в логе мог быть записан относительно другой папки — берём имя файла
            image_name = Path(record["image_path"]).name

            if image_name not in cache:
                print(f"  {run_dir.name}/{image_name}...")
                seen, verdict = recovery.recover(run_dir / image_name, log["idea"], brief)
                details = [str(mark).lower() for mark in verdict.get("details", [])][:len(brief)]
                conveyed = sum(DETAIL_SCORES.get(mark, 0.0) for mark in details)
                cache[image_name] = asdict(Recovery(
                    artist_id=artist_id,
                    iteration=record["iteration"],
                    image=image_name,
                    title=str(seen["title"]),
                    description=str(seen["description"]),
                    idea_recognized=bool(verdict["idea_recognized"]),
                    idea_named=bool(verdict.get("idea_named")),
                    details=details,
                    details_conveyed=conveyed / len(brief) if brief else 0.0,
                ))
                cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

            results.append(Recovery(**cache[image_name]))
    return results


def percent(values: list[float]) -> str:
    return f"{100 * statistics.mean(values):.0f}%" if values else "—"


def recover_and_summarize(paths: list[Path], settings: Settings) -> str:
    """Прогнать «угадай заказ» по всем log.json в папках и собрать таблицу «идея × художник»."""
    recovery = BriefRecovery(settings)

    # (идея, режим) → artist_id → результаты; плюс подписи художников
    table: dict[tuple[str, str], dict[str, list[Recovery]]] = {}
    labels = {}
    for path in paths:
        for log_path in ([path] if path.is_file() else sorted(path.rglob("log.json"))):
            log = json.loads(log_path.read_text(encoding="utf-8"))
            if not log.get("history"):
                continue
            mode = "shared" if log.get("shared_questions") else "isolated"
            for artist in log.get("artists", []):
                language = artist.get("prompt_language", "")[:3]
                labels[artist["artist_id"]] = (
                    f"{artist['artist_id']} ({artist['generator']}, {language})")
            for result in recover_run(log_path, recovery):
                row = table.setdefault((log["idea"], mode), {})
                row.setdefault(result.artist_id, []).append(result)

    if not table:
        return "Нет прогонов с картинами."

    artist_ids = sorted({artist_id for row in table.values() for artist_id in row})
    lines = [
        "## Угадай заказ\n",
        "Слепой зритель видит только картину. В ячейке: доля картин, по которым он узнал идею "
        "(в скобках — назвал почти дословно) · средняя доля переданных пунктов заказа.\n",
        "| Идея | Режим | " + " | ".join(labels.get(a, a) for a in artist_ids) + " |",
        "|---|---|" + "---|" * len(artist_ids),
    ]
    for (idea, mode), row in sorted(table.items()):
        cells = []
        for artist_id in artist_ids:
            results = row.get(artist_id, [])
            if not results:
                cells.append("—")
                continue
            recognized = percent([float(r.idea_recognized) for r in results])
            named = percent([float(r.idea_named) for r in results if r.idea_named is not None])
            conveyed = percent([r.details_conveyed for r in results])
            cells.append(f"{recognized} ({named}) · {conveyed} (n={len(results)})")
        lines.append(f"| {idea} | {mode} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
