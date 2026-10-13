"""Сводка оценок эксперта по готовым прогонам: по идеям, режимам и итерациям."""

import statistics
from dataclasses import dataclass
from pathlib import Path

from .runlogs import artist_labels, find_logs, has_scores, markdown_table, mode_name, read_json


@dataclass
class RunScores:
    """Оценки одного прогона: artist_id → оценка на каждой итерации."""

    idea: str
    mode: str
    expert_sees_image: bool
    scores: dict[str, list[int]]
    artist_labels: dict[str, str]   # artist_id → "B (yandex_art, рус)"


def load_run(log_path: Path) -> RunScores | None:
    """Оценки из log.json; лог старого формата или без итераций — None."""
    log = read_json(log_path)
    if not has_scores(log):
        return None
    return RunScores(
        idea=log["idea"],
        mode=mode_name(log),
        # До появления этого поля эксперт всегда оценивал только по описанию
        expert_sees_image=log.get("expert_sees_image", False),
        scores={artist_id: [record["review"]["score"] for record in records]
                for artist_id, records in log["history"].items()},
        artist_labels=artist_labels(log),
    )


def mean_and_spread(values: list[float]) -> str:
    """Среднее ± стандартное отклонение (отклонение — если значений больше одного)."""
    if not values:
        return "—"
    mean = statistics.mean(values)
    if len(values) == 1:
        return f"{mean:.1f}"
    return f"{mean:.1f} ± {statistics.stdev(values):.1f}"


def series_of(runs: list[RunScores], artist_id: str) -> list[list[int]]:
    """Оценки художника по итерациям — по одному списку на прогон."""
    return [run.scores[artist_id] for run in runs if run.scores.get(artist_id)]


def build_ideas_overview(runs: list[RunScores], labels: dict[str, str]) -> str:
    """Общая таблица «идея × художник»: средняя первая и лучшая оценка по всем прогонам идеи."""
    artist_ids = sorted({artist_id for run in runs for artist_id in run.scores})
    rows = []
    for idea in sorted({run.idea for run in runs}):
        idea_runs = [run for run in runs if run.idea == idea]
        cells = []
        for artist_id in artist_ids:
            series = series_of(idea_runs, artist_id)
            cells.append(f"{statistics.mean(s[0] for s in series):.1f} / "
                         f"{statistics.mean(max(s) for s in series):.1f}" if series else "—")
        rows.append([idea, len(idea_runs)] + cells)

    return ("## Сводка по идеям\n\n"
            "В ячейке: средняя оценка первой картины / средняя лучшая оценка за прогон.\n\n"
            + markdown_table(["Идея", "Прогонов"] + [labels.get(a, a) for a in artist_ids], rows)
            + "\n")


def build_idea_section(idea: str, expert_sees_image: bool, groups: dict[str, list[RunScores]],
                       labels: dict[str, str]) -> str:
    """Итоги по режимам и средние по итерациям для одной идеи. groups: режим → прогоны."""
    modes = [mode for mode in ("isolated", "shared") if mode in groups]
    artist_ids = sorted({artist_id for mode in modes for run in groups[mode]
                         for artist_id in run.scores})
    max_iterations = max(len(scores) for mode in modes for run in groups[mode]
                         for scores in run.scores.values())

    totals, by_iteration = [], []
    for artist_id in artist_ids:
        for mode in modes:
            series = series_of(groups[mode], artist_id)
            label = labels.get(artist_id, artist_id)
            totals.append([label, mode, len(series),
                           mean_and_spread([s[0] for s in series]),
                           mean_and_spread([s[-1] for s in series]),
                           mean_and_spread([max(s) for s in series]),
                           mean_and_spread([s[-1] - s[0] for s in series])])
            by_iteration.append([label, mode] + [
                mean_and_spread([s[index] for s in series if len(s) > index])
                for index in range(max_iterations)])

    expert_note = "эксперт видит картинку" if expert_sees_image else "эксперт видит только описание"
    return (
        f"## {idea} — {expert_note}\n\n"
        + markdown_table(["Художник", "Режим", "Прогонов", "Первая", "Последняя", "Лучшая",
                          "Прирост"], totals)
        + "\n\nСредняя оценка по итерациям:\n\n"
        + markdown_table(["Художник", "Режим"] + [str(i) for i in range(1, max_iterations + 1)],
                         by_iteration)
        + "\n"
    )


def build_summary(runs: list[RunScores]) -> str:
    """Markdown-сводка: общая таблица по идеям и раздел на каждую идею."""
    labels = {}
    groups: dict[tuple[str, bool], dict[str, list[RunScores]]] = {}
    for run in runs:
        labels.update(run.artist_labels)
        groups.setdefault((run.idea, run.expert_sees_image), {}).setdefault(run.mode, []).append(run)

    sections = [build_ideas_overview(runs, labels)] if len({run.idea for run in runs}) > 1 else []
    sections += [build_idea_section(idea, sees_image, modes, labels)
                 for (idea, sees_image), modes in sorted(groups.items())]
    return "\n".join(sections)


def summarize(paths: list[Path]) -> str:
    """Найти все log.json в указанных папках и собрать по ним сводку."""
    runs = []
    for log_path in find_logs(paths):
        run = load_run(log_path)
        if run is None:
            print(f"  ! Пропускаю {log_path}: нет оценок или старый формат лога")
        else:
            runs.append(run)
    return build_summary(runs) if runs else "Нет прогонов с оценками."
