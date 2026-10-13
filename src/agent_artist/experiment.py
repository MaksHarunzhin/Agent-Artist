"""
Эксперимент: помогает ли художникам общий контекст диалога с экспертом.

Одна и та же идея прогоняется несколько раз в двух режимах — художники видят
вопросы друг друга (shared) или только свои (isolated), — и оценки сводятся в таблицу.
LLM и Yandex ART недетерминированы (seed Yandex ART через API не фиксируется),
поэтому вывод можно делать только по нескольким повторам, а не по одному прогону.
"""

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path

from .cli import add_experiment_arguments, load_environment, settings_from_arguments
from .config import Settings
from .generators import create_image_generators
from .llm import create_yandex_client
from .recovery import recover_and_summarize
from .telephone import run_telephone
from .timelapse import render_run
from .translation import classify_and_summarize
from .studio import ArtStudio

MODE_NAMES = {True: "shared", False: "isolated"}


# =============================================================================
# Серия прогонов
# =============================================================================

def run_series(base_settings: Settings, repeats: int, root: Path,
               modes: tuple[bool, ...] = (False, True)) -> list[Path]:
    """
    Прогнать идею repeats раз в каждом из режимов (shared_questions = False/True).
    Возвращает папки прогонов.

    Режимы чередуются, а порядок внутри пары меняется от повтора к повтору,
    чтобы ни один режим систематически не шёл первым.
    """
    yandex_client = create_yandex_client(base_settings.yandex_api_key,
                                         base_settings.yandex_folder_id)
    # SDXL загружается один раз на всю серию
    image_generators = create_image_generators(base_settings, yandex_client)

    run_dirs = []
    for repeat in range(1, repeats + 1):
        if repeat % 2 == 1:
            ordered_modes = modes
        else:
            ordered_modes = tuple(reversed(modes))

        for shared in ordered_modes:
            output_dir = root / f"{repeat:02d}_{MODE_NAMES[shared]}"
            settings = replace(base_settings, shared_questions=shared, output_dir=output_dir)
            print(f"\n######## Повтор {repeat} из {repeats}, режим {MODE_NAMES[shared]} ########")

            try:
                ArtStudio(settings, image_generators).run()
            except Exception as error:
                # Один упавший прогон не должен обрывать всю серию
                print(f"  ! Прогон завершился ошибкой: {error}")
            run_dirs.append(output_dir)

    return run_dirs


# =============================================================================
# Сводка по логам
# =============================================================================

@dataclass
class RunScores:
    """Оценки одного прогона: artist_id → оценка на каждой итерации."""

    idea: str
    mode: str
    expert_sees_image: bool
    scores: dict[str, list[int]]
    artist_labels: dict[str, str]   # artist_id → "B (yandex_art, рус)"


def load_run(log_path: Path) -> RunScores | None:
    """Прочитать оценки из log.json. Лог старого формата или без итераций — None."""
    log = json.loads(log_path.read_text(encoding="utf-8"))
    scores = {}
    for artist_id, records in log.get("history", {}).items():
        if not records or "review" not in records[0]:
            return None
        scores[artist_id] = [record["review"]["score"] for record in records]
    if not scores:
        return None

    return RunScores(
        idea=log["idea"],
        mode=MODE_NAMES[bool(log["shared_questions"])],
        # До появления этого поля эксперт всегда оценивал только по описанию
        expert_sees_image=log.get("expert_sees_image", False),
        scores=scores,
        artist_labels=artist_labels(log),
    )


def artist_labels(log: dict) -> dict[str, str]:
    """Подписи художников с генератором и языком промпта — чтобы таблицу можно было читать."""
    labels = {}
    for artist in log.get("artists", []):
        language = artist.get("prompt_language", "")[:3]
        labels[artist["artist_id"]] = f"{artist['artist_id']} ({artist['generator']}, {language})"
    return labels


def mean_and_spread(values: list[float]) -> str:
    """Среднее ± стандартное отклонение (отклонение — если значений больше одного)."""
    if not values:
        return "—"
    mean = statistics.mean(values)
    if len(values) == 1:
        return f"{mean:.1f}"
    return f"{mean:.1f} ± {statistics.stdev(values):.1f}"


def build_summary(runs: list[RunScores]) -> str:
    """Markdown-сводка: итоги по режимам и средняя оценка на каждой итерации."""
    groups: dict[tuple, list[RunScores]] = {}
    for run in runs:
        key = (run.idea, run.expert_sees_image, run.mode)
        groups.setdefault(key, []).append(run)

    labels = {}
    for run in runs:
        labels.update(run.artist_labels)

    lines = []
    if len({run.idea for run in runs}) > 1:
        lines.append(build_ideas_overview(runs, labels))

    for idea, expert_sees_image in sorted({key[:2] for key in groups}):
        if expert_sees_image:
            expert_note = "эксперт видит картинку"
        else:
            expert_note = "эксперт видит только описание"
        lines.append(f"## {idea} — {expert_note}\n")

        modes = [mode for mode in ("isolated", "shared") if (idea, expert_sees_image, mode) in groups]
        artist_ids = sorted({artist_id for mode in modes
                             for run in groups[(idea, expert_sees_image, mode)]
                             for artist_id in run.scores})
        max_iterations = max(len(scores) for mode in modes
                             for run in groups[(idea, expert_sees_image, mode)]
                             for scores in run.scores.values())

        lines.append("| Художник | Режим | Прогонов | Первая | Последняя | Лучшая | Прирост |")
        lines.append("|---|---|---|---|---|---|---|")
        for artist_id in artist_ids:
            for mode in modes:
                series = [run.scores[artist_id] for run in groups[(idea, expert_sees_image, mode)]
                          if run.scores.get(artist_id)]
                lines.append(
                    f"| {labels.get(artist_id, artist_id)} | {mode} | {len(series)} "
                    f"| {mean_and_spread([s[0] for s in series])} "
                    f"| {mean_and_spread([s[-1] for s in series])} "
                    f"| {mean_and_spread([max(s) for s in series])} "
                    f"| {mean_and_spread([s[-1] - s[0] for s in series])} |"
                )

        lines.append("\nСредняя оценка по итерациям:\n")
        header = " | ".join(f"{i}" for i in range(1, max_iterations + 1))
        lines.append(f"| Художник | Режим | {header} |")
        lines.append("|---|---|" + "---|" * max_iterations)
        for artist_id in artist_ids:
            for mode in modes:
                series = [run.scores[artist_id] for run in groups[(idea, expert_sees_image, mode)]
                          if run.scores.get(artist_id)]
                cells = []
                for index in range(max_iterations):
                    cells.append(mean_and_spread([s[index] for s in series if len(s) > index]))
                lines.append(f"| {labels.get(artist_id, artist_id)} | {mode} | "
                             + " | ".join(cells) + " |")
        lines.append("")

    return "\n".join(lines)


def build_ideas_overview(runs: list[RunScores], labels: dict[str, str]) -> str:
    """Общая таблица «идея × художник»: средняя первая и лучшая оценка по всем прогонам идеи."""
    ideas = sorted({run.idea for run in runs})
    artist_ids = sorted({artist_id for run in runs for artist_id in run.scores})

    lines = ["## Сводка по идеям\n",
             "В ячейке: средняя оценка первой картины / средняя лучшая оценка за прогон.\n"]
    header = " | ".join(labels.get(artist_id, artist_id) for artist_id in artist_ids)
    lines.append(f"| Идея | Прогонов | {header} |")
    lines.append("|---|---|" + "---|" * len(artist_ids))

    for idea in ideas:
        idea_runs = [run for run in runs if run.idea == idea]
        cells = []
        for artist_id in artist_ids:
            series = [run.scores[artist_id] for run in idea_runs if run.scores.get(artist_id)]
            if series:
                first = statistics.mean(s[0] for s in series)
                best = statistics.mean(max(s) for s in series)
                cells.append(f"{first:.1f} / {best:.1f}")
            else:
                cells.append("—")
        lines.append(f"| {idea} | {len(idea_runs)} | " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


def summarize(paths: list[Path]) -> str:
    """Найти все log.json в указанных папках и собрать по ним сводку."""
    runs = []
    for path in paths:
        log_paths = [path] if path.is_file() else sorted(path.rglob("log.json"))
        for log_path in log_paths:
            run = load_run(log_path)
            if run is None:
                print(f"  ! Пропускаю {log_path}: нет оценок или старый формат лога")
            else:
                runs.append(run)

    if not runs:
        return "Нет прогонов с оценками."
    return build_summary(runs)


# =============================================================================
# Точка входа
# =============================================================================

def main(argv: list[str] | None = None) -> None:
    defaults = Settings()
    parser = argparse.ArgumentParser(
        prog="agent-artist-experiment",
        description="Сравнение режимов: художники видят вопросы друг друга или только свои",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="прогнать идею в обоих режимах")
    add_experiment_arguments(run_parser, defaults)
    run_parser.add_argument("--repeats", type=int, default=3,
                            help="повторов каждого режима")
    run_parser.add_argument("--iterations", type=int, default=3,
                            help="итераций в каждом прогоне (без ранней остановки)")
    run_parser.add_argument("--modes", default="isolated,shared",
                            help="режимы через запятую: isolated, shared")
    run_parser.add_argument("--output-dir", type=Path,
                            default=Path("runs") / f"experiment_{time.strftime('%Y%m%d_%H%M%S')}")

    telephone_parser = commands.add_parser(
        "telephone", help="«испорченный телефон»: рисуют по пересказу предыдущей картины")
    add_experiment_arguments(telephone_parser, defaults)
    telephone_parser.add_argument("--steps", type=int, default=6, help="длина цепочки")
    telephone_parser.add_argument(
        "--output-dir", type=Path,
        default=Path("runs") / f"telephone_{time.strftime('%Y%m%d_%H%M%S')}")

    summary_parser = commands.add_parser("summary", help="сводка по готовым прогонам")
    summary_parser.add_argument("paths", type=Path, nargs="+",
                                help="папки с прогонами (log.json ищется рекурсивно)")

    recover_parser = commands.add_parser(
        "recover", help="«угадай заказ»: сколько из заказа дошло до картин")
    recover_parser.add_argument("paths", type=Path, nargs="+",
                                help="папки с прогонами (log.json ищется рекурсивно)")

    translation_parser = commands.add_parser(
        "translation", help="как художники с английским промптом перенесли русскую идею")
    translation_parser.add_argument("paths", type=Path, nargs="+",
                                    help="папки с прогонами (log.json ищется рекурсивно)")

    timelapse_parser = commands.add_parser(
        "timelapse", help="таймлапс эволюции картин: timelapse.webp/.gif и evolution.png")
    timelapse_parser.add_argument("paths", type=Path, nargs="+",
                                  help="папки с прогонами (log.json ищется рекурсивно)")

    args = parser.parse_args(argv)

    if args.command == "timelapse":
        for path in args.paths:
            for log_path in sorted(path.rglob("log.json")):
                rendered = render_run(log_path.parent)
                if rendered is None:
                    print(f"  ! Пропускаю {log_path}: нет итераций или старый формат лога")
                else:
                    print("\n".join(str(path) for path in rendered))
        return

    if args.command == "summary":
        print(summarize(args.paths))
        return

    load_environment()

    if args.command == "telephone":
        settings = settings_from_arguments(args)
        settings.check()
        print(f"Лента: {run_telephone(settings, args.steps, args.output_dir)}")
        return

    if args.command in ("recover", "translation"):
        settings = Settings(yandex_api_key=os.environ.get("YANDEX_API_KEY", ""),
                            yandex_folder_id=os.environ.get("YANDEX_FOLDER_ID", ""))
        settings.check()
        if args.command == "recover":
            print(recover_and_summarize(args.paths, settings))
        else:
            print(classify_and_summarize(args.paths, settings))
        return

    # Все прогоны одной длины: min = max, ранней остановки по оценке нет
    settings = settings_from_arguments(
        args, min_iterations=args.iterations, max_iterations=args.iterations)
    settings.check()

    shared_by_name = {name: shared for shared, name in MODE_NAMES.items()}
    modes = tuple(shared_by_name[name] for name in args.modes.split(","))
    run_series(settings, args.repeats, args.output_dir, modes)

    summary = summarize([args.output_dir])
    (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"\nСводка сохранена в {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
