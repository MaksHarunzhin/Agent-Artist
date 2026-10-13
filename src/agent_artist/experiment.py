"""
agent-artist-experiment: серии прогонов и анализ готовых прогонов.

Главный эксперимент — помогает ли художникам общий контекст диалога с экспертом:
одна идея прогоняется несколько раз в режимах shared (художники видят вопросы друг
друга) и isolated (только свои). LLM и Yandex ART недетерминированы (seed Yandex ART
через API не фиксируется), поэтому выводы делаются по повторам, а не по одному прогону.
"""

import argparse
import os
import time
from dataclasses import replace
from pathlib import Path

from .cli import add_experiment_arguments, load_environment, settings_from_arguments
from .config import Settings
from .generators import create_image_generators
from .limits import classify_and_summarize as limits_summary
from .llm import create_yandex_client
from .recovery import recover_and_summarize
from .runlogs import find_logs
from .studio import ArtStudio
from .summary import summarize
from .telephone import run_telephone
from .timelapse import render_gallery, render_run
from .translation import classify_and_summarize

MODE_NAMES = {True: "shared", False: "isolated"}


def run_series(base_settings: Settings, repeats: int, root: Path,
               modes: tuple[bool, ...] = (False, True)) -> list[Path]:
    """
    Прогнать идею repeats раз в каждом из режимов (shared_questions = False/True).
    Порядок режимов меняется от повтора к повтору, чтобы ни один не шёл всегда первым.
    """
    yandex_client = create_yandex_client(base_settings.yandex_api_key,
                                         base_settings.yandex_folder_id)
    # SDXL загружается один раз на всю серию
    image_generators = create_image_generators(base_settings, yandex_client)

    run_dirs = []
    for repeat in range(1, repeats + 1):
        for shared in (modes if repeat % 2 == 1 else tuple(reversed(modes))):
            output_dir = root / f"{repeat:02d}_{MODE_NAMES[shared]}"
            print(f"\n######## Повтор {repeat} из {repeats}, режим {MODE_NAMES[shared]} ########")
            try:
                ArtStudio(replace(base_settings, shared_questions=shared, output_dir=output_dir),
                          image_generators).run()
            except Exception as error:
                # Один упавший прогон не должен обрывать всю серию
                print(f"  ! Прогон завершился ошибкой: {error}")
            run_dirs.append(output_dir)
    return run_dirs


def render_timelapses(paths: list[Path]) -> None:
    for log_path in find_logs(paths):
        rendered = render_run(log_path.parent)
        if rendered is None:
            print(f"  ! Пропускаю {log_path}: нет итераций или старый формат лога")
        else:
            print("\n".join(str(path) for path in rendered))


def keys_only_settings() -> Settings:
    """Настройки для анализа: нужны только ключи Яндекса."""
    settings = Settings(yandex_api_key=os.environ.get("YANDEX_API_KEY", ""),
                        yandex_folder_id=os.environ.get("YANDEX_FOLDER_ID", ""))
    settings.check()
    return settings


# Команды анализа: имя → (описание, нужны ли ключи Яндекса, действие над папками)
ANALYSIS_COMMANDS = {
    "summary": ("сводка оценок эксперта по готовым прогонам", False,
                lambda paths: print(summarize(paths))),
    "recover": ("«угадай заказ»: сколько из заказа дошло до картин", True,
                lambda paths: print(recover_and_summarize(paths, keys_only_settings()))),
    "translation": ("как художники с английским промптом перенесли русскую идею", True,
                    lambda paths: print(classify_and_summarize(paths, keys_only_settings()))),
    "limits": ("карта непосильных деталей: в чём генераторы систематически промахиваются", True,
               lambda paths: print(limits_summary(paths, keys_only_settings()))),
    "timelapse": ("таймлапс эволюции картин: timelapse.webp/.gif и evolution.png", False,
                  render_timelapses),
}


def main(argv: list[str] | None = None) -> None:
    defaults = Settings()
    parser = argparse.ArgumentParser(prog="agent-artist-experiment",
                                     description="Серии прогонов и анализ их результатов")
    commands = parser.add_subparsers(dest="command", required=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    run_parser = commands.add_parser("run", help="прогнать идею в режимах isolated и shared")
    add_experiment_arguments(run_parser, defaults)
    run_parser.add_argument("--repeats", type=int, default=3, help="повторов каждого режима")
    run_parser.add_argument("--iterations", type=int, default=3,
                            help="итераций в каждом прогоне (без ранней остановки)")
    run_parser.add_argument("--modes", default="isolated,shared",
                            help="режимы через запятую: isolated, shared")
    run_parser.add_argument("--output-dir", type=Path,
                            default=Path("runs") / f"experiment_{timestamp}")

    telephone_parser = commands.add_parser(
        "telephone", help="«испорченный телефон»: рисуют по пересказу предыдущей картины")
    add_experiment_arguments(telephone_parser, defaults)
    telephone_parser.add_argument("--steps", type=int, default=6, help="длина цепочки")
    telephone_parser.add_argument("--output-dir", type=Path,
                                  default=Path("runs") / f"telephone_{timestamp}")

    gallery_parser = commands.add_parser(
        "gallery", help="сетка лучших картин нескольких прогонов: художники × прогоны")
    gallery_parser.add_argument("paths", type=Path, nargs="+",
                                help="папки с прогонами (log.json ищется рекурсивно)")
    gallery_parser.add_argument("--label", choices=["expert", "idea", "mode"], default="idea",
                                help="чем подписать столбцы")
    gallery_parser.add_argument("--title", default="Лучшие картины")
    gallery_parser.add_argument("--output", type=Path, default=Path("gallery.png"))

    for name, (help_text, _, _) in ANALYSIS_COMMANDS.items():
        commands.add_parser(name, help=help_text).add_argument(
            "paths", type=Path, nargs="+", help="папки с прогонами (log.json ищется рекурсивно)")

    args = parser.parse_args(argv)

    if args.command == "gallery":
        render_gallery(find_logs(args.paths), args.output, args.label, args.title)
        print(f"Галерея: {args.output}")
        return

    if args.command in ANALYSIS_COMMANDS:
        _, needs_keys, action = ANALYSIS_COMMANDS[args.command]
        if needs_keys:
            load_environment()
        action(args.paths)
        return

    load_environment()

    if args.command == "telephone":
        settings = settings_from_arguments(args)
        settings.check()
        print(f"Лента: {run_telephone(settings, args.steps, args.output_dir)}")
        return

    # Все прогоны одной длины: min = max, ранней остановки по оценке нет
    settings = settings_from_arguments(
        args, min_iterations=args.iterations, max_iterations=args.iterations)
    settings.check()
    shared_by_name = {name: shared for shared, name in MODE_NAMES.items()}
    run_series(settings, args.repeats, args.output_dir,
               tuple(shared_by_name[name] for name in args.modes.split(",")))

    summary = summarize([args.output_dir])
    (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"\nСводка сохранена в {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
