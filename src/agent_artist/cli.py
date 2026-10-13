"""Точка входа: аргументы командной строки → Settings → запуск мастерской."""

import argparse
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from .config import Settings
from .studio import ArtStudio


def parse_settings(argv: list[str] | None = None) -> Settings:
    """Собрать настройки прогона: значения по умолчанию, .env и аргументы командной строки."""
    defaults = Settings()

    parser = argparse.ArgumentParser(prog="agent-artist", description="Агент-художник")
    parser.add_argument("--idea", default=defaults.idea, help="идея картины")
    parser.add_argument("--shared", action=argparse.BooleanOptionalAction,
                        default=defaults.shared_questions,
                        help="художники видят вопросы друг друга")
    parser.add_argument("--min-iter", type=int, default=defaults.min_iterations)
    parser.add_argument("--max-iter", type=int, default=defaults.max_iterations)
    parser.add_argument("--target", type=int, default=defaults.target_score,
                        help="целевая оценка для остановки")
    parser.add_argument("--start-rounds", type=int, default=defaults.start_question_rounds,
                        help="раундов вопросов до первой картины")
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--seed", type=int, default=defaults.seed,
                        help="seed локального генератора")
    args = parser.parse_args(argv)

    return Settings(
        idea=args.idea,
        shared_questions=args.shared,
        min_iterations=args.min_iter,
        max_iterations=args.max_iter,
        target_score=args.target,
        start_question_rounds=args.start_rounds,
        output_dir=args.output_dir,
        seed=args.seed,
        yandex_api_key=os.environ.get("YANDEX_API_KEY", ""),
        yandex_folder_id=os.environ.get("YANDEX_FOLDER_ID", ""),
    )


def main(argv: list[str] | None = None) -> None:
    # .env ищем от текущей папки, чтобы ключи находились при любом способе запуска
    if load_dotenv(find_dotenv(usecwd=True), override=True):
        print("Переменные окружения загружены")

    settings = parse_settings(argv)
    settings.check()
    ArtStudio(settings).run()
