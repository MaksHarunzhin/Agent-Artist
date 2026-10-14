"""Точка входа: аргументы командной строки → Settings → запуск мастерской."""

import argparse
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from .config import DEFAULT_ARTIST_IDS, EXPERT_TASTES, Settings, artists_by_ids
from .studio import ArtStudio


def load_environment() -> None:
    # .env ищем от текущей папки, чтобы ключи находились при любом способе запуска
    if load_dotenv(find_dotenv(usecwd=True), override=True):
        print("Переменные окружения загружены")


def add_experiment_arguments(parser: argparse.ArgumentParser, defaults: Settings) -> None:
    """Аргументы, общие для одиночного прогона и серии прогонов."""
    parser.add_argument("--idea", default=defaults.idea, help="идея картины")
    parser.add_argument("--expert-sees-image", action=argparse.BooleanOptionalAction,
                        default=defaults.expert_sees_image,
                        help="эксперт оценивает саму картинку, а не только её описание")
    parser.add_argument("--describe", action=argparse.BooleanOptionalAction,
                        default=defaults.describe_images,
                        help="описывать картины локальной VLM (для лога и эксперта без картинки)")
    parser.add_argument("--rephrase", type=int, default=defaults.rephrase_attempts,
                        help="сколько раз переписать промпт после отказа модерации")
    parser.add_argument("--experienced", action="store_true",
                        help="художники знают типичные промахи своего генератора")
    parser.add_argument("--keep-best", action="store_true",
                        help="художник правит лучшую попытку, а не последнюю")
    parser.add_argument("--expert-votes", type=int, default=defaults.expert_votes,
                        help="сколько раз эксперт оценивает картину (берётся медиана)")
    parser.add_argument("--start-rounds", type=int, default=defaults.start_question_rounds,
                        help="раундов вопросов до первой картины")
    parser.add_argument("--seed", type=int, default=defaults.seed,
                        help="seed локального генератора")
    parser.add_argument("--expert", choices=list(EXPERT_TASTES), default=defaults.expert_name,
                        help="вкус эксперта-заказчика")
    parser.add_argument("--artists", default=",".join(DEFAULT_ARTIST_IDS),
                        help="художники через запятую: A — SDXL, англ.; "
                             "B — Yandex ART, рус.; C — Yandex ART, англ.")


def settings_from_arguments(args: argparse.Namespace, **overrides) -> Settings:
    """Настройки из общих аргументов, ключей в окружении и явных значений overrides."""
    values = dict(
        idea=args.idea,
        expert_sees_image=args.expert_sees_image,
        describe_images=args.describe,
        rephrase_attempts=args.rephrase,
        experienced_artists=args.experienced,
        keep_best=args.keep_best,
        expert_votes=args.expert_votes,
        start_question_rounds=args.start_rounds,
        seed=args.seed,
        artists=artists_by_ids(args.artists.split(",")),
        expert_name=args.expert,
        expert_taste=EXPERT_TASTES[args.expert],
        yandex_api_key=os.environ.get("YANDEX_API_KEY", ""),
        yandex_folder_id=os.environ.get("YANDEX_FOLDER_ID", ""),
    )
    values.update(overrides)
    return Settings(**values)


def parse_settings(argv: list[str] | None = None) -> Settings:
    """Собрать настройки одиночного прогона: значения по умолчанию, .env и аргументы."""
    defaults = Settings()

    parser = argparse.ArgumentParser(prog="agent-artist", description="Агент-художник")
    add_experiment_arguments(parser, defaults)
    parser.add_argument("--shared", action=argparse.BooleanOptionalAction,
                        default=defaults.shared_questions,
                        help="художники видят вопросы друг друга")
    parser.add_argument("--min-iter", type=int, default=defaults.min_iterations)
    parser.add_argument("--max-iter", type=int, default=defaults.max_iterations)
    parser.add_argument("--target", type=int, default=defaults.target_score,
                        help="целевая оценка для остановки")
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    args = parser.parse_args(argv)

    return settings_from_arguments(
        args,
        shared_questions=args.shared,
        min_iterations=args.min_iter,
        max_iterations=args.max_iter,
        target_score=args.target,
        output_dir=args.output_dir,
    )


def main(argv: list[str] | None = None) -> None:
    load_environment()
    settings = parse_settings(argv)
    settings.check()
    ArtStudio(settings).run()
