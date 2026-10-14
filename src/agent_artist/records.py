"""Данные, которые накапливаются за прогон и попадают в log.json."""

from dataclasses import dataclass


@dataclass
class QuestionAnswer:
    """Один уточняющий вопрос художника и ответ эксперта."""

    artist_id: str
    question: str
    answer: str


@dataclass
class Review:
    """Оценка картины экспертом."""

    score: int
    problems: list
    instructions: str


@dataclass
class IterationRecord:
    """Всё, что произошло с одним художником за одну итерацию."""

    iteration: int
    prompt: str
    artist_reasoning: str
    image_path: str | None
    description: str
    review: Review
    error: str | None = None


def parse_review(data: dict) -> Review:
    """Превратить ответ эксперта в объект Review, даже если каких-то полей нет."""
    try:
        score = int(data.get("score", 0))
    except (TypeError, ValueError):
        score = 0

    problems = data.get("problems", [])
    if not isinstance(problems, list):
        problems = [str(problems)]

    instructions = str(data.get("instructions", ""))
    return Review(score=score, problems=problems, instructions=instructions)
