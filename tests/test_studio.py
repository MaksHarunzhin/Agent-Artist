from agent_artist import studio
from agent_artist.config import ARTIST_PRESETS, Settings
from agent_artist.dialogue import Dialogue
from agent_artist.generators.base import ImageGenerator
from agent_artist.records import IterationRecord, Review

from .conftest import LengthLimitedGenerator, moderation_refusal


def test_dialogue_visibility_depends_on_mode():
    dialogue = Dialogue(shared_questions=False)
    dialogue.add("A", "q1", "a1")
    dialogue.add("B", "q2", "a2")
    assert [entry.question for entry in dialogue.visible_to("A")] == ["q1"]
    dialogue.shared_questions = True
    assert len(dialogue.visible_to("A")) == 2
    assert Dialogue.to_text(dialogue.entries) == (
        "Художник A: q1\nЭксперт: a1\nХудожник B: q2\nЭксперт: a2")
    assert Dialogue.to_text([]) == "Пока ничего не согласовано."


def test_fit_prompt_trims_whole_phrases_from_the_end():
    generator = LengthLimitedGenerator(40)
    assert generator.fit_prompt("short, ok") == "short, ok"
    trimmed = generator.fit_prompt(
        "Lighthouse in storm, keeper on railing. Rain, waves, spray, teal palette.")
    assert trimmed == "Lighthouse in storm, keeper on railing."


class Picky(ImageGenerator):
    """Отказывает, пока в промпте есть слово «чучело»."""

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, output_path):
        self.prompts.append(prompt)
        if "чучело" in prompt:
            raise moderation_refusal()


def test_draw_with_rephrasing(monkeypatch):
    contexts = []

    def rewrite(llm, artist, generator, context):
        contexts.append(context)
        return "Масленица, блины, хоровод", ""

    monkeypatch.setattr(studio, "write_fitting_prompt", rewrite)
    artist = ARTIST_PRESETS["B"]

    generator = Picky()
    prompt, error = studio.draw_with_rephrasing(None, artist, generator, "ctx",
                                                "горит чучело", None, attempts=2)
    assert error is None and prompt == "Масленица, блины, хоровод"
    assert len(generator.prompts) == 2 and "отклонил" in contexts[0]

    generator = Picky()
    _, error = studio.draw_with_rephrasing(None, artist, generator, "ctx", "горит чучело",
                                           None, attempts=0)
    assert error is not None and len(generator.prompts) == 1


def make_record(iteration: int, score: int, prompt: str) -> IterationRecord:
    return IterationRecord(iteration, prompt, "", None, "", Review(score, ["p"], "fix"))


def studio_with_history(keep_best: bool, records: list[IterationRecord]) -> studio.ArtStudio:
    art_studio = studio.ArtStudio.__new__(studio.ArtStudio)
    art_studio.settings = Settings(keep_best=keep_best)
    art_studio.dialogue = Dialogue(False)
    art_studio.history = {"A": records}
    return art_studio


def test_context_shows_last_attempt_by_default():
    records = [make_record(1, 4, "first"), make_record(2, 7, "good"), make_record(3, 5, "worse")]
    context = studio_with_history(False, records).build_artist_context(ARTIST_PRESETS["A"])
    assert "Твой прошлый промпт: worse" in context and "лучший" not in context


def test_keep_best_context_returns_to_best_attempt():
    records = [make_record(1, 4, "first"), make_record(2, 7, "good"), make_record(3, 5, "worse")]
    context = studio_with_history(True, records).build_artist_context(ARTIST_PRESETS["A"])
    assert "Твой лучший промпт: good" in context
    assert "Последняя попытка оценена ниже (5/10)" in context


def test_median_review_resists_outliers():
    reviews = [Review(4, ["a"], "x"), Review(9, ["b"], "y"), Review(5, ["c"], "z")]
    median = studio.median_review(reviews)
    assert median.score == 5 and median.problems == ["c"]
    assert studio.median_review([Review(6, [], "")]).score == 6
    # При чётном числе оценок — нижняя из двух средних: строже, а не мягче
    assert studio.median_review([Review(3, [], ""), Review(8, [], "")]).score == 3
