from pathlib import Path

from agent_artist import analysis
from agent_artist.cli import parse_settings
from agent_artist.config import EXPERT_TASTES
from agent_artist.summary import RunScores, build_summary


def test_summary_math():
    labels = {"A": "A (local, анг)", "B": "B (yandex_art, рус)"}
    runs = [
        RunScores("Кот", "isolated", True, {"A": [2, 4, 6], "B": [5, 6, 7]}, labels),
        RunScores("Кот", "isolated", True, {"A": [4, 4, 4], "B": [7, 7, 7]}, labels),
        RunScores("Кот", "shared", True, {"A": [3, 5], "B": [6, 8]}, labels),
    ]
    summary = build_summary(runs)
    assert "| A (local, анг) | isolated | 2 | 3.0 ± 1.4 | 5.0 ± 1.4 | 5.0 ± 1.4 | 2.0 ± 2.8 |" in summary
    assert "| A (local, анг) | shared | 1 | 3.0 | 5.0 | 5.0 | 2.0 |" in summary
    assert "| A (local, анг) | shared | 3.0 | 5.0 | — |" in summary


def test_ideas_overview_appears_for_several_ideas():
    runs = [RunScores("Масленица", "isolated", True, {"B": [6, 8]}, {}),
            RunScores("Маяк", "isolated", True, {"B": [6, 7]}, {})]
    summary = build_summary(runs)
    assert summary.startswith("## Сводка по идеям")
    assert "| Масленица | 1 | 6.0 / 8.0 |" in summary


def test_empty_frames_keep_their_columns():
    for frame in (analysis.scores_frame, analysis.recovery_frame, analysis.translation_frame,
                  analysis.limits_frame, analysis.telephone_frame):
        empty = frame([Path("/nonexistent")])
        assert empty.empty and len(empty.columns) > 3


def test_command_line_settings():
    defaults = parse_settings([])
    assert (defaults.shared_questions, defaults.expert_sees_image, defaults.rephrase_attempts,
            defaults.expert_name) == (False, True, 2, "cinematic")
    custom = parse_settings(["--idea", "Кот", "--shared", "--artists", "A,C", "--expert", "child",
                             "--keep-best", "--experienced", "--no-describe", "--rephrase", "0"])
    assert custom.idea == "Кот" and custom.shared_questions
    assert [artist.artist_id for artist in custom.artists] == ["A", "C"]
    assert custom.expert_taste == EXPERT_TASTES["child"]
    assert custom.keep_best and custom.experienced_artists
    assert not custom.describe_images and custom.rephrase_attempts == 0
