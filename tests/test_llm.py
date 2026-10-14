import pytest
from openai import APIConnectionError, BadRequestError

from agent_artist import llm
from agent_artist.llm import extract_json

from .conftest import connection_error, moderation_refusal


def test_extract_json_ignores_thinking_and_markdown():
    text = '<think>сначала подумаю</think>\n```json\n{"score": 7}\n```'
    assert extract_json(text) == {"score": 7}


def test_ask_json_retries_until_required_field_present(scripted_llm):
    model, client = scripted_llm(['{"prompt": ""}', "не JSON", '{"prompt": "a dog"}'])
    assert model.ask_json("s", "u", required_fields=("prompt",))["prompt"] == "a dog"
    assert len(client.calls) == 3


def test_ask_json_accepts_zero_score_and_empty_optional_fields(scripted_llm):
    model, _ = scripted_llm(['{"score": 0, "questions": []}'])
    assert model.ask_json("s", "u", required_fields=("score",))["score"] == 0


def test_ask_json_gives_up_after_attempts(scripted_llm):
    model, _ = scripted_llm(['{"prompt": ""}'] * 3)
    with pytest.raises(RuntimeError):
        model.ask_json("s", "u", required_fields=("prompt",))


@pytest.fixture
def no_delays(monkeypatch):
    monkeypatch.setattr(llm, "RETRY_DELAYS_SECONDS", (0, 0, 0))


def test_call_with_retries_survives_network_errors(no_delays):
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise connection_error()
        return "ok"

    assert llm.call_with_retries(flaky) == "ok"
    assert len(calls) == 3


def test_call_with_retries_gives_up(no_delays):
    calls = []

    def down():
        calls.append(1)
        raise connection_error()

    with pytest.raises(APIConnectionError):
        llm.call_with_retries(down)
    assert len(calls) == 4   # три повтора и последняя попытка


def test_call_with_retries_does_not_retry_moderation(no_delays):
    calls = []

    def refused():
        calls.append(1)
        raise moderation_refusal()

    with pytest.raises(BadRequestError):
        llm.call_with_retries(refused)
    assert len(calls) == 1
