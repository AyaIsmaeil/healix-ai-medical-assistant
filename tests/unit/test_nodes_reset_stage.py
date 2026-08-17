import pytest

import llm_client
from llm_client import set_provider
from nodes.reset_stage import reset_stage


class _ExplodingProvider:
    """Fails the test immediately if the node calls the LLM at all."""

    name = "exploding"

    def generate(self, *, model, prompt, schema, timeout_seconds):
        raise AssertionError("reset_stage must not call the LLM")


@pytest.fixture(autouse=True)
def _isolate_llm_env(monkeypatch):
    monkeypatch.setenv("HEALIX_MODEL_QUALITY", "fake-quality-model")
    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)
    yield
    set_provider(None)


# --- node contract: partial state dict ------------------------------------------


def test_reset_stage_returns_only_the_stage_key():
    result = reset_stage({"thread_id": "t1"})

    assert set(result) == {"stage"}


def test_reset_stage_sets_stage_to_none():
    result = reset_stage({"thread_id": "t1"})

    assert result["stage"] is None


def test_reset_stage_does_not_call_the_llm():
    set_provider(_ExplodingProvider())

    reset_stage({"thread_id": "t1"})  # must not raise


# --- clears regardless of what stage previously held -----------------------------


@pytest.mark.parametrize("previous_stage", ["crisis", "emergency", "followup", "diagnosis", None])
def test_reset_stage_clears_any_previous_stage_value(previous_stage):
    # This is the whole point: a stale value from an earlier turn in the
    # same checkpointed thread (CLAUDE.md > State) must not survive into
    # a new turn, regardless of what it was.
    result = reset_stage({"thread_id": "t1", "stage": previous_stage})

    assert result["stage"] is None
