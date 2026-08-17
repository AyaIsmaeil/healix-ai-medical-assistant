import json

import pytest

import llm_client
from llm_client import _ProviderResponse, set_provider
from nodes.extract_symptoms import extract_symptoms
from schemas.symptoms import SYMPTOM_NAMES, SymptomExtraction
from state import merge_symptoms

# Real vocabulary entries, taken from the vocabulary rather than hardcoded
# Arabic strings, so these tests don't break when the list changes.
NAME_A, NAME_B, NAME_C = SYMPTOM_NAMES[0], SYMPTOM_NAMES[1], SYMPTOM_NAMES[2]


class FakeProvider:
    """Scripted provider: returns `responses` in order."""

    name = "fake"

    def __init__(self, *, responses=None):
        self._responses = list(responses or [])
        self.calls = []

    def generate(self, *, model, prompt, schema, timeout_seconds):
        self.calls.append({"model": model, "prompt": prompt, "schema": schema})
        return self._responses.pop(0)


def extraction_response(symptoms=(), negated=(), unmatched=()) -> _ProviderResponse:
    payload = {
        "symptoms": [dict(s) for s in symptoms],
        "negated_symptoms": [dict(n) for n in negated],
        "unmatched_mentions": list(unmatched),
    }
    return _ProviderResponse(text=json.dumps(payload, ensure_ascii=False))


@pytest.fixture(autouse=True)
def _isolate_llm_env(monkeypatch):
    monkeypatch.setenv("HEALIX_MODEL_QUALITY", "fake-quality-model")
    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)
    yield
    set_provider(None)


def _state(messages, thread_id="thread-1", symptoms=()):
    return {"thread_id": thread_id, "messages": messages, "symptoms": list(symptoms)}


def _user(content):
    return {"role": "user", "content": content}


def _assistant(content):
    return {"role": "assistant", "content": content}


# --- normal extraction ---------------------------------------------------------


def test_extract_symptoms_returns_confirmed_symptoms_ready_for_the_reducer():
    provider = FakeProvider(
        responses=[
            extraction_response(symptoms=[{"name": NAME_A, "raw_mention": "حاسس بالتعب"}])
        ]
    )
    set_provider(provider)

    result = extract_symptoms(_state([_user("حاسس بالتعب من يومين")]))

    assert result["symptoms"] == [
        {
            "name": NAME_A,
            "raw_mention": "حاسس بالتعب",
            "duration": None,
            "severity": None,
            "onset": None,
            "duration_days": None,
        }
    ]


def test_extract_symptoms_output_merges_correctly_via_the_real_reducer():
    # Proves the node's output shape is genuinely compatible with
    # state.merge_symptoms, not just plausible-looking — no reimplemented
    # merge logic in the node itself.
    provider = FakeProvider(
        responses=[
            extraction_response(symptoms=[{"name": NAME_A, "duration": "يومين"}])
        ]
    )
    set_provider(provider)

    existing = [{"name": NAME_A, "severity": "mild"}, {"name": NAME_B}]
    result = extract_symptoms(_state([_user("...")]))
    merged = merge_symptoms(existing, result["symptoms"])

    assert merged == [
        {"name": NAME_A, "severity": "mild", "duration": "يومين", "duration_days": 2},
        {"name": NAME_B},
    ]


# --- negated symptoms captured separately ---------------------------------------


def test_extract_symptoms_captures_negated_symptoms_in_their_own_key():
    provider = FakeProvider(
        responses=[
            extraction_response(
                symptoms=[{"name": NAME_A}],
                negated=[{"name": NAME_B, "raw_mention": "ما عندي"}],
            )
        ]
    )
    set_provider(provider)

    result = extract_symptoms(_state([_user("عندي كذا بس ما عندي كذا")]))

    assert [s["name"] for s in result["symptoms"]] == [NAME_A]
    assert [n["name"] for n in result["negated_symptoms"]] == [NAME_B]
    assert result["negated_symptoms"][0]["raw_mention"] == "ما عندي"


def test_negated_and_confirmed_symptoms_do_not_bleed_into_each_other():
    provider = FakeProvider(
        responses=[
            extraction_response(
                symptoms=[{"name": NAME_A}],
                negated=[{"name": NAME_A}],
            )
        ]
    )
    set_provider(provider)

    # Same name confirmed AND negated is a real, valid scenario (e.g. "كان
    # عندي حمى بس هلق ما عاد في") — the two lists are independent, neither
    # should silently cancel or merge the other out.
    result = extract_symptoms(_state([_user("...")]))

    assert len(result["symptoms"]) == 1
    assert len(result["negated_symptoms"]) == 1


def test_real_end_to_end_rule_layer_recovers_a_negation_the_llm_missed():
    # rules/negation.py wired in as a genuine second layer, OR-combined
    # with the LLM's own negated_symptoms — same pattern as crisis_check
    # and check_red_flags's dual-layer detection, not a documentation
    # claim this time (see nodes/extract_symptoms.py's own module
    # docstring, "Negation" section, for the incident this closes).
    #
    # "ما عندي حمى بس عندي سعال وصداع" ("I don't have a fever, but I have
    # a cough and a headache") — verified directly (not assumed) that
    # rules.negation.detect_negated_symptoms(message) on this exact
    # string returns only {"حمى"}: bounded-window negation correctly
    # attaches "ما عندي" to the fever immediately after it, and does NOT
    # leak across to "سعال"/"صداع", which sit outside NEGATION_WINDOW.
    #
    # The scripted LLM response below simulates a real, plausible miss:
    # it correctly reports the two confirmed symptoms but — as models
    # sometimes do — never surfaces the explicit fever denial as its own
    # negated_symptoms entry. Real graph-level correctness now depends on
    # the rule layer recovering exactly this.
    message = "ما عندي حمى بس عندي سعال وصداع"
    provider = FakeProvider(
        responses=[
            extraction_response(
                symptoms=[{"name": "سعال"}, {"name": "صداع"}],
                negated=[],  # the simulated miss: LLM reports no negation at all
            )
        ]
    )
    set_provider(provider)

    result = extract_symptoms(_state([_user(message)]))

    confirmed_names = {s["name"] for s in result["symptoms"]}
    negated_names = {n["name"] for n in result["negated_symptoms"]}

    # The rule layer's recovery: present despite the LLM never reporting it.
    assert "حمى" in negated_names
    # The rule-only entry carries no raw_mention — the deterministic layer
    # detects negated canonical wording, it does not capture free text.
    fever_entry = next(n for n in result["negated_symptoms"] if n["name"] == "حمى")
    assert "raw_mention" not in fever_entry or fever_entry.get("raw_mention") is None
    # The bounded window's precision, proven end to end, not just at the
    # rules/negation.py unit level: neither confirmed symptom bled into
    # negated_symptoms just because a negation cue appeared earlier in
    # the same message.
    assert "سعال" not in negated_names
    assert "صداع" not in negated_names
    assert confirmed_names == {"سعال", "صداع"}


def test_rule_and_llm_negation_of_the_same_symptom_does_not_duplicate():
    # Both layers independently flagging the same symptom (the common
    # case, not the recovery case above) must union to ONE entry, not
    # two — rules.negation.merge_negations's own dedup responsibility,
    # confirmed here at the node level where the entries actually get
    # assembled, not just against the bare frozenset merge_negations
    # returns in isolation (tests/unit/test_rules_negation.py already
    # covers that).
    message = "ما عندي حمى"
    provider = FakeProvider(
        responses=[
            extraction_response(
                symptoms=[],
                negated=[{"name": "حمى", "raw_mention": "ما عندي حمى"}],
            )
        ]
    )
    set_provider(provider)

    result = extract_symptoms(_state([_user(message)]))

    assert len(result["negated_symptoms"]) == 1
    assert result["negated_symptoms"][0]["name"] == "حمى"
    # The LLM's own entry — with its raw_mention — wins over a bare
    # rule-only duplicate; nothing here should overwrite it with a
    # detail-poorer version.
    assert result["negated_symptoms"][0]["raw_mention"] == "ما عندي حمى"


def test_extract_symptoms_audit_logs_both_negation_verdicts_separately(monkeypatch):
    # Same pattern as
    # test_nodes_check_red_flags.py::test_check_red_flags_audit_logs_both_verdicts_separately
    # — replaces the real audit call with a spy so both layers' verdicts,
    # not just the merged result, are provably recorded side by side.
    logged = []
    monkeypatch.setattr(
        "nodes.extract_symptoms.log_negation_detection",
        lambda **kwargs: logged.append(kwargs),
    )

    message = "ما عندي حمى بس عندي سعال وصداع"
    provider = FakeProvider(
        responses=[
            extraction_response(
                symptoms=[{"name": "سعال"}, {"name": "صداع"}],
                negated=[],
            )
        ]
    )
    set_provider(provider)

    extract_symptoms(_state([_user(message)], thread_id="thread-negation-42"))

    assert len(logged) == 1
    record = logged[0]
    assert record["thread_id"] == "thread-negation-42"
    assert record["rule_negated"] == ["حمى"]
    assert record["llm_negated"] == []
    assert record["combined"] == ["حمى"]


# --- duration: free text normalized to a day count, alongside the original -------


def test_real_end_to_end_free_text_duration_is_normalized_to_duration_days():
    # vocabulary.duration.parse_duration_days wired into extract_symptoms
    # (see nodes/extract_symptoms.py's own module docstring, "Duration"
    # section) — proven here at the real node level, not just against
    # parse_duration_days in isolation (tests/unit/test_vocabulary_duration.py
    # already covers that).
    #
    # "من 5 أيام" -> normalize() -> "من 5 ايام", matched by _COUNTED_DURATION
    # (digit + unit word, optional "من"/"منذ" prefix) -> 5 * 1 day = 5.
    # Verified directly against vocabulary.duration.parse_duration_days
    # before writing this test, not assumed.
    message = "عندي سعال من 5 أيام"
    provider = FakeProvider(
        responses=[
            extraction_response(
                symptoms=[{"name": "سعال", "duration": "من 5 أيام"}],
            )
        ]
    )
    set_provider(provider)

    result = extract_symptoms(_state([_user(message)]))

    cough = next(s for s in result["symptoms"] if s["name"] == "سعال")
    # The patient's own free-text wording is preserved unchanged...
    assert cough["duration"] == "من 5 أيام"
    # ...alongside the new normalized field, never replacing it.
    assert cough["duration_days"] == 5


def test_a_duration_the_parser_cannot_read_yields_duration_days_none_not_zero():
    # parse_duration_days's own contract: None means "not stated /
    # unparseable", never "assume zero" (vocabulary/duration.py's own
    # docstring). "بيجي وبيروح" ("comes and goes") is real patient
    # phrasing with no extractable count — confirmed directly against
    # parse_duration_days before writing this test that it returns None,
    # not 0, for this exact string.
    provider = FakeProvider(
        responses=[
            extraction_response(
                symptoms=[{"name": "سعال", "duration": "بيجي وبيروح"}],
            )
        ]
    )
    set_provider(provider)

    result = extract_symptoms(_state([_user("عندي سعال بيجي وبيروح")]))

    cough = next(s for s in result["symptoms"] if s["name"] == "سعال")
    assert cough["duration"] == "بيجي وبيروح"
    assert cough["duration_days"] is None


# --- no clear symptoms: empty, not an error --------------------------------------


def test_extract_symptoms_returns_empty_lists_for_a_message_with_no_symptoms():
    provider = FakeProvider(responses=[extraction_response()])
    set_provider(provider)

    result = extract_symptoms(_state([_user("مرحبا كيفك")]))

    assert result == {"symptoms": [], "negated_symptoms": [], "unmatched_mentions": []}


# --- a real symptom outside the vocabulary: unmatched_mentions, not lost --------


def test_out_of_vocabulary_symptom_lands_in_unmatched_mentions_not_dropped():
    # The reported bug: "طنين بالأذن" (tinnitus) isn't in the vocabulary,
    # so the model correctly emits no symptoms/negated_symptoms entry for
    # it rather than forcing a near-match name — but the mention itself
    # must survive, not vanish.
    provider = FakeProvider(
        responses=[extraction_response(unmatched=["طنين بالأذن"])]
    )
    set_provider(provider)

    result = extract_symptoms(_state([_user("عندي طنين بالأذن من كم يوم")]))

    assert result["symptoms"] == []
    assert result["negated_symptoms"] == []
    assert result["unmatched_mentions"] == ["طنين بالأذن"]


# --- conversational context: previous_question + known_symptoms ----------------
#
# CONFIRMED BUG this section guards: extract_symptoms used to see ONLY the
# latest message, with no way to know a bare reply ("من الصبح ومستمر")
# was answering a specific follow-up question about a specific symptom.
# Verified directly against the real model that the identical reply
# attaches correctly to the existing symptom once previous_question and
# known_symptoms both reach the prompt, and is silently dropped (not
# even landing in unmatched_mentions) with neither — see
# nodes/extract_symptoms.py's own module docstring for the full story.


def test_prompt_includes_the_previous_assistant_question_when_one_exists():
    provider = FakeProvider(responses=[extraction_response()])
    set_provider(provider)

    extract_symptoms(
        _state([_user("عندي صداع"), _assistant("منذ متى بدأ هذا الصداع؟"), _user("من الصبح")])
    )

    prompt = provider.calls[0]["prompt"]
    assert "منذ متى بدأ هذا الصداع؟" in prompt


def test_prompt_uses_the_placeholder_when_there_is_no_previous_question():
    # First turn of a thread — no assistant message exists yet. Must not
    # raise (previous_assistant_message returning None), and must not
    # leave an unresolved $previous_question token in the prompt either.
    provider = FakeProvider(responses=[extraction_response()])
    set_provider(provider)

    extract_symptoms(_state([_user("عندي صداع")]))

    prompt = provider.calls[0]["prompt"]
    assert "لا يوجد" in prompt


def test_prompt_includes_the_currently_known_symptoms():
    provider = FakeProvider(responses=[extraction_response()])
    set_provider(provider)

    extract_symptoms(
        _state(
            [_user("عندي صداع"), _assistant("منذ متى؟"), _user("من الصبح")],
            symptoms=[{"name": NAME_A, "duration": None, "severity": None, "onset": None}],
        )
    )

    prompt = provider.calls[0]["prompt"]
    assert NAME_A in prompt


def test_real_end_to_end_a_bare_follow_up_reply_attaches_to_the_existing_symptom():
    # The exact reported failure, pinned down as a permanent regression
    # test: "عندي صداع" -> "منذ متى بدأ هذا الصداع؟" -> "من الصبح ومستمر"
    # (a bare reply that never restates "صداع"). The scripted response
    # below is not invented — it is the real shape verified against the
    # actual model for this exact exchange (see the module docstring's
    # "CONFIRMED BUG" note and the manual verification this fix required).
    #
    # This is a FakeProvider test, so it cannot catch a regression in the
    # MODEL's own interpretation — only a real call can (CLAUDE.md's own
    # manual-script convention exists for exactly that reason; see
    # scripts/try_full_chain_manually.py). What this guards is the
    # mechanism the fix depends on: that previous_question and
    # known_symptoms actually reach the prompt, and that a correctly-
    # shaped response genuinely merges onto the EXISTING entry rather
    # than a fresh one — regressing either would silently reopen the bug
    # even if the model's own behavior stayed correct.
    existing_symptom = {
        "name": "صداع",
        "raw_mention": "صداع",
        "duration": None,
        "severity": None,
        "onset": None,
    }
    provider = FakeProvider(
        responses=[
            extraction_response(
                symptoms=[
                    {
                        "name": "صداع",
                        "raw_mention": "صداع",
                        "duration": "من الصبح ومستمر",
                    }
                ]
            )
        ]
    )
    set_provider(provider)

    state = _state(
        [
            _user("عندي صداع"),
            _assistant("منذ متى بدأ هذا الصداع؟"),
            _user("من الصبح ومستمر"),
        ],
        symptoms=[existing_symptom],
    )

    result = extract_symptoms(state)
    merged = merge_symptoms(state["symptoms"], result["symptoms"])

    # The critical assertion: ONE entry, enriched — not a second entry,
    # and not the duration silently missing from both.
    assert len(merged) == 1
    assert merged[0]["name"] == "صداع"
    assert merged[0]["duration"] == "من الصبح ومستمر"


# --- node contract: partial state dict, quality tier -----------------------------


def test_extract_symptoms_returns_only_the_three_keys():
    provider = FakeProvider(responses=[extraction_response()])
    set_provider(provider)

    result = extract_symptoms(_state([_user("...")]))

    assert set(result) == {"symptoms", "negated_symptoms", "unmatched_mentions"}


def test_extract_symptoms_calls_the_llm_on_the_quality_tier_with_its_schema():
    provider = FakeProvider(responses=[extraction_response()])
    set_provider(provider)

    extract_symptoms(_state([_user("...")]))

    assert provider.calls[0]["model"] == "fake-quality-model"
    assert provider.calls[0]["schema"] is SymptomExtraction


def test_extract_symptoms_prompt_bounds_context_to_one_prior_turn():
    # The patient's latest message plus the single immediately-preceding
    # assistant message (previous_question) reach the prompt — CLAUDE.md
    # > working notes: this is deliberately ONE turn of context, not the
    # full conversation history. An older user message from two turns
    # back must not leak in just because it's in state["messages"].
    provider = FakeProvider(responses=[extraction_response()])
    set_provider(provider)

    extract_symptoms(
        _state([_user("رسالة قديمة"), _assistant("سؤال"), _user("رسالة جديدة")])
    )

    prompt = provider.calls[0]["prompt"]
    assert "رسالة جديدة" in prompt
    assert "سؤال" in prompt
    assert "رسالة قديمة" not in prompt


# --- malformed output: the existing audit path, not a new one --------------------


def test_a_duplicate_name_in_one_response_is_collapsed_via_the_schemas_own_audit(
    monkeypatch,
):
    import schemas.symptoms as symptoms_schema

    logged = []
    monkeypatch.setattr(symptoms_schema, "log_malformed_output", lambda **kw: logged.append(kw))

    provider = FakeProvider(
        responses=[
            extraction_response(
                symptoms=[{"name": NAME_C, "duration": "يوم"}, {"name": NAME_C}]
            )
        ]
    )
    set_provider(provider)

    result = extract_symptoms(_state([_user("...")]))

    # Collapsed to one entry by SymptomExtraction's own validator — the
    # node did not need to do anything about the duplicate itself.
    assert len(result["symptoms"]) == 1
    assert result["symptoms"][0]["duration"] == "يوم"
    assert len(logged) == 1
    assert logged[0]["reason"] == "duplicate_symptom_name:symptoms"
