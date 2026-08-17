"""Tests for the adjudication stage.

Nothing here touches the network and nothing needs an API key. A fake client
stands in for the library, which is why the adjudicator takes one through
its constructor.

Most of these tests are about refusal rather than about success, and that
weighting is the point. Applying a sensible edit is the easy half. The half
worth testing is what happens when the model does something a helpful model
would do: edit a word nobody asked about, suggest a word nobody said,
quietly reword a span into something that no longer fits the audio, or pick
whichever amount of money reads better. Each of those has a test below, each
one drives a fake client returning exactly that answer, and each one proves
the same thing: the words did not move.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from types import SimpleNamespace

import pytest

from audio_transcriber.transcription import adjudication as adjudication_module
from audio_transcriber.transcription.adjudication import (
    ADJUDICATION_PURPOSE,
    ADJUDICATION_SCHEMA,
    Adjudicator,
    Dispute,
    EditAction,
    RefusalReason,
    build_prompt,
)
from audio_transcriber.transcription.model import (
    AlignmentStatus,
    Candidate,
    Confidence,
    FinalToken,
    Language,
    LanguageEvidence,
    Provider,
    ProviderToken,
    ReviewReason,
    RiskCategory,
    TimingStatus,
)
from audio_transcriber.transcription.normalise import normalise

#: Obviously not real, but distinctive, so a test can prove it appears
#: nowhere in what gets written to provenance.
API_KEY = "sk-adjudication-secret-do-not-record"

#: What the user typed into Settings, which is an alias rather than a
#: snapshot. What comes back names the snapshot the alias resolved to.
CONFIGURED_MODEL = "gpt-5.6-sol"
RESOLVED_MODEL = "gpt-5.6-sol-2026-07-14"


# -- The fake client -----------------------------------------------------


class FakeResponses:
    def __init__(self, answer: str, error: Exception | None, model: str) -> None:
        self.answer = answer
        self.error = error
        self.model = model
        self.calls: list[dict] = []

    def parse(self, **arguments):
        self.calls.append(arguments)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(model=self.model, id="resp_123", output_text=self.answer)


class FakeClient:
    def __init__(
        self,
        answer: str = "",
        error: Exception | None = None,
        model: str = RESOLVED_MODEL,
    ) -> None:
        self.responses = FakeResponses(answer, error, model)

    @property
    def calls(self) -> list[dict]:
        return self.responses.calls


def answer(*decisions: dict) -> str:
    """The JSON the strict schema would have made the model produce."""
    return json.dumps({"decisions": list(decisions)})


def decision(
    span_id: str,
    action: str,
    from_text: str,
    to_text: str,
    reason: str = "The evidence points this way.",
) -> dict:
    return {
        "span_id": span_id,
        "action": action,
        "from_text": from_text,
        "to_text": to_text,
        "reason": reason,
    }


# -- Building a transcript to argue about --------------------------------


def word(
    text: str,
    *,
    identifier: str,
    start: float = 12.0,
    end: float = 12.4,
    speaker: str = "speaker_0",
    candidates: tuple[tuple[str, Provider], ...] = (),
    risk: tuple[RiskCategory, ...] = (),
) -> FinalToken:
    """One unresolved word, timed and attributed by the backbone service."""
    return FinalToken(
        id=identifier,
        text=text,
        normalised_text=normalise(text),
        text_source=Provider.ELEVENLABS,
        text_confidence=Confidence.UNRESOLVED,
        start=start,
        end=end,
        timing_source=Provider.ELEVENLABS,
        timing_status=TimingStatus.EXACT_PROVIDER_TIME,
        timing_confidence=Confidence.HIGH,
        speaker=speaker,
        speaker_source=Provider.ELEVENLABS,
        speaker_confidence=Confidence.HIGH,
        language=Language.ENGLISH,
        language_evidence=LanguageEvidence(scores={Language.ENGLISH: 0.92}),
        alignment_status=AlignmentStatus.SUBSTITUTION,
        candidates=[
            Candidate(text=candidate, providers=(provider,), score=0.5)
            for candidate, provider in candidates
        ],
        risk_categories=list(risk),
    )


def identity_of(token: FinalToken) -> tuple:
    """Everything about a word that this stage must never touch."""
    return (
        token.id,
        token.start,
        token.end,
        token.speaker,
        token.speaker_source,
        token.timing_source,
        token.timing_status,
        token.timing_confidence,
    )


@pytest.fixture
def name_dispute() -> Dispute:
    """Two words a service heard as a name and another heard differently."""
    tokens = (
        word(
            "Jurgen",
            identifier="tok-101",
            start=61.2,
            end=61.6,
            candidates=(("Jurgen", Provider.ELEVENLABS), ("Jorgen", Provider.OPENAI)),
        ),
        word(
            "Miller",
            identifier="tok-102",
            start=61.6,
            end=62.1,
            candidates=(("Miller", Provider.ELEVENLABS), ("Muller", Provider.OPENAI)),
        ),
    )
    return Dispute(
        tokens=tokens,
        backbone_tokens=(
            ProviderToken(
                provider=Provider.ELEVENLABS,
                index=340,
                text="Jurgen",
                start=61.2,
                end=61.6,
                speaker="speaker_0",
                log_probability=-1.84,
            ),
        ),
        preceding_text="and then I spoke to",
        following_text="about the contract",
        note="The services disagree about a proper name.",
    )


@pytest.fixture
def money_dispute() -> Dispute:
    """The example from the specification: one amount, two readings."""
    return Dispute(
        tokens=(
            word(
                "15,000",
                identifier="tok-500",
                candidates=(
                    ("15,000", Provider.ELEVENLABS),
                    ("50,000", Provider.OPENAI),
                    ("50,000", Provider.MICROSOFT),
                ),
                risk=(RiskCategory.MONEY,),
            ),
        ),
        preceding_text="the invoice came to",
        following_text="rand in total",
    )


@pytest.fixture
def plain_dispute() -> Dispute:
    """One ordinary word, where a model's judgement is welcome."""
    return Dispute(
        tokens=(
            word(
                "affect",
                identifier="tok-200",
                candidates=(("affect", Provider.ELEVENLABS), ("effect", Provider.OPENAI)),
            ),
        ),
        preceding_text="that will have no",
        following_text="on the schedule",
    )


def build_adjudicator(client=None, **overrides) -> Adjudicator:
    arguments = {"api_key": API_KEY, "model": CONFIGURED_MODEL, "client": client}
    arguments.update(overrides)
    return Adjudicator(**arguments)


# -- Choosing well -------------------------------------------------------


def test_an_offered_reading_is_chosen_and_applied(plain_dispute):
    """The whole point of the stage, when it goes right."""
    client = FakeClient(
        answer(decision("tok-200", "replace", "affect", "effect", "It is a noun here."))
    )
    outcome = build_adjudicator(client).adjudicate([plain_dispute])

    assert outcome.succeeded
    assert plain_dispute.tokens[0].text == "effect"
    assert plain_dispute.tokens[0].normalised_text == "effect"
    assert outcome.applied[0].action is EditAction.REPLACE
    assert outcome.applied[0].to_text == "effect"
    assert outcome.refused == ()


def test_a_chosen_word_is_never_marked_as_settled(plain_dispute):
    """A model weighed the evidence. That is worth more than nothing and less
    than a person, so the word stays in the review queue."""
    client = FakeClient(answer(decision("tok-200", "replace", "affect", "effect")))
    build_adjudicator(client).adjudicate([plain_dispute])

    token = plain_dispute.tokens[0]
    assert token.text_confidence is Confidence.REVIEW_SUGGESTED
    assert token.text_confidence is not Confidence.HIGH
    assert "effect" in (token.llm_decision or "")


def test_keeping_the_current_reading_is_a_decision_too(plain_dispute):
    client = FakeClient(answer(decision("tok-200", "keep", "affect", "affect")))
    outcome = build_adjudicator(client).adjudicate([plain_dispute])

    assert plain_dispute.tokens[0].text == "affect"
    assert outcome.applied[0].action is EditAction.KEEP
    assert plain_dispute.tokens[0].text_confidence is Confidence.REVIEW_SUGGESTED


def test_a_word_from_the_vocabulary_may_replace_a_span(name_dispute):
    """The specification's own example: two words nobody offered together,
    both of which are in the list of known names."""
    client = FakeClient(
        answer(decision("tok-101", "replace", "Jurgen Miller", "Jürgen Müller"))
    )
    outcome = build_adjudicator(client).adjudicate(
        [name_dispute], vocabulary_terms=("Jürgen", "Müller")
    )

    assert [token.text for token in name_dispute.tokens] == ["Jürgen", "Müller"]
    assert outcome.refused == ()


# -- Refusal 1: a word it was not asked about ----------------------------


def test_a_span_that_was_never_in_dispute_is_out_of_bounds(plain_dispute):
    """A real word, elsewhere in the transcript, that nobody asked about."""
    elsewhere = word("numbers", identifier="tok-999")
    before = (elsewhere.text, identity_of(elsewhere))
    client = FakeClient(
        answer(decision("tok-999", "replace", "numbers", "figures", "It reads better."))
    )

    outcome = build_adjudicator(client).adjudicate([plain_dispute])

    assert (elsewhere.text, identity_of(elsewhere)) == before
    assert outcome.refused[0].refusal is RefusalReason.SPAN_NOT_IN_DISPUTE
    # The dispute it was actually asked about got no answer, so it waits for
    # a person rather than looking settled.
    assert outcome.unanswered == ("tok-200",)
    assert plain_dispute.tokens[0].text == "affect"


def test_a_span_identifier_that_does_not_exist_at_all_is_refused(plain_dispute):
    client = FakeClient(answer(decision("tok-invented", "replace", "affect", "effect")))
    outcome = build_adjudicator(client).adjudicate([plain_dispute])

    assert outcome.refused[0].refusal is RefusalReason.SPAN_NOT_IN_DISPUTE
    assert plain_dispute.tokens[0].text == "affect"


def test_misquoting_the_span_it_is_editing_is_refused(plain_dispute):
    """A model that has misread which words it is looking at has reached its
    conclusion about some other words."""
    client = FakeClient(
        answer(decision("tok-200", "replace", "the schedule", "effect"))
    )
    outcome = build_adjudicator(client).adjudicate([plain_dispute])

    assert outcome.refused[0].refusal is RefusalReason.WRONG_ORIGINAL_TEXT
    assert plain_dispute.tokens[0].text == "affect"


# -- Refusal 2: a word nobody said ---------------------------------------


def test_a_word_invented_from_nothing_is_refused(plain_dispute):
    """Not offered by any service, not in the vocabulary, and plausible. That
    combination is the failure this whole module exists to prevent."""
    client = FakeClient(
        answer(decision("tok-200", "replace", "affect", "impact", "It sounds natural."))
    )
    outcome = build_adjudicator(client).adjudicate([plain_dispute])

    assert outcome.refused[0].refusal is RefusalReason.TEXT_NOT_OFFERED
    assert plain_dispute.tokens[0].text == "affect"
    assert plain_dispute.tokens[0].text_confidence is Confidence.UNRESOLVED
    assert ReviewReason.ESCALATION_UNRESOLVED in plain_dispute.tokens[0].review_reasons


def test_half_an_invented_name_is_still_an_invented_name(name_dispute):
    """One word from the vocabulary and one from nowhere is still from nowhere."""
    client = FakeClient(
        answer(decision("tok-101", "replace", "Jurgen Miller", "Jürgen Schmidt"))
    )
    outcome = build_adjudicator(client).adjudicate(
        [name_dispute], vocabulary_terms=("Jürgen", "Müller")
    )

    assert outcome.refused[0].refusal is RefusalReason.TEXT_NOT_OFFERED
    assert [token.text for token in name_dispute.tokens] == ["Jurgen", "Miller"]


def test_a_keep_may_not_smuggle_different_text_in_with_it(plain_dispute):
    client = FakeClient(answer(decision("tok-200", "keep", "affect", "effect")))
    outcome = build_adjudicator(client).adjudicate([plain_dispute])

    assert outcome.refused[0].refusal is RefusalReason.TEXT_NOT_OFFERED
    assert plain_dispute.tokens[0].text == "affect"


# -- Refusal 3: nothing but the text may move ----------------------------


def test_the_times_the_speakers_and_the_identifiers_survive_a_real_edit(name_dispute):
    """Adjudication decides what was said. It decides nothing else."""
    before = [identity_of(token) for token in name_dispute.tokens]
    client = FakeClient(
        answer(decision("tok-101", "replace", "Jurgen Miller", "Jürgen Müller"))
    )

    build_adjudicator(client).adjudicate([name_dispute], vocabulary_terms=("Jürgen", "Müller"))

    assert [identity_of(token) for token in name_dispute.tokens] == before


def test_a_replacement_that_changes_the_word_count_is_refused(name_dispute):
    """Two spans cannot become one without somebody deciding where the join
    now falls, and that decision belongs to the audio."""
    client = FakeClient(
        answer(decision("tok-101", "replace", "Jurgen Miller", "Jürgen"))
    )
    outcome = build_adjudicator(client).adjudicate(
        [name_dispute], vocabulary_terms=("Jürgen", "Müller")
    )

    assert outcome.refused[0].refusal is RefusalReason.WORD_COUNT_CHANGED
    assert [token.text for token in name_dispute.tokens] == ["Jurgen", "Miller"]
    assert all(token.start is not None for token in name_dispute.tokens)


def test_the_schema_gives_the_model_nowhere_to_put_a_timestamp():
    """It cannot ask to move a time, because there is no field for one."""
    fields = ADJUDICATION_SCHEMA["properties"]["decisions"]["items"]["properties"]

    assert set(fields) == {"span_id", "action", "from_text", "to_text", "reason"}
    assert not {"start", "end", "speaker", "timestamp", "token_id"} & set(fields)


# -- Refusal 4: a value that must not be guessed -------------------------


def test_money_is_never_settled_by_which_reading_is_more_plausible(money_dispute):
    """Two providers against one is real evidence, and it is still not enough
    to change an amount without a person seeing it."""
    client = FakeClient(
        answer(
            decision(
                "tok-500",
                "replace",
                "15,000",
                "50,000",
                "Two of the three services heard fifty.",
            )
        )
    )
    outcome = build_adjudicator(client).adjudicate([money_dispute])

    token = money_dispute.tokens[0]
    assert token.text == "15,000"
    assert token.text_confidence is Confidence.UNRESOLVED
    assert ReviewReason.HIGH_RISK_ENTITY in token.review_reasons
    assert outcome.refused[0].refusal is RefusalReason.HIGH_RISK_ENTITY
    # The opinion is worth reading; it is simply not worth applying.
    assert "fifty" in (token.llm_decision or "")
    assert token.is_uncertain_between() == ("15,000", "50,000")


@pytest.mark.parametrize(
    "category",
    [
        RiskCategory.MONEY,
        RiskCategory.DATE,
        RiskCategory.TIME,
        RiskCategory.ACCOUNT_NUMBER,
        RiskCategory.TELEPHONE,
        RiskCategory.MEDICAL_MEASUREMENT,
    ],
)
def test_every_high_risk_category_is_left_for_a_person(category):
    dispute = Dispute(
        tokens=(
            word(
                "eleven",
                identifier="tok-700",
                candidates=(("eleven", Provider.ELEVENLABS), ("seven", Provider.OPENAI)),
                risk=(category,),
            ),
        )
    )
    client = FakeClient(answer(decision("tok-700", "replace", "eleven", "seven")))

    outcome = build_adjudicator(client).adjudicate([dispute])

    assert dispute.tokens[0].text == "eleven"
    assert outcome.refused[0].refusal is RefusalReason.HIGH_RISK_ENTITY


def test_a_high_risk_span_may_still_be_confirmed_as_it_stands(money_dispute):
    """Refusing to change it is not the same as refusing to look at it."""
    client = FakeClient(answer(decision("tok-500", "keep", "15,000", "15,000")))
    outcome = build_adjudicator(client).adjudicate([money_dispute])

    assert money_dispute.tokens[0].text == "15,000"
    assert outcome.applied[0].action is EditAction.KEEP


# -- Refusal 5: declining, which is a good answer ------------------------


def test_declining_to_decide_is_recorded_and_welcomed(plain_dispute):
    client = FakeClient(
        answer(
            decision(
                "tok-200",
                "decline",
                "affect",
                "affect",
                "Both readings fit the sentence equally.",
            )
        )
    )
    outcome = build_adjudicator(client).adjudicate([plain_dispute])

    token = plain_dispute.tokens[0]
    assert outcome.succeeded
    assert outcome.declined == ("tok-200",)
    assert token.text == "affect"
    assert token.text_confidence is Confidence.UNRESOLVED
    assert ReviewReason.ADJUDICATION_DECLINED in token.review_reasons
    assert "Both readings" in (token.llm_decision or "")


def test_saying_nothing_about_a_span_counts_as_declining(plain_dispute, money_dispute):
    """Silence must not leave a word looking settled."""
    client = FakeClient(answer(decision("tok-200", "keep", "affect", "affect")))
    outcome = build_adjudicator(client).adjudicate([plain_dispute, money_dispute])

    assert outcome.unanswered == ("tok-500",)
    assert ReviewReason.ADJUDICATION_DECLINED in money_dispute.tokens[0].review_reasons
    assert money_dispute.tokens[0].text_confidence is Confidence.UNRESOLVED


def test_deciding_the_same_span_twice_is_refused(plain_dispute):
    """Taking the later of two answers would make the result depend on order."""
    client = FakeClient(
        answer(
            decision("tok-200", "keep", "affect", "affect"),
            decision("tok-200", "replace", "affect", "effect"),
        )
    )
    outcome = build_adjudicator(client).adjudicate([plain_dispute])

    assert plain_dispute.tokens[0].text == "affect"
    assert outcome.refused[0].refusal is RefusalReason.SPAN_NOT_IN_DISPUTE


# -- What gets sent ------------------------------------------------------


def test_the_reasoning_effort_is_left_off_when_nobody_chose_one(plain_dispute):
    """Blank means blank. Sending a default nobody asked for is how a request
    starts failing after the model name changes."""
    client = FakeClient(answer(decision("tok-200", "keep", "affect", "affect")))
    build_adjudicator(client, reasoning_effort="").adjudicate([plain_dispute])

    assert "reasoning" not in client.calls[0]


def test_the_reasoning_effort_is_nested_the_way_the_responses_api_wants_it(plain_dispute):
    """Flat is the Chat Completions shape. Sent here it would do nothing at all."""
    client = FakeClient(answer(decision("tok-200", "keep", "affect", "affect")))
    build_adjudicator(client, reasoning_effort="low").adjudicate([plain_dispute])

    call = client.calls[0]
    assert call["reasoning"] == {"effort": "low"}
    assert "reasoning_effort" not in call


def test_an_unfamiliar_effort_is_passed_through_rather_than_dropped(plain_dispute):
    """The model name is editable text, so a newer effort must reach the API
    and be answered there rather than being silently discarded here."""
    client = FakeClient(answer(decision("tok-200", "keep", "affect", "affect")))
    build_adjudicator(client, reasoning_effort="xhigh").adjudicate([plain_dispute])

    assert client.calls[0]["reasoning"] == {"effort": "xhigh"}


def test_the_answer_is_constrained_by_a_strict_schema(plain_dispute):
    client = FakeClient(answer(decision("tok-200", "keep", "affect", "affect")))
    build_adjudicator(client).adjudicate([plain_dispute])

    text_format = client.calls[0]["text"]["format"]
    assert text_format["type"] == "json_schema"
    assert text_format["strict"] is True
    assert text_format["schema"] is ADJUDICATION_SCHEMA
    assert client.calls[0]["model"] == CONFIGURED_MODEL


def test_free_form_settings_cannot_switch_off_the_schema(plain_dispute):
    """The structured answer is what makes the reply safe to apply, so a
    stray parameter must not be able to remove it."""
    client = FakeClient(answer(decision("tok-200", "keep", "affect", "affect")))
    build_adjudicator(
        client, parameters={"text": {"format": {"type": "text"}}, "top_p": 0.4}
    ).adjudicate([plain_dispute])

    call = client.calls[0]
    assert call["text"]["format"]["type"] == "json_schema"
    assert call["top_p"] == 0.4


def test_the_prompt_carries_the_evidence_and_not_the_recording(name_dispute):
    prompt = build_prompt(
        [name_dispute],
        recording_context="A call about a supply contract.",
        vocabulary_terms=("Jürgen", "Müller"),
        languages=(Language.ENGLISH, Language.GERMAN),
    )

    # Everything section 17 asks for about this region.
    assert "tok-101" in prompt
    assert "'Jorgen'" in prompt and "'Muller'" in prompt
    assert "61.20-61.60s" in prompt
    assert "speaker speaker_0" in prompt
    assert "log probability -1.840" in prompt
    assert "Jürgen, Müller" in prompt
    assert "and then I spoke to" in prompt
    assert "about the contract" in prompt
    assert "English, German" in prompt
    assert "substitution" in prompt
    assert "OpenAI" in prompt and "ElevenLabs Scribe" in prompt
    # And nothing from the rest of the recording.
    assert "supply contract" in prompt
    assert len(prompt) < 2000


def test_a_high_risk_span_says_so_in_the_prompt_as_well(money_dispute):
    """The prompt asks. The validation guarantees. Both are worth having."""
    prompt = build_prompt([money_dispute])

    assert "HIGH RISK" in prompt
    assert "Money" in prompt


# -- Provenance ----------------------------------------------------------


def test_the_resolved_snapshot_is_what_reaches_provenance(plain_dispute):
    """Settings hold an alias. Only the response knows what it resolved to."""
    client = FakeClient(answer(decision("tok-200", "keep", "affect", "affect")))
    outcome = build_adjudicator(client, reasoning_effort="medium").adjudicate([plain_dispute])

    record = outcome.request
    assert record is not None
    assert record.provider is Provider.OPENAI
    assert record.model_identifier == RESOLVED_MODEL
    assert record.model_version == RESOLVED_MODEL
    assert record.request_parameters["model"] == CONFIGURED_MODEL
    assert record.request_parameters["reasoning"] == {"effort": "medium"}
    assert record.request_parameters["disputed_span_count"] == 1
    assert record.request_parameters["response_id"] == "resp_123"
    assert record.purpose == ADJUDICATION_PURPOSE
    assert record.succeeded is True


def test_the_record_says_plainly_when_no_reasoning_was_configured(plain_dispute):
    client = FakeClient(answer(decision("tok-200", "keep", "affect", "affect")))
    outcome = build_adjudicator(client).adjudicate([plain_dispute])

    assert outcome.request.request_parameters["reasoning"] == "not configured"


def test_no_credential_and_no_recording_content_ever_reaches_the_record(plain_dispute):
    client = FakeClient(answer(decision("tok-200", "keep", "affect", "affect")))
    outcome = build_adjudicator(
        client, parameters={"api_key": API_KEY, "authorization": API_KEY, "top_p": 0.2}
    ).adjudicate([plain_dispute])

    written = repr(outcome.request)
    assert API_KEY not in written
    parameters = outcome.request.request_parameters
    assert "api_key" not in parameters
    assert "authorization" not in parameters
    assert parameters["top_p"] == 0.2
    # The prompt holds what people said. Provenance is read by people who
    # have no business reading the transcript twice over.
    assert "input" not in parameters
    assert "instructions" not in parameters
    assert "affect" not in written


def test_a_failed_request_is_still_recorded(plain_dispute):
    client = FakeClient(error=RuntimeError("Rate limit reached"))
    client.responses.error.status_code = 429
    outcome = build_adjudicator(client).adjudicate([plain_dispute])

    record = outcome.request
    assert record is not None
    assert record.succeeded is False
    assert record.purpose == ADJUDICATION_PURPOSE
    assert record.model_identifier == CONFIGURED_MODEL


# -- Failing well --------------------------------------------------------


def test_a_missing_key_leaves_the_words_alone_and_says_why(plain_dispute):
    client = FakeClient(answer(decision("tok-200", "replace", "affect", "effect")))
    adjudicator = build_adjudicator(client, api_key="")

    assert adjudicator.is_configured() is False
    outcome = adjudicator.adjudicate([plain_dispute])

    assert outcome.succeeded is False
    assert "API key" in outcome.error
    assert client.calls == []
    assert plain_dispute.tokens[0].text == "affect"
    assert ReviewReason.ADJUDICATION_DECLINED in plain_dispute.tokens[0].review_reasons


def test_a_missing_model_name_is_reported_before_anything_is_sent(plain_dispute):
    adjudicator = build_adjudicator(FakeClient(), model="")

    assert adjudicator.describe_configuration_problem() is not None
    assert adjudicator.adjudicate([plain_dispute]).succeeded is False


def test_a_missing_package_disables_only_this_stage(plain_dispute, monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    outcome = build_adjudicator(client=None).adjudicate([plain_dispute])

    assert outcome.succeeded is False
    assert "pip install openai" in outcome.error
    assert plain_dispute.tokens[0].text == "affect"
    assert plain_dispute.tokens[0].text_confidence is Confidence.UNRESOLVED


def test_the_module_imports_with_no_package_installed(monkeypatch):
    """Importing must never need the library, or the application cannot start.

    The module source is executed into a module object of its own rather than
    reloaded in place, and that choice is worth explaining, because getting it
    wrong produces one of the most baffling failures in Python.
    ``importlib.reload`` re-runs the source in the *existing* module
    namespace, which rebinds every class in it to a brand-new class object.
    Anything imported earlier - here, every name at the top of this file -
    still points at the old objects, while the module's own functions now
    build the new ones. The two sets look identical when printed, so the next
    test to compare an enum member with ``is`` fails with the impossible
    message "assert RefusalReason.X is RefusalReason.X". Running the source
    into a separate module leaves the real one untouched and proves the same
    thing.
    """
    monkeypatch.setitem(sys.modules, "openai", None)
    specification = importlib.util.spec_from_file_location(
        "adjudication_with_no_openai", adjudication_module.__file__
    )
    fresh = importlib.util.module_from_spec(specification)
    # Registered under its own name while it runs, because dataclasses looks
    # its owner up in sys.modules while it builds each class. monkeypatch
    # takes the entry out again, so nothing survives this test.
    monkeypatch.setitem(sys.modules, specification.name, fresh)
    specification.loader.exec_module(fresh)

    assert fresh.Adjudicator(api_key="k", model="m").is_configured()
    assert fresh._stopped_early_errors() == ()
    # The real module is still the one everything else in this file imported.
    assert adjudication_module.RefusalReason is RefusalReason


def test_an_api_error_becomes_a_graceful_non_change(plain_dispute):
    """Losing this stage costs a little accuracy. It must never cost words."""
    before = (plain_dispute.tokens[0].text, identity_of(plain_dispute.tokens[0]))
    failure = RuntimeError("Rate limit reached for this model")
    failure.status_code = 429
    outcome = build_adjudicator(FakeClient(error=failure)).adjudicate([plain_dispute])

    assert outcome.succeeded is False
    assert "429" in outcome.error
    assert (plain_dispute.tokens[0].text, identity_of(plain_dispute.tokens[0])) == before
    assert ReviewReason.ADJUDICATION_DECLINED in plain_dispute.tokens[0].review_reasons


def test_an_answer_cut_short_is_reported_as_what_it_is(plain_dispute, monkeypatch):
    """Without catching these two by name, running out of output tokens part
    way through a structured answer surfaces as a confusing parse error."""

    class LengthFinishReasonError(Exception):
        pass

    monkeypatch.setattr(
        adjudication_module, "_stopped_early_errors", lambda: (LengthFinishReasonError,)
    )
    client = FakeClient(error=LengthFinishReasonError("ran out of output tokens"))

    outcome = build_adjudicator(client).adjudicate([plain_dispute])

    assert outcome.succeeded is False
    assert "stopped before it finished" in outcome.error
    assert plain_dispute.tokens[0].text == "affect"


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "not json at all",
        json.dumps({"decisions": "keep everything"}),
        json.dumps({"edits": []}),
        json.dumps(["keep"]),
        json.dumps({"decisions": ["keep"]}),
    ],
)
def test_an_answer_of_the_wrong_shape_changes_nothing(reply, plain_dispute):
    """The schema is strict, so this should not happen. "Should not happen" is
    a poor reason to apply an unexamined answer to somebody's transcript."""
    outcome = build_adjudicator(FakeClient(reply)).adjudicate([plain_dispute])

    assert outcome.succeeded is False
    assert outcome.refused[0].refusal is RefusalReason.MALFORMED_ANSWER
    assert plain_dispute.tokens[0].text == "affect"
    assert outcome.request is not None


def test_an_action_that_does_not_exist_is_refused(plain_dispute):
    client = FakeClient(answer(decision("tok-200", "rewrite", "affect", "effect")))
    outcome = build_adjudicator(client).adjudicate([plain_dispute])

    assert outcome.refused[0].refusal is RefusalReason.MALFORMED_ANSWER
    assert plain_dispute.tokens[0].text == "affect"


def test_nothing_is_sent_when_there_is_nothing_to_settle():
    client = FakeClient()
    outcome = build_adjudicator(client).adjudicate([])

    assert outcome.succeeded
    assert client.calls == []
    assert outcome.applied == ()


def test_a_dispute_must_cover_at_least_one_word():
    with pytest.raises(ValueError):
        Dispute(tokens=())
