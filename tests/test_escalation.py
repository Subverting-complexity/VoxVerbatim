"""Tests for deciding what deserves a second opinion, and getting one.

Most of this is arithmetic about time and can be tested with nothing but
data: which disputes share a window, how far the padding reaches, what the
ceiling does when it is reached. Those tests build words and disputes
directly and never touch a file.

One question cannot be answered that way. The two ways of asking a service
about a few seconds of a recording, by naming a time window on an uploaded
file and by cutting a clip and sending it, run through completely different
arithmetic and have to produce identical canonical times. That is tested on
a real recording, with a stand-in for the service that answers in the
timebase of whatever audio it was really given, because a fake that answered
in canonical time whatever it was sent would prove nothing at all.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from vox_verbatim.transcription.canonical import probe_audio
from vox_verbatim.transcription.escalation import (
    Dispute,
    EscalationOptions,
    EscalationReason,
    EscalationResult,
    EscalationWindow,
    EvidenceStrength,
    apply_outcome,
    choose_model,
    escalate,
    find_disputes,
    group_into_windows,
    plan_escalation,
    reasons_for,
    strength_of,
)
from vox_verbatim.transcription.model import (
    AudioSpan,
    Candidate,
    CanonicalAudio,
    Confidence,
    FinalToken,
    Language,
    LanguageEvidence,
    Provider,
    ProviderResult,
    ProviderToken,
    ReviewReason,
    RiskCategory,
    TokenReference,
    Transcript,
)
from vox_verbatim.transcription.providers.assemblyai import (
    ASSEMBLYAI_CAPABILITIES,
    AssemblyAiProvider,
)

from tests.conftest import write_real_audio

#: Short margins keep the numbers in these tests easy to check by hand.
TIGHT = EscalationOptions(context_seconds_before=1.0, context_seconds_after=1.0)


def word(
    text: str,
    start: float,
    end: float,
    candidates: tuple[tuple[str, tuple[Provider, ...]], ...] = (),
    language: Language = Language.UNKNOWN,
    risks: tuple[RiskCategory, ...] = (),
    reasons: tuple[ReviewReason, ...] = (),
    in_vocabulary: str | None = None,
    sources: tuple[TokenReference, ...] = (),
) -> FinalToken:
    """One finished word, with as much or as little evidence as a test needs."""
    token = FinalToken(
        text=text,
        start=start,
        end=end,
        language=language,
        candidates=[
            Candidate(
                text=candidate,
                providers=providers,
                in_vocabulary=candidate == in_vocabulary,
            )
            for candidate, providers in candidates
        ],
        risk_categories=list(risks),
        source_tokens=list(sources),
    )
    for reason in reasons:
        token.flag(reason)
    return token


def dispute(start: float, end: float, *reasons: EscalationReason) -> Dispute:
    return Dispute(
        span=AudioSpan(start, end),
        reasons=reasons or (EscalationReason.ALL_PROVIDERS_DISAGREE,),
    )


# -- Grouping nearby disputes into one window ----------------------------


def test_disputes_a_few_seconds_apart_share_one_window():
    """Three arguments in one sentence are one question, not three.

    They are close enough that their padded windows would overlap anyway, so
    joining them asks for no audio that was not already going to be sent and
    saves two whole requests.
    """
    windows = group_into_windows(
        [dispute(10.0, 10.4), dispute(11.2, 11.6), dispute(12.0, 12.5)],
        duration=60.0,
        options=TIGHT,
    )

    assert len(windows) == 1
    assert len(windows[0].disputes) == 3
    assert windows[0].target == AudioSpan(10.0, 12.5)
    assert windows[0].span == AudioSpan(9.0, 13.5)


def test_a_dispute_on_its_own_gets_a_window_of_its_own():
    windows = group_into_windows(
        [dispute(10.0, 10.4), dispute(40.0, 40.3)], duration=60.0, options=TIGHT
    )

    assert [len(window.disputes) for window in windows] == [1, 1]
    assert windows[0].span == AudioSpan(9.0, 11.4)
    assert windows[1].span == AudioSpan(39.0, 41.3)


def test_a_window_stops_growing_once_it_has_swallowed_enough_speech():
    """Grouping saves money only while the window stays a question.

    A run of disputes spread across a minute would otherwise join into one
    window in which the disputed words are a small part of what the service
    is listening to.
    """
    options = replace(TIGHT, maximum_window_seconds=10.0, join_gap_seconds=5.0)
    disputes = [dispute(start, start + 0.3) for start in (0.0, 4.0, 8.0, 12.0, 16.0)]

    windows = group_into_windows(disputes, duration=60.0, options=options)

    assert len(windows) > 1
    assert all(window.span.duration <= 10.0 for window in windows)
    assert sum(len(window.disputes) for window in windows) == len(disputes)


def test_the_padding_is_clamped_at_both_ends_of_the_recording():
    """A dispute in the first seconds cannot be given audio from before them."""
    options = EscalationOptions(context_seconds_before=5.0, context_seconds_after=5.0)

    first, last = group_into_windows(
        [dispute(0.4, 0.9), dispute(29.2, 29.8)], duration=30.0, options=options
    )

    assert first.span == AudioSpan(0.0, 5.9)
    assert first.clamped is True
    assert last.span == AudioSpan(24.2, 30.0)
    assert last.clamped is True


def test_a_window_in_the_middle_is_not_reported_as_clamped():
    window, = group_into_windows([dispute(10.0, 10.5)], duration=30.0, options=TIGHT)

    assert window.span == AudioSpan(9.0, 11.5)
    assert window.clamped is False


def test_a_window_holding_one_afrikaans_dispute_is_an_afrikaans_window():
    """Afrikaans decides the whole window, because it decides the model."""
    disputes = [
        Dispute(AudioSpan(10.0, 10.4), (EscalationReason.ALL_PROVIDERS_DISAGREE,),
                language=Language.ENGLISH),
        Dispute(AudioSpan(10.8, 11.2), (EscalationReason.LANGUAGE_BOUNDARY_UNCLEAR,),
                language=Language.AFRIKAANS),
    ]

    window, = group_into_windows(disputes, duration=60.0, options=TIGHT)

    assert window.is_afrikaans


# -- Deciding what is in dispute -----------------------------------------


def test_every_service_hearing_something_different_is_escalated():
    token = word(
        "fifteen",
        1.0,
        1.4,
        candidates=(
            ("fifteen", (Provider.ELEVENLABS,)),
            ("fifty", (Provider.OPENAI,)),
            ("sixty", (Provider.MICROSOFT,)),
        ),
    )

    assert EscalationReason.ALL_PROVIDERS_DISAGREE in reasons_for(token)


def test_two_services_agreeing_against_a_third_is_not_everybody_disagreeing():
    """Agreement is evidence the scoring rules can weigh on their own."""
    token = word(
        "fifty",
        1.0,
        1.4,
        candidates=(
            ("fifty", (Provider.OPENAI, Provider.MICROSOFT)),
            ("fifteen", (Provider.ELEVENLABS,)),
        ),
    )

    assert EscalationReason.ALL_PROVIDERS_DISAGREE not in reasons_for(token)


def test_a_disagreement_the_evidence_could_not_settle_is_escalated():
    """Two against one is weighed by the scoring rules. Where that weighing
    still left the word open, a fourth voice is exactly what is missing."""
    token = word(
        "carton",
        1.0,
        1.4,
        candidates=(
            ("carton", (Provider.ELEVENLABS,)),
            ("garden", (Provider.OPENAI, Provider.MICROSOFT)),
        ),
    )
    token.text_confidence = Confidence.UNRESOLVED
    assert EscalationReason.UNSETTLED_DISAGREEMENT in reasons_for(token)

    token.text_confidence = Confidence.HIGH
    assert EscalationReason.UNSETTLED_DISAGREEMENT not in reasons_for(token)


def test_two_spellings_of_the_same_number_are_not_a_disagreement():
    """Escalating these would fill the queue with differences that are not."""
    token = word(
        "25",
        1.0,
        1.4,
        candidates=(
            ("25", (Provider.ELEVENLABS,)),
            ("twenty-five", (Provider.OPENAI,)),
        ),
    )

    assert reasons_for(token) == ()


def test_a_name_that_differs_is_escalated():
    token = word(
        "Jürgen",
        1.0,
        1.4,
        candidates=(("Jürgen", (Provider.ELEVENLABS,)), ("Jergen", (Provider.OPENAI,))),
    )

    assert EscalationReason.PROPER_NOUN_DIFFERS in reasons_for(token)


def test_a_number_that_differs_is_escalated():
    token = word(
        "15,000",
        1.0,
        1.4,
        candidates=(("15,000", (Provider.ELEVENLABS,)), ("50,000", (Provider.OPENAI,))),
    )

    assert EscalationReason.NUMBER_DIFFERS in reasons_for(token)


def test_an_amount_is_escalated_even_where_the_services_agree():
    """A wrong amount reads perfectly well, which is what makes it dangerous."""
    token = word("R15,000", 1.0, 1.4, risks=(RiskCategory.MONEY,))

    assert EscalationReason.NUMBER_DIFFERS in reasons_for(token)


def test_a_known_term_that_differs_is_escalated():
    token = word(
        "Kubernetes",
        1.0,
        1.4,
        candidates=(
            ("Kubernetes", (Provider.ELEVENLABS,)),
            ("cooper netties", (Provider.OPENAI,)),
        ),
        in_vocabulary="Kubernetes",
    )

    assert EscalationReason.TECHNICAL_TERM_DIFFERS in reasons_for(token)


def test_an_unclear_language_is_escalated_only_where_afrikaans_is_enabled():
    """With Afrikaans off there is no code switch to find and no model to route to."""
    token = word("more", 1.0, 1.4)
    token.language_evidence = LanguageEvidence({Language.ENGLISH: 0.5, Language.AFRIKAANS: 0.45})

    assert reasons_for(token, options=EscalationOptions(afrikaans_enabled=False)) == ()
    assert EscalationReason.LANGUAGE_BOUNDARY_UNCLEAR in reasons_for(
        token, options=EscalationOptions(afrikaans_enabled=True)
    )


def test_a_word_the_backbone_was_unsure_of_is_escalated_where_it_is_contested():
    """The probability lives on the evidence, so it is followed back to it."""
    token = word(
        "contract",
        1.0,
        1.4,
        candidates=(("contract", (Provider.ELEVENLABS,)), ("contact", (Provider.OPENAI,))),
        sources=(TokenReference(Provider.ELEVENLABS, 7),),
    )
    results = {
        Provider.ELEVENLABS: ProviderResult(
            provider=Provider.ELEVENLABS,
            tokens=[
                ProviderToken(
                    provider=Provider.ELEVENLABS,
                    index=7,
                    text="contract",
                    start=1.0,
                    end=1.4,
                    log_probability=-1.2,
                )
            ],
        )
    }

    disputes = find_disputes([token], results)

    assert disputes
    assert EscalationReason.LOW_PROBABILITY_BACKBONE_WORD in disputes[0].reasons


def test_a_speaker_or_overlap_doubt_is_escalated():
    token = word(
        "yes",
        1.0,
        1.2,
        reasons=(ReviewReason.SPEAKER_UNCERTAIN, ReviewReason.OVERLAPPING_SPEECH),
    )

    found = reasons_for(token)

    assert EscalationReason.SPEAKER_BOUNDARY_UNCERTAIN in found
    assert EscalationReason.OVERLAPPING_SPEECH in found


def test_a_word_with_no_audio_behind_it_is_left_for_a_person():
    """There is nothing to send, so there is nothing to ask."""
    token = FinalToken(text="the")
    token.candidates = [
        Candidate(text="the", providers=(Provider.OPENAI,)),
        Candidate(text="a", providers=(Provider.MICROSOFT,)),
    ]

    assert find_disputes([token]) == []


def test_planning_finds_the_disputes_and_gathers_them():
    tokens = [
        word(
            "fifteen",
            10.0,
            10.4,
            candidates=(
                ("fifteen", (Provider.ELEVENLABS,)),
                ("fifty", (Provider.OPENAI,)),
                ("sixty", (Provider.MICROSOFT,)),
            ),
        ),
        word("thousand", 10.5, 11.0),
        word(
            "Jürgen",
            11.2,
            11.6,
            candidates=(("Jürgen", (Provider.ELEVENLABS,)), ("Jergen", (Provider.OPENAI,))),
        ),
    ]

    windows = plan_escalation(tokens, duration=60.0, options=TIGHT)

    assert len(windows) == 1
    assert len(windows[0].disputes) == 2


# -- Choosing the model and weighing the answer --------------------------


def test_an_english_window_goes_to_the_strong_model_and_can_settle_things():
    options = EscalationOptions(afrikaans_enabled=True)
    window = EscalationWindow(
        span=AudioSpan(9.0, 12.0),
        target=AudioSpan(10.0, 11.0),
        disputes=(dispute(10.0, 11.0),),
        language=Language.ENGLISH,
    )

    assert choose_model(window, options) == options.primary_model
    assert strength_of(window, options) is EvidenceStrength.DECIDING
    assert strength_of(window, options).confidence_ceiling is Confidence.HIGH


def test_an_afrikaans_window_goes_to_the_older_model_and_only_informs():
    """Its Afrikaans is documented as moderate, so it cannot settle anything.

    The answer arrives in the same shape as an English one, and without this
    distinction it would be weighed as though it were as strong.
    """
    options = EscalationOptions(afrikaans_enabled=True)
    window = EscalationWindow(
        span=AudioSpan(9.0, 12.0),
        target=AudioSpan(10.0, 11.0),
        disputes=(dispute(10.0, 11.0),),
        language=Language.AFRIKAANS,
    )

    assert choose_model(window, options) == options.afrikaans_model
    assert strength_of(window, options) is EvidenceStrength.INFORMING
    assert strength_of(window, options).confidence_ceiling is Confidence.REVIEW_SUGGESTED


def test_afrikaans_switched_off_never_reaches_the_afrikaans_model():
    """A German span must not be sent to it because somebody enabled nothing."""
    options = EscalationOptions(afrikaans_enabled=False)
    window = EscalationWindow(
        span=AudioSpan(9.0, 12.0),
        target=AudioSpan(10.0, 11.0),
        disputes=(dispute(10.0, 11.0),),
        language=Language.AFRIKAANS,
    )

    assert choose_model(window, options) == options.primary_model


# -- Asking, in both of the two ways -------------------------------------


class FakeAssemblyAiClient:
    """A stand-in for the service, answering in the timebase it was given.

    This is the part that makes the two-path test mean anything. Asked about
    a time window of an uploaded recording, it counts from the start of the
    recording, because that is what it is holding. Given a clip, it counts
    from the start of the clip and cannot see anything outside it. A fake
    that answered in canonical time whichever way it was asked would agree
    with itself no matter how wrong the arithmetic under test was.
    """

    def __init__(self, words: tuple[tuple[str, float, float], ...], clip_span=None):
        self.words = words
        self.clip_span = clip_span
        self.uploads: list[str] = []
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def upload_file(self, audio_path: str) -> str:
        self.uploads.append(audio_path)
        return "https://uploads.example/one"

    def transcribe(self, audio: Any, parameters: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((audio, dict(parameters)))
        start = parameters.get("audio_start_from")
        end = parameters.get("audio_end_at")
        if start is not None and end is not None:
            heard, offset = (start / 1000.0, end / 1000.0), 0.0
        elif self.clip_span is not None:
            heard, offset = self.clip_span, self.clip_span[0]
        else:
            heard, offset = (0.0, float("inf")), 0.0
        return {
            "id": "transcript-1",
            "status": "completed",
            "audio_duration": heard[1] - heard[0],
            "speech_model_used": parameters["speech_models"][0],
            "words": [
                {
                    "text": text,
                    "start": round((first - offset) * 1000),
                    "end": round((last - offset) * 1000),
                    "confidence": 0.9,
                    "speaker": "A",
                }
                for text, first, last in self.words
                if first >= heard[0] and last <= heard[1]
            ],
        }


def real_recording(tmp_path: Path, seconds: float = 6.0) -> CanonicalAudio:
    """A real six-second recording, measured as the canonical file."""
    source = tmp_path / "talk.wav"
    write_real_audio(source, level_db=-20.0, seconds=seconds, codec="pcm_s16le",
                     layout="mono", rate=16000)
    probe = probe_audio(source)
    return CanonicalAudio(
        path=str(source),
        original_path=str(source),
        duration=probe.duration,
        sample_rate=probe.sample_rate,
        channels=probe.channels,
        size_bytes=probe.size_bytes,
        container=probe.container,
    )


SPOKEN = (("fifteen", 2.0, 2.4), ("thousand", 2.5, 3.0), ("rand", 3.1, 3.4))


def test_both_ways_of_asking_agree_on_canonical_time(tmp_path):
    """The cheap path and the fallback must produce identical times.

    One asks about a region of a recording the service already holds; the
    other cuts that region out and sends it, and then has to add back where
    the cut began. If those two ever disagreed, a transcript's words would
    point at different moments depending on which service happened to answer,
    and nothing downstream could tell.
    """
    canonical = real_recording(tmp_path)
    window, = group_into_windows([dispute(2.0, 2.4)], duration=canonical.duration, options=TIGHT)
    assert window.span == AudioSpan(1.0, 3.4)

    uploaded = FakeAssemblyAiClient(SPOKEN)
    by_window = escalate(
        [window], AssemblyAiProvider(api_key="k", client=uploaded), canonical, TIGHT
    )

    cut = FakeAssemblyAiClient(SPOKEN, clip_span=(1.0, 3.4))
    clipping = AssemblyAiProvider(api_key="k", client=cut)
    # A service that cannot be asked about part of a recording in place has
    # a clip cut for it instead, which is the only difference between these.
    clipping.capabilities = replace(ASSEMBLYAI_CAPABILITIES, time_window=False)
    by_clip = escalate([window], clipping, canonical, TIGHT)

    assert by_window.results[0].used_time_window is True
    assert by_clip.results[0].used_time_window is False
    assert uploaded.uploads and not cut.uploads

    windowed_times = [(token.text, token.start, token.end) for token in by_window.results[0].tokens]
    clipped_times = [(token.text, token.start, token.end) for token in by_clip.results[0].tokens]
    assert windowed_times == [("fifteen", 2.0, 2.4), ("thousand", 2.5, 3.0), ("rand", 3.1, 3.4)]
    assert clipped_times == pytest.approx(windowed_times, abs=1e-6)


def test_the_clip_path_reports_the_audio_it_really_cut(tmp_path):
    """The recording runs out before the window does, and the answer says so.

    The offset used afterwards is the one the cut reported, not the one that
    was asked for. Using the requested one would move every word in the clip
    by the difference, and nothing downstream could detect it.
    """
    canonical = real_recording(tmp_path)
    # Grouped without a length, so the padding runs past the end of the
    # recording and the cut is what discovers it.
    window, = group_into_windows([dispute(5.5, 5.9)], options=TIGHT)
    assert window.span.end > canonical.duration

    provider = AssemblyAiProvider(api_key="k", client=FakeAssemblyAiClient(SPOKEN))
    provider.capabilities = replace(ASSEMBLYAI_CAPABILITIES, time_window=False)

    outcome = escalate([window], provider, canonical, TIGHT)
    result = outcome.results[0]

    assert result.canonical_offset == pytest.approx(4.5)
    assert result.audio_span.start == pytest.approx(4.5)
    assert result.audio_span.end == pytest.approx(canonical.duration)


def test_the_clip_is_kept_where_a_folder_was_named_and_removed_where_it_was_not(tmp_path):
    canonical = real_recording(tmp_path)
    window, = group_into_windows([dispute(2.0, 2.4)], duration=canonical.duration, options=TIGHT)
    provider = AssemblyAiProvider(api_key="k", client=FakeAssemblyAiClient(SPOKEN, (1.0, 3.4)))
    provider.capabilities = replace(ASSEMBLYAI_CAPABILITIES, time_window=False)
    folder = tmp_path / "windows"

    escalate([window], provider, canonical, TIGHT, folder=folder)

    assert list(folder.glob("escalation-*.wav"))
    before = set(tmp_path.iterdir())
    escalate([window], provider, canonical, TIGHT)
    assert set(tmp_path.iterdir()) == before


# -- The ceiling, and services that do not answer ------------------------


class StubProvider:
    """A service that answers, or does not, without any audio being involved."""

    provider = Provider.ASSEMBLYAI
    capabilities = ASSEMBLYAI_CAPABILITIES

    def __init__(self, error: str | None = None):
        self.error = error
        self.asked: list[AudioSpan] = []

    def upload(self, audio_path: Path) -> str:
        return "https://uploads.example/one"

    def transcribe_window(self, request, *, audio_url=None, window=None, cancelled=None):
        self.asked.append(window)
        if self.error is not None:
            return ProviderResult(provider=self.provider, error=self.error)
        return ProviderResult(
            provider=self.provider,
            tokens=[
                ProviderToken(
                    provider=self.provider,
                    index=0,
                    text="fifty",
                    start=window.start,
                    end=window.start + 0.3,
                )
            ],
        )


def windows_for(count: int) -> list[EscalationWindow]:
    return group_into_windows(
        [dispute(index * 30.0, index * 30.0 + 0.4) for index in range(count)],
        duration=count * 30.0 + 10.0,
        options=TIGHT,
    )


def test_the_ceiling_stops_the_run_and_says_so_plainly():
    """Reaching it is not an error, and it is never silent.

    The passages it stopped are named as still needing attention, because a
    ceiling that quietly dropped them would leave the user believing every
    dispute had been looked at again.
    """
    provider = StubProvider()
    windows = windows_for(4)

    outcome = escalate(
        windows,
        provider,
        CanonicalAudio("a.wav", "a.wav", 130.0, 16000, 1, 100, "wav"),
        replace(TIGHT, maximum_escalations=2),
    )

    assert len(provider.asked) == 2
    assert len(outcome.results) == 2
    assert len(outcome.skipped) == 2
    assert outcome.reached_limit is True
    assert any("Only 2 of 4" in warning for warning in outcome.warnings)
    assert any("review queue" in warning for warning in outcome.warnings)


def test_a_ceiling_of_none_sends_nothing_and_still_explains_itself():
    provider = StubProvider()

    outcome = escalate(
        windows_for(2),
        provider,
        CanonicalAudio("a.wav", "a.wav", 70.0, 16000, 1, 100, "wav"),
        replace(TIGHT, maximum_escalations=0),
    )

    assert provider.asked == []
    assert len(outcome.skipped) == 2
    assert outcome.warnings


def test_a_service_that_does_not_answer_never_fails_the_transcript():
    provider = StubProvider(error="AssemblyAI could not be reached: timed out")

    outcome = escalate(
        windows_for(2),
        provider,
        CanonicalAudio("a.wav", "a.wav", 70.0, 16000, 1, 100, "wav"),
        TIGHT,
    )

    assert len(outcome.results) == 2
    assert outcome.failed == outcome.results
    assert any("review queue" in warning for warning in outcome.warnings)


def test_what_a_second_opinion_did_not_reach_is_flagged_for_a_person():
    transcript = Transcript(recording_name="talk")
    first = word("fifteen", 10.0, 10.4)
    second = word("fifty", 40.0, 40.4)
    transcript.tokens = [first, second]
    windows = [
        EscalationWindow(
            span=AudioSpan(9.0, 11.4),
            target=AudioSpan(10.0, 10.4),
            disputes=(Dispute(AudioSpan(10.0, 10.4), (EscalationReason.NUMBER_DIFFERS,),
                              token_ids=(first.id,)),),
        )
    ]
    outcome = escalate(
        windows,
        StubProvider(error="AssemblyAI could not be reached"),
        CanonicalAudio("a.wav", "a.wav", 60.0, 16000, 1, 100, "wav"),
        TIGHT,
    )

    apply_outcome(transcript, outcome)

    assert ReviewReason.ESCALATION_UNRESOLVED in first.review_reasons
    assert second.review_reasons == []
    assert transcript.warnings


def test_the_options_are_read_off_the_settings_without_importing_them():
    """A partly configured application can still ask a question."""
    from vox_verbatim.settings import AssemblyAiSettings, ProcessingSettings
    from vox_verbatim.transcription.model import RecordingConfiguration

    processing = ProcessingSettings()
    processing.escalation_context_seconds_before = 4.0
    processing.maximum_escalations_per_recording = 12
    configuration = RecordingConfiguration(afrikaans_enabled=True, expected_speaker_count=2)

    options = EscalationOptions.from_settings(
        processing, AssemblyAiSettings(), configuration, vocabulary_terms=("Kubernetes", " ")
    )

    assert options.context_seconds_before == 4.0
    assert options.maximum_escalations == 12
    assert options.afrikaans_enabled is True
    assert options.expected_speaker_count == 2
    assert Language.AFRIKAANS in options.languages
    assert options.vocabulary_terms == ("Kubernetes",)
    # Nothing configured at all still gives usable options rather than an error.
    assert EscalationOptions.from_settings() == EscalationOptions()


def test_only_the_words_that_were_asked_about_count_as_answers():
    """The padding is context. A word from it is not a correction."""
    window = EscalationWindow(
        span=AudioSpan(9.0, 13.0),
        target=AudioSpan(10.0, 10.5),
        disputes=(dispute(10.0, 10.5),),
    )
    result = EscalationResult(
        window=window,
        model="universal-3-5-pro",
        strength=EvidenceStrength.DECIDING,
        audio_span=window.span,
        canonical_offset=0.0,
        used_time_window=True,
        tokens=(
            ProviderToken(Provider.ASSEMBLYAI, 0, "and", start=9.2, end=9.5),
            ProviderToken(Provider.ASSEMBLYAI, 1, "fifty", start=10.0, end=10.5),
            ProviderToken(Provider.ASSEMBLYAI, 2, "please", start=11.0, end=11.4),
        ),
    )

    assert [token.text for token in result.tokens_in_target()] == ["fifty"]


# -- Choosing which windows to send when there are too many --------------


def _window_with(reasons, candidates=("fifty", "fifteen"), start=10.0) -> EscalationWindow:
    one = Dispute(
        span=AudioSpan(start, start + 0.5),
        reasons=tuple(reasons),
        token_ids=("t",),
        candidates=tuple(candidates),
    )
    return EscalationWindow(
        span=AudioSpan(start - 1.0, start + 1.5),
        target=one.span,
        disputes=(one,),
    )


def test_real_disagreements_are_sent_before_agreed_upon_numbers():
    """The ceiling takes the list from the front, so the front must hold
    the windows a second opinion can actually change."""
    from vox_verbatim.transcription.escalation import prioritise

    agreed_amount = _window_with((EscalationReason.NUMBER_DIFFERS,), ("50",), start=5.0)
    three_ways = _window_with((EscalationReason.ALL_PROVIDERS_DISAGREE,), start=100.0)
    contested_number = _window_with((EscalationReason.NUMBER_DIFFERS,), start=200.0)
    speaker_only = _window_with(
        (EscalationReason.SPEAKER_BOUNDARY_UNCERTAIN,), ("yes",), start=300.0
    )

    ordered = prioritise([agreed_amount, speaker_only, contested_number, three_ways])

    assert ordered[:2] == [three_ways, contested_number]
    # Equal worth keeps time order, so a user reading the warnings can
    # still follow the recording.
    assert ordered[2:] == [agreed_amount, speaker_only]


def test_escalation_reports_each_window_as_it_is_answered():
    provider = StubProvider()
    seen: list[tuple[int, int]] = []

    escalate(
        windows_for(3),
        provider,
        CanonicalAudio("a.wav", "a.wav", 100.0, 16000, 1, 100, "wav"),
        TIGHT,
        progress=lambda done, total: seen.append((done, total)),
    )

    assert seen == [(1, 3), (2, 3), (3, 3)]


# -- Putting the answers back into the transcript ------------------------


def _answered(
    token: FinalToken,
    heard: tuple[tuple[str, float, float], ...],
    strength: EvidenceStrength = EvidenceStrength.DECIDING,
    error: str | None = None,
) -> EscalationResult:
    """A window over one word, and what the second service said there."""
    one = Dispute(
        span=token.span,
        reasons=(EscalationReason.ALL_PROVIDERS_DISAGREE,),
        token_ids=(token.id,),
    )
    window = EscalationWindow(span=token.span.padded(2.0, 2.0), target=one.span, disputes=(one,))
    return EscalationResult(
        window=window,
        model="universal-3-5-pro",
        strength=strength,
        audio_span=window.span,
        canonical_offset=0.0,
        used_time_window=True,
        tokens=tuple(
            ProviderToken(Provider.ASSEMBLYAI, index, text, start=start, end=end)
            for index, (text, start, end) in enumerate(heard)
        ),
        error=error,
    )


def _disputed_word() -> FinalToken:
    token = word(
        "fifteen",
        10.0,
        10.5,
        candidates=(("fifteen", (Provider.ELEVENLABS,)), ("fifty", (Provider.OPENAI,))),
        reasons=(ReviewReason.PROVIDER_DISAGREEMENT,),
    )
    token.text_confidence = Confidence.REVIEW_REQUIRED
    return token


def test_a_deciding_answer_that_sides_with_the_other_reading_corrects_the_word():
    from vox_verbatim.transcription.escalation import EscalationOutcome, apply_answers

    token = _disputed_word()
    outcome = EscalationOutcome(results=(_answered(token, (("fifty", 10.0, 10.5),)),))

    applied = apply_answers([token], outcome)

    assert applied.settled == 1
    assert token.text == "fifty"
    assert token.text_source is Provider.ASSEMBLYAI
    assert token.text_confidence is Confidence.HIGH
    assert token.needs_review is False
    # The timing was measured for "fifteen"; saying a different word now
    # sits in it is what lets the timing stage look again.
    assert token.timing_status.value == "mapped_substitution"
    # The second service now counts as one of the voices for that reading.
    fifty = next(candidate for candidate in token.candidates if candidate.text == "fifty")
    assert Provider.ASSEMBLYAI in fifty.providers
    assert applied.evidence is not None and applied.evidence.provider is Provider.ASSEMBLYAI


def test_an_answer_that_agrees_with_the_word_confirms_it_and_clears_the_queue():
    from vox_verbatim.transcription.escalation import EscalationOutcome, apply_answers

    token = _disputed_word()
    outcome = EscalationOutcome(results=(_answered(token, (("fifteen", 10.0, 10.5),)),))

    applied = apply_answers([token], outcome)

    assert applied.confirmed == 1
    assert token.text == "fifteen"
    assert token.text_confidence is Confidence.HIGH
    assert token.needs_review is False


def test_an_answer_nobody_else_offered_is_kept_as_evidence_and_never_written_in():
    from vox_verbatim.transcription.escalation import EscalationOutcome, apply_answers

    token = _disputed_word()
    outcome = EscalationOutcome(results=(_answered(token, (("sixty", 10.0, 10.5),)),))

    applied = apply_answers([token], outcome)

    assert applied.unsettled == 1
    assert token.text == "fifteen"
    assert token.text_confidence is Confidence.REVIEW_REQUIRED
    assert ReviewReason.ESCALATION_UNRESOLVED in token.review_reasons
    assert [candidate.text for candidate in token.candidates] == ["fifteen", "fifty", "sixty"]


def test_an_informing_answer_adds_evidence_but_settles_nothing():
    from vox_verbatim.transcription.escalation import EscalationOutcome, apply_answers

    token = _disputed_word()
    outcome = EscalationOutcome(
        results=(
            _answered(token, (("fifty", 10.0, 10.5),), strength=EvidenceStrength.INFORMING),
        )
    )

    applied = apply_answers([token], outcome)

    assert applied.unsettled == 1
    assert token.text == "fifteen"
    assert token.needs_review is True
    fifty = next(candidate for candidate in token.candidates if candidate.text == "fifty")
    assert Provider.ASSEMBLYAI in fifty.providers


def test_a_value_that_must_not_be_guessed_is_never_settled_by_a_second_opinion():
    from vox_verbatim.transcription.escalation import EscalationOutcome, apply_answers

    token = _disputed_word()
    token.risk_categories = [RiskCategory.MONEY]
    outcome = EscalationOutcome(results=(_answered(token, (("fifty", 10.0, 10.5),)),))

    apply_answers([token], outcome)

    assert token.text == "fifteen"
    assert token.needs_review is True
    assert ReviewReason.ESCALATION_UNRESOLVED in token.review_reasons


def test_a_second_opinion_clears_only_the_reasons_that_are_about_the_text():
    from vox_verbatim.transcription.escalation import EscalationOutcome, apply_answers

    token = _disputed_word()
    token.flag(ReviewReason.SPEAKER_UNCERTAIN)
    outcome = EscalationOutcome(results=(_answered(token, (("fifty", 10.0, 10.5),)),))

    apply_answers([token], outcome)

    assert token.text == "fifty"
    assert token.review_reasons == [ReviewReason.SPEAKER_UNCERTAIN]
    assert token.needs_review is True


def test_silence_where_the_word_was_leaves_it_for_a_person():
    from vox_verbatim.transcription.escalation import EscalationOutcome, apply_answers

    token = _disputed_word()
    outcome = EscalationOutcome(results=(_answered(token, (("hello", 8.0, 8.4),)),))

    applied = apply_answers([token], outcome)

    assert applied.unsettled == 1
    assert token.text == "fifteen"
    assert ReviewReason.ESCALATION_UNRESOLVED in token.review_reasons


def test_a_person_who_already_corrected_the_word_is_not_overruled():
    from vox_verbatim.transcription.escalation import EscalationOutcome, apply_answers

    token = _disputed_word()
    token.human_corrected = True
    outcome = EscalationOutcome(results=(_answered(token, (("fifty", 10.0, 10.5),)),))

    apply_answers([token], outcome)

    assert token.text == "fifteen"


def test_everything_the_second_service_heard_is_kept_as_one_numbered_result():
    from vox_verbatim.transcription.escalation import EscalationOutcome, apply_answers

    first = _disputed_word()
    second = word("Bosh", 50.0, 50.4, candidates=(("Bosh", (Provider.ELEVENLABS,)),))
    outcome = EscalationOutcome(
        results=(
            _answered(first, (("and", 9.0, 9.3), ("fifty", 10.0, 10.5))),
            _answered(second, (("Bosch", 50.0, 50.4),)),
        )
    )

    applied = apply_answers([first, second], outcome)

    assert [token.index for token in applied.evidence.tokens] == [0, 1, 2]
    assert [token.text for token in applied.evidence.tokens] == ["and", "fifty", "Bosch"]
    # The reference on the word points into that numbering.
    fifty = next(candidate for candidate in first.candidates if candidate.text == "fifty")
    assert TokenReference(Provider.ASSEMBLYAI, 1) in fifty.source_tokens


def test_a_neighbour_that_overlaps_by_a_few_milliseconds_is_not_part_of_the_answer():
    """Services measure their own boundaries, and adjacent words abut, so the
    word after the disputed one always overlaps it slightly. The answer to a
    question about one word is one word."""
    from vox_verbatim.transcription.escalation import EscalationOutcome, apply_answers

    token = _disputed_word()  # 10.0 to 10.5
    outcome = EscalationOutcome(
        results=(
            _answered(
                token,
                (("we", 9.5, 9.98), ("fifty", 9.98, 10.48), ("dollars", 10.48, 10.9)),
            ),
        )
    )

    applied = apply_answers([token], outcome)

    assert applied.settled == 1
    assert token.text == "fifty"
    assert "fifty dollars" not in [candidate.text for candidate in token.candidates]


def test_one_service_word_across_two_disputed_words_settles_neither():
    """One word where the transcript has two is a disagreement about the word
    count, not a confirmation of both."""
    from vox_verbatim.transcription.escalation import EscalationOutcome, apply_answers

    first = word("the", 10.0, 10.2, candidates=(("the", (Provider.ELEVENLABS,)), ("a", (Provider.OPENAI,))))
    second = word("the", 10.2, 10.4, candidates=(("the", (Provider.ELEVENLABS,)), ("uh", (Provider.OPENAI,))))
    for token in (first, second):
        token.text_confidence = Confidence.REVIEW_REQUIRED
        token.flag(ReviewReason.PROVIDER_DISAGREEMENT)
    disputes = tuple(
        Dispute(span=token.span, reasons=(EscalationReason.ALL_PROVIDERS_DISAGREE,), token_ids=(token.id,))
        for token in (first, second)
    )
    window = EscalationWindow(span=AudioSpan(8.0, 12.0), target=AudioSpan(10.0, 10.4), disputes=disputes)
    result = EscalationResult(
        window=window,
        model="universal-3-5-pro",
        strength=EvidenceStrength.DECIDING,
        audio_span=window.span,
        canonical_offset=0.0,
        used_time_window=True,
        tokens=(ProviderToken(Provider.ASSEMBLYAI, 0, "the", start=10.0, end=10.4),),
    )

    applied = apply_answers([first, second], EscalationOutcome(results=(result,)))

    # The word's middle is at 10.2, on the boundary, so it answers one of
    # the two and the other is left for a person.
    assert applied.confirmed == 1
    assert applied.unsettled == 1
    assert sum(1 for token in (first, second) if token.needs_review) == 1


def test_the_time_window_question_is_retried_like_any_other_request():
    """A rate limit on one of two hundred short questions should cost a
    pause, not that passage's second opinion."""
    from vox_verbatim.transcription.providers.base import ProviderError, TranscriptionProvider

    class FlakyProvider(TranscriptionProvider):
        provider = Provider.ASSEMBLYAI
        capabilities = ASSEMBLYAI_CAPABILITIES
        maximum_retries = 2
        retry_backoff_seconds = 0.0

        def __init__(self):
            self.calls = 0

        @property
        def model_identifier(self):
            return "fake"

        def is_configured(self):
            return True

        def describe_configuration_problem(self):
            return None

        def _transcribe(self, request, cancelled=None):
            raise AssertionError("not used")

        def upload(self, audio_path):
            return "https://uploads.example/one"

        def transcribe_window(self, request, *, audio_url=None, window=None, cancelled=None):
            self.calls += 1
            if self.calls == 1:
                raise ProviderError("Too many requests", retryable=True, status_code=429)
            return ProviderResult(
                provider=self.provider,
                tokens=[ProviderToken(self.provider, 0, "fifty", start=window.start, end=window.start + 0.3)],
            )

    provider = FlakyProvider()
    outcome = escalate(
        windows_for(1),
        provider,
        CanonicalAudio("a.wav", "a.wav", 100.0, 16000, 1, 100, "wav"),
        TIGHT,
    )

    assert provider.calls == 2
    assert outcome.results[0].succeeded


def test_a_short_word_heard_a_little_later_by_the_service_is_still_the_answer():
    """Two engines can disagree about where a 100 ms word starts by more than
    half its length. The word whose middle is just outside the span is it."""
    from vox_verbatim.transcription.escalation import EscalationOutcome, apply_answers

    token = word(
        "fifteen", 1.0, 1.1,
        candidates=(("fifteen", (Provider.ELEVENLABS,)), ("fifty", (Provider.OPENAI,))),
        reasons=(ReviewReason.PROVIDER_DISAGREEMENT,),
    )
    token.text_confidence = Confidence.REVIEW_REQUIRED
    outcome = EscalationOutcome(
        results=(
            _answered(token, (("and", 0.8, 0.96), ("fifty", 1.06, 1.16), ("dollars", 1.16, 1.5))),
        )
    )

    applied = apply_answers([token], outcome)

    assert applied.settled == 1
    assert token.text == "fifty"


def test_a_word_the_service_split_in_two_still_confirms_it():
    from vox_verbatim.transcription.escalation import EscalationOutcome, apply_answers

    token = word(
        "database", 10.0, 10.5,
        candidates=(("database", (Provider.ELEVENLABS,)), ("data bus", (Provider.OPENAI,))),
        reasons=(ReviewReason.PROVIDER_DISAGREEMENT,),
    )
    token.text_confidence = Confidence.REVIEW_REQUIRED
    outcome = EscalationOutcome(
        results=(_answered(token, (("data", 10.0, 10.2), ("base", 10.2, 10.5))),)
    )

    applied = apply_answers([token], outcome)

    assert applied.confirmed == 1
    assert token.needs_review is False


def test_when_neighbours_lean_into_a_long_span_the_word_in_the_middle_answers():
    """Two words can have their middles inside a long word's span. Joined they
    are nobody's reading; the one nearest the middle is what was asked."""
    from vox_verbatim.transcription.escalation import EscalationOutcome, apply_answers

    token = _disputed_word()  # 10.0 to 10.5
    outcome = EscalationOutcome(
        results=(_answered(token, (("fifty", 9.98, 10.30), ("dollars", 10.30, 10.7))),)
    )

    applied = apply_answers([token], outcome)

    assert applied.settled == 1
    assert token.text == "fifty"
    assert "fifty dollars" not in [candidate.text for candidate in token.candidates]
