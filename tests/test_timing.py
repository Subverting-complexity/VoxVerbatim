"""Tests for the timing states and the rules that keep them honest.

None of this needs audio. The question in every test is what a word is
entitled to claim about when it was spoken, given what somebody measured and
what reconciliation then did to the text, and that is decided by data rather
than by sound. The forced aligner is driven by a stand-in, because what
matters is how its answers and its failures are treated, not whether a
network call can be made.

Two of these tests are here because getting them wrong would be invisible.
A split that invented a boundary would produce two perfectly plausible
timestamps that nothing in the recording supports. An alignment loss read
the wrong way round would mark the worst measurements as the best. Neither
would ever fail visibly, so both are checked directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_transcriber.transcription.model import (
    AlignmentStatus,
    AudioSpan,
    Confidence,
    FinalToken,
    Language,
    Provider,
    ProviderToken,
    ReviewReason,
    TimingStatus,
    TokenReference,
)
from audio_transcriber.transcription.providers.base import ForcedAligner, ProviderError
from audio_transcriber.transcription.timing import (
    AlignedWord,
    TimingOptions,
    TimingOutcome,
    apply_timing,
    map_timing,
    may_enter_transcript,
    needs_forced_alignment,
    note_removed_word,
    read_aligned_words,
    realign,
    status_for_loss,
)

AUDIO = Path("recording.wav")


def heard(text: str, start: float, end: float, index: int = 0) -> ProviderToken:
    """One word as the timing service measured it."""
    return ProviderToken(
        provider=Provider.ELEVENLABS, index=index, text=text, start=start, end=end
    )


def final(
    text: str,
    start: float | None = None,
    end: float | None = None,
    status: TimingStatus = TimingStatus.EXACT_PROVIDER_TIME,
    containing: AudioSpan | None = None,
) -> FinalToken:
    return FinalToken(
        text=text,
        start=start,
        end=end,
        timing_status=status,
        timing_source=Provider.ELEVENLABS,
        source_audio_span=containing,
    )


class FakeAligner(ForcedAligner):
    """A forced aligner that answers, refuses or fails, on demand.

    It reports a loss per word where a test asks it to, because that number
    is the whole of the difference between a measured span and an
    approximate one.
    """

    provider = Provider.ELEVENLABS

    def __init__(self, rows=None, error: str | None = None, configured: bool = True):
        self.rows = rows or []
        self.error = error
        self.configured = configured
        self.asked: list[tuple[str, Language, float]] = []

    def is_configured(self) -> bool:
        return self.configured

    def supports(self, language: Language) -> bool:
        return language is not Language.AFRIKAANS

    def align(self, audio_path, text, language, canonical_offset=0.0):
        self.asked.append((text, language, canonical_offset))
        if self.error is not None:
            raise ProviderError(self.error)
        return self.rows


# -- Mapping the timing of words that changed ----------------------------


def test_an_unchanged_word_keeps_the_time_that_was_measured():
    outcome, = map_timing(["contract"], [heard("contract", 12.0, 12.5)])

    assert outcome.status is TimingStatus.EXACT_PROVIDER_TIME
    assert outcome.span == AudioSpan(12.0, 12.5)
    assert outcome.confidence is Confidence.HIGH
    assert outcome.is_exact


def test_a_word_only_written_differently_keeps_the_measured_time():
    """The sound did not change, so the span fits exactly as well as it did."""
    outcome, = map_timing(["Jürgen"], [heard("Jurgen", 12.0, 12.5)])

    assert outcome.status is TimingStatus.MAPPED_FORMAT_EQUIVALENT
    assert outcome.span == AudioSpan(12.0, 12.5)
    assert outcome.is_exact


def test_a_confident_substitution_inherits_the_span_and_says_that_it_did():
    """The span is the best answer there is, and it is not a measurement of this word."""
    outcome, = map_timing(["contact"], [heard("contract", 12.0, 12.5)])

    assert outcome.status is TimingStatus.MAPPED_SUBSTITUTION
    assert outcome.span == AudioSpan(12.0, 12.5)
    assert outcome.is_exact is False
    assert outcome.confidence is Confidence.REVIEW_SUGGESTED


def test_an_unsure_substitution_is_flagged_as_well_as_inherited():
    outcome, = map_timing(["contact"], [heard("contract", 12.0, 12.5)], confident=False)

    assert ReviewReason.WEAK_ALIGNMENT in outcome.review_reasons


def test_a_merge_covers_the_span_of_everything_it_replaced():
    """Both outer edges were measured, so the combined span is defensible."""
    outcome, = map_timing(
        ["database"], [heard("data", 3.0, 3.4, 0), heard("base", 3.45, 3.9, 1)]
    )

    assert outcome.span == AudioSpan(3.0, 3.9)
    assert outcome.status is TimingStatus.MAPPED_FORMAT_EQUIVALENT


def test_a_split_shares_one_span_and_invents_no_boundary_between_the_words():
    """The rule this module exists for.

    Nobody measured where "can" ends and "not" begins. Two different numbers
    here would be a guess wearing the clothes of a measurement, and every
    part of the application that trusts a timestamp would then trust it.
    """
    first, second = map_timing(["can", "not"], [heard("cannot", 8.0, 8.6)])

    assert first.span == AudioSpan(8.0, 8.6)
    assert second.span == first.span
    assert first.status is second.status is TimingStatus.SHARED_PHRASE_SPAN
    assert first.shares_span and second.shares_span
    assert first.is_exact is False


def test_an_added_word_is_unaligned_and_may_not_enter_the_transcript_unverified():
    region = AudioSpan(20.0, 21.0)

    outcome, = map_timing(["the"], [], containing_span=region)

    assert outcome.status is TimingStatus.UNALIGNED
    assert outcome.span is None
    assert outcome.containing_span == region
    assert ReviewReason.UNALIGNED_WORD in outcome.review_reasons
    assert outcome.may_enter_transcript is False


def test_a_word_a_person_confirmed_may_enter_the_transcript():
    token = final("the")
    token.timing_status = TimingStatus.UNALIGNED

    assert may_enter_transcript(token) is False
    token.human_corrected = True
    assert may_enter_transcript(token) is True


def test_a_removed_word_has_no_timing_and_stays_in_provenance():
    """It is kept as evidence and never shown, which is how a hallucinating
    service can be recognised later."""
    assert map_timing([], [heard("um", 5.0, 5.2)]) == []

    neighbour = final("yes", 5.3, 5.6)
    note_removed_word(neighbour, heard("um", 5.0, 5.2, index=11))
    note_removed_word(neighbour, heard("um", 5.0, 5.2, index=11))

    assert neighbour.rejected_tokens == [TokenReference(Provider.ELEVENLABS, 11)]
    assert neighbour.text == "yes"


def test_a_source_word_with_no_timing_leaves_the_final_word_unaligned():
    untimed = ProviderToken(provider=Provider.OPENAI, index=0, text="the")

    outcome, = map_timing(["the"], [untimed], containing_span=AudioSpan(1.0, 2.0))

    assert outcome.status is TimingStatus.UNALIGNED


def test_applying_a_decision_changes_the_timing_and_nothing_else():
    token = final("contact", 12.0, 12.5)
    token.text_confidence = Confidence.HIGH
    token.speaker = "speaker_1"
    token.speaker_confidence = Confidence.HIGH

    apply_timing(
        token,
        TimingOutcome(
            status=TimingStatus.FORCED_ALIGNED,
            span=AudioSpan(12.1, 12.6),
            source=Provider.ELEVENLABS,
            confidence=Confidence.HIGH,
        ),
    )

    assert (token.start, token.end) == (12.1, 12.6)
    assert token.timing_status is TimingStatus.FORCED_ALIGNED
    assert token.text == "contact"
    assert token.text_confidence is Confidence.HIGH
    assert token.speaker == "speaker_1"
    assert token.speaker_confidence is Confidence.HIGH


# -- Deciding when to measure again --------------------------------------


def test_a_word_nobody_changed_is_not_measured_again():
    """Running it over everything would pay to confirm what is already known."""
    token = final("contract", 12.0, 12.5, TimingStatus.EXACT_PROVIDER_TIME)

    assert needs_forced_alignment(token) is False


def test_a_word_whose_text_moved_is_worth_measuring_again():
    for status in (
        TimingStatus.MAPPED_SUBSTITUTION,
        TimingStatus.SHARED_PHRASE_SPAN,
        TimingStatus.APPROXIMATE_SPAN,
        TimingStatus.UNALIGNED,
    ):
        assert needs_forced_alignment(final("x", 1.0, 2.0, status)) is True


def test_a_settled_dispute_is_measured_again_whatever_its_state():
    token = final("fifty", 12.0, 12.5, TimingStatus.EXACT_PROVIDER_TIME)

    assert needs_forced_alignment(token, settled_dispute=True) is True


def test_a_merge_or_split_is_measured_again_even_where_the_span_looks_settled():
    token = final("database", 3.0, 3.9, TimingStatus.MAPPED_FORMAT_EQUIVALENT)
    token.alignment_status = AlignmentStatus.MERGE

    assert needs_forced_alignment(token) is True


def test_switching_forced_alignment_off_stops_it_being_asked_for():
    token = final("contact", 12.0, 12.5, TimingStatus.MAPPED_SUBSTITUTION)

    assert needs_forced_alignment(
        token, options=TimingOptions(forced_alignment_enabled=False)
    ) is False


# -- The loss, and which way round it reads ------------------------------


def test_a_small_loss_is_a_good_alignment_and_a_large_one_is_not():
    """Lower is better here, the opposite way round to a log probability.

    Reading these the wrong way round would mark the worst measurements as
    the best, and nothing would ever fail visibly.
    """
    assert status_for_loss(0.05) is TimingStatus.FORCED_ALIGNED
    assert status_for_loss(0.9) is TimingStatus.APPROXIMATE_SPAN

    better, worse = 0.05, 0.9
    assert better < worse
    assert status_for_loss(better) is TimingStatus.FORCED_ALIGNED
    assert status_for_loss(worse) is not TimingStatus.FORCED_ALIGNED


def test_no_loss_at_all_is_not_a_bad_loss():
    """The word was still measured, and an absent complaint is not a complaint."""
    assert status_for_loss(None) is TimingStatus.FORCED_ALIGNED


def test_the_loss_decides_the_state_of_a_measured_word():
    aligner = FakeAligner([("can", 8.0, 8.3, 0.02), ("not", 8.3, 8.6, 0.95)])
    tokens = [
        final("can", 8.0, 8.6, TimingStatus.SHARED_PHRASE_SPAN),
        final("not", 8.0, 8.6, TimingStatus.SHARED_PHRASE_SPAN),
    ]

    first, second = realign(tokens, aligner, AUDIO, language=Language.ENGLISH)

    assert first.status is TimingStatus.FORCED_ALIGNED
    assert first.confidence is Confidence.HIGH
    assert second.status is TimingStatus.APPROXIMATE_SPAN
    assert second.confidence is Confidence.REVIEW_SUGGESTED


def test_the_words_are_sent_as_plain_running_text():
    aligner = FakeAligner([("can", 8.0, 8.3), ("not", 8.3, 8.6)])
    tokens = [final("can", 8.0, 8.6), final("not", 8.0, 8.6)]

    realign(tokens, aligner, AUDIO, canonical_offset=4.0, language=Language.ENGLISH)

    assert aligner.asked == [("can not", Language.ENGLISH, 4.0)]


def test_an_answer_of_any_shape_is_read_the_same_way():
    rows = [
        ("can", 8.0, 8.3),
        ("not", 8.3, 8.6, 0.4),
        {"text": "yet", "start": 8.6, "end": 8.9, "loss": 0.1},
        {"text": "nowhere", "start": None, "end": None},
    ]

    words = read_aligned_words(rows)

    assert words == [
        AlignedWord("can", 8.0, 8.3, None),
        AlignedWord("not", 8.3, 8.6, 0.4),
        AlignedWord("yet", 8.6, 8.9, 0.1),
    ]


# -- When measuring again cannot be done ---------------------------------


def test_a_failed_alignment_keeps_the_span_it_had_and_marks_it_approximate():
    """Never a reason to divide a span into plausible-looking pieces."""
    aligner = FakeAligner(error="Forced alignment could not be reached")
    tokens = [
        final("can", 8.0, 8.6, TimingStatus.SHARED_PHRASE_SPAN),
        final("not", 8.0, 8.6, TimingStatus.SHARED_PHRASE_SPAN),
    ]

    first, second = realign(tokens, aligner, AUDIO, language=Language.ENGLISH)

    assert first.status is TimingStatus.APPROXIMATE_SPAN
    assert first.span == AudioSpan(8.0, 8.6)
    assert second.span == first.span, "a failure must not invent a boundary"
    assert ReviewReason.FORCED_ALIGNMENT_FAILED in first.review_reasons


def test_an_answer_that_does_not_match_the_phrase_is_treated_as_a_failure():
    """Which measurement belongs to which word is not known, so none is used."""
    aligner = FakeAligner([("can", 8.0, 8.3)])
    tokens = [final("can", 8.0, 8.6), final("not", 8.0, 8.6)]

    first, second = realign(tokens, aligner, AUDIO, language=Language.ENGLISH)

    assert first.status is TimingStatus.APPROXIMATE_SPAN
    assert first.span == second.span == AudioSpan(8.0, 8.6)


def test_an_answer_that_runs_backwards_is_treated_as_a_failure():
    aligner = FakeAligner([("can", 8.4, 8.6), ("not", 8.0, 8.3)])
    tokens = [final("can", 8.0, 8.6), final("not", 8.0, 8.6)]

    first, _ = realign(tokens, aligner, AUDIO, language=Language.ENGLISH)

    assert first.status is TimingStatus.APPROXIMATE_SPAN
    assert ReviewReason.FORCED_ALIGNMENT_FAILED in first.review_reasons


def test_an_afrikaans_span_keeps_its_mapped_timing_and_is_marked_uncertain():
    """This is the correct answer rather than a limitation to work around.

    The aligner does not hear Afrikaans, so there is nothing to measure with.
    The words keep the timing the backbone gave them and say plainly that it
    is uncertain, which leaves room for an Afrikaans-capable aligner later.
    """
    aligner = FakeAligner([("hierdie", 8.0, 8.3), ("kontrak", 8.3, 8.6)])
    tokens = [
        final("hierdie", 8.0, 8.3, TimingStatus.MAPPED_SUBSTITUTION),
        final("kontrak", 8.3, 8.6, TimingStatus.MAPPED_SUBSTITUTION),
    ]

    first, second = realign(tokens, aligner, AUDIO, language=Language.AFRIKAANS)

    assert aligner.asked == [], "the aligner must not be asked a question it cannot answer"
    assert first.status is second.status is TimingStatus.UNCERTAIN
    assert first.span == AudioSpan(8.0, 8.3)
    assert second.span == AudioSpan(8.3, 8.6)
    assert first.containing_span == AudioSpan(8.0, 8.6)


def test_an_afrikaans_word_with_no_boundaries_of_its_own_falls_back_to_the_phrase():
    aligner = FakeAligner()
    tokens = [
        final("hierdie", 8.0, 8.6, TimingStatus.SHARED_PHRASE_SPAN),
        final("nuwe", None, None, TimingStatus.UNALIGNED, containing=AudioSpan(8.0, 8.6)),
    ]

    _, second = realign(tokens, aligner, AUDIO, language=Language.AFRIKAANS)

    assert second.status is TimingStatus.SHARED_PHRASE_SPAN
    assert second.span == AudioSpan(8.0, 8.6)
    assert second.shares_span


def test_no_aligner_at_all_leaves_the_spans_alone_and_calls_them_approximate():
    tokens = [final("contact", 12.0, 12.5, TimingStatus.MAPPED_SUBSTITUTION)]

    outcome, = realign(tokens, None, AUDIO, language=Language.ENGLISH)

    assert outcome.status is TimingStatus.APPROXIMATE_SPAN
    assert outcome.span == AudioSpan(12.0, 12.5)
    assert outcome.review_reasons == ()


def test_an_unconfigured_aligner_is_the_same_as_no_aligner():
    tokens = [final("contact", 12.0, 12.5, TimingStatus.MAPPED_SUBSTITUTION)]

    outcome, = realign(
        tokens, FakeAligner(configured=False), AUDIO, language=Language.ENGLISH
    )

    assert outcome.status is TimingStatus.APPROXIMATE_SPAN


def test_a_word_with_nothing_behind_it_stays_unaligned_when_alignment_cannot_run():
    tokens = [final("the", None, None, TimingStatus.UNALIGNED)]

    outcome, = realign(tokens, None, AUDIO, language=Language.ENGLISH)

    assert outcome.status is TimingStatus.UNALIGNED
    assert outcome.span is None
    assert ReviewReason.UNALIGNED_WORD in outcome.review_reasons


def test_the_options_are_read_off_the_settings_without_importing_them():
    from audio_transcriber.settings import ProcessingSettings

    settings = ProcessingSettings()
    settings.forced_alignment_enabled = False

    assert TimingOptions.from_settings(settings).forced_alignment_enabled is False
    assert TimingOptions.from_settings().forced_alignment_enabled is True


def test_a_measured_phrase_reports_the_aligner_as_the_source_of_its_timing():
    aligner = FakeAligner([("can", 8.0, 8.3, 0.01), ("not", 8.35, 8.6, 0.01)])
    tokens = [final("can", 8.0, 8.6), final("not", 8.0, 8.6)]

    outcomes = realign(tokens, aligner, AUDIO, language=Language.ENGLISH)

    assert [outcome.source for outcome in outcomes] == [Provider.ELEVENLABS] * 2
    assert outcomes[0].span == AudioSpan(8.0, 8.3)
    assert outcomes[1].span.start == pytest.approx(8.35)
