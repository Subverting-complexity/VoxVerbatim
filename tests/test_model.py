"""Tests for the corrections a person makes to a finished transcript.

The interesting thing about a correction is not that the word changes. It is
everything that has to change with it and everything that must not. A word a
person retyped is high confidence, keeps what it said before so the change
can be explained and undone, and says nothing whatever about when the word
was spoken or who spoke it.

The comparison form is the part that is easy to forget, because nothing on
screen shows it. It is what every rule, lookup and grouping decision matches
on, so a corrected word still carrying the comparison form of the word it
replaced would be found by a search for the old spelling, and answered again
by a project rule that has already answered it. The visible symptom is a
correction that quietly reapplies itself every time the project is
reprocessed, which is exactly the kind of fault nobody reports because it
looks like nothing happened.
"""

from __future__ import annotations

from vox_verbatim.transcription.model import (
    Confidence,
    FinalToken,
    Provider,
    ProviderResult,
    ProviderToken,
    ReviewReason,
    ReviewStatus,
    Transcript,
)
from vox_verbatim.transcription.normalise import normalise


def _transcript(text: str = "Bosh") -> tuple[Transcript, FinalToken]:
    token = FinalToken()
    token.text = text
    token.normalised_text = normalise(text)
    token.text_confidence = Confidence.REVIEW_REQUIRED
    token.confidence_strength = 0.42
    token.speaker = "speaker_0"
    token.start = 12.0
    token.end = 12.4
    return Transcript(recording_name="Interview 01.m4a", tokens=[token]), token


def test_correcting_the_text_rebuilds_the_comparison_form() -> None:
    transcript, token = _transcript("Bosh")
    corrected = transcript.with_correction(token.id, text="Bosch")
    word = corrected.token_by_id(token.id)
    assert word is not None
    assert word.text == "Bosch"
    assert word.normalised_text == normalise("Bosch")
    # The old form is the one that would go on matching rules the project has
    # already answered, so it must be gone rather than merely accompanied.
    assert word.normalised_text != normalise("Bosh")


def test_the_word_it_replaced_is_kept() -> None:
    transcript, token = _transcript("Bosh")
    corrected = transcript.with_correction(token.id, text="Bosch")
    word = corrected.token_by_id(token.id)
    assert word is not None
    assert word.original_text == "Bosh"
    assert word.human_corrected is True
    assert word.text_corrected is True


def test_correcting_only_the_speaker_does_not_claim_the_text_was_replaced() -> None:
    """The distinction the whole ``text_corrected`` field exists for.

    Naming the speaker is a person's decision about the word, so
    ``human_corrected`` is right. It is not a statement about the spelling,
    and anything that read it as one would leave the word saying whatever the
    services said and beyond the reach of every replacement rule.
    """
    transcript, token = _transcript("Bosh")
    corrected = transcript.with_correction(token.id, speaker="speaker_1")
    word = corrected.token_by_id(token.id)
    assert word is not None
    assert word.speaker == "speaker_1"
    assert word.human_corrected is True
    assert word.text_corrected is False
    assert word.text == "Bosh"
    assert word.original_text is None


def test_correcting_the_text_to_what_it_already_said_changes_nothing() -> None:
    """Retyping the same word is not a replacement and must not read as one."""
    transcript, token = _transcript("Bosh")
    corrected = transcript.with_correction(token.id, text="Bosh")
    word = corrected.token_by_id(token.id)
    assert word is not None
    assert word.text_corrected is False
    assert word.original_text is None


def test_correcting_twice_still_remembers_what_the_services_said() -> None:
    """The original is what a service produced, not the previous correction.

    A person changing their mind must not overwrite the only record of what
    was actually heard, because that record is what the correction is
    explained against and what teaches anything downstream.
    """
    transcript, token = _transcript("Bosh")
    once = transcript.with_correction(token.id, text="Bosch")
    twice = once.with_correction(token.id, text="Bosche")
    word = twice.token_by_id(token.id)
    assert word is not None
    assert word.text == "Bosche"
    assert word.original_text == "Bosh"
    assert word.normalised_text == normalise("Bosche")


def test_a_correction_is_treated_as_the_strongest_evidence_there_is() -> None:
    transcript, token = _transcript("Bosh")
    corrected = transcript.with_correction(token.id, text="Bosch")
    word = corrected.token_by_id(token.id)
    assert word is not None
    assert word.text_confidence is Confidence.HIGH
    assert word.confidence_strength == 1.0


def test_correcting_the_text_leaves_the_timing_and_the_speaker_alone() -> None:
    transcript, token = _transcript("Bosh")
    corrected = transcript.with_correction(token.id, text="Bosch")
    word = corrected.token_by_id(token.id)
    assert word is not None
    assert word.speaker == "speaker_0"
    assert word.start == 12.0
    assert word.end == 12.4


def test_correcting_the_speaker_leaves_the_word_alone() -> None:
    """Including its comparison form, which is a fact about the text only."""
    transcript, token = _transcript("Bosh")
    corrected = transcript.with_correction(token.id, speaker="speaker_1")
    word = corrected.token_by_id(token.id)
    assert word is not None
    assert word.speaker == "speaker_1"
    assert word.text == "Bosh"
    assert word.normalised_text == normalise("Bosh")


def test_correcting_a_word_does_not_change_the_transcript_it_came_from() -> None:
    """The original object is evidence somebody may still be looking at."""
    transcript, token = _transcript("Bosh")
    transcript.with_correction(token.id, text="Bosch")
    assert transcript.tokens[0].text == "Bosh"
    assert transcript.tokens[0].normalised_text == normalise("Bosh")



# -- Finding a service's token by its number ------------------------------
#
# ``ProviderResult.token_at`` is asked once for every reference in every
# column by three different callers, so it answers from a lookup built the
# first time. The result is a mutable dataclass, and the lookup must not
# outlive the tokens it was built from.


def _word(index: int, text: str) -> ProviderToken:
    return ProviderToken(provider=Provider.OPENAI, index=index, text=text)


def test_a_token_is_found_by_the_number_its_service_gave_it() -> None:
    result = ProviderResult(provider=Provider.OPENAI, tokens=[_word(0, "we"), _word(1, "signed")])
    assert result.token_at(1) is result.tokens[1]
    assert result.token_at(0) is result.tokens[0]
    assert result.token_at(7) is None


def test_the_first_of_two_tokens_with_one_number_wins() -> None:
    """As it did when the lookup was a scan from the front."""
    first, second = _word(4, "first"), _word(4, "second")
    result = ProviderResult(provider=Provider.OPENAI, tokens=[first, second])
    assert result.token_at(4) is first


def test_the_lookup_follows_a_fresh_list_of_tokens() -> None:
    result = ProviderResult(provider=Provider.OPENAI, tokens=[_word(0, "old")])
    assert result.token_at(0).text == "old"

    result.tokens = [_word(0, "new"), _word(1, "word")]

    assert result.token_at(0).text == "new"
    assert result.token_at(1).text == "word"


def test_the_lookup_follows_tokens_added_to_or_taken_from_the_list() -> None:
    result = ProviderResult(provider=Provider.OPENAI, tokens=[_word(0, "we")])
    assert result.token_at(1) is None

    result.tokens.append(_word(1, "signed"))
    assert result.token_at(1).text == "signed"

    result.tokens.pop(0)
    assert result.token_at(0) is None
    assert result.token_at(1).text == "signed"


def test_a_list_replaced_by_an_equal_sized_one_is_not_mistaken_for_the_old() -> None:
    """Two lists of the same length at, possibly, the same address.

    The lookup keeps hold of the list it was built from, so a replacement
    that happens to be allocated where the old list was cannot pass for it.
    """
    result = ProviderResult(provider=Provider.OPENAI, tokens=[_word(0, "old")])
    assert result.token_at(0).text == "old"
    for _ in range(50):
        result.tokens = [_word(0, "new")]
        assert result.token_at(0).text == "new"
        result.tokens = [_word(0, "old")]
        assert result.token_at(0).text == "old"


def test_an_empty_result_answers_none_without_complaint() -> None:
    result = ProviderResult(provider=Provider.OPENAI, error="it fell over")
    assert result.token_at(0) is None


# -- The overall rating of a word with no time ---------------------------------


def _untimed(text_confidence: Confidence) -> FinalToken:
    token = FinalToken()
    token.text = "contract"
    token.text_confidence = text_confidence
    token.timing_confidence = Confidence.UNRESOLVED
    token.speaker_confidence = Confidence.HIGH
    return token


def test_an_agreed_word_with_no_time_is_rated_review_suggested() -> None:
    """Its text was chosen, so "Unresolved" would tell the reader something false."""
    token = _untimed(Confidence.HIGH)

    assert token.confidence is Confidence.REVIEW_SUGGESTED
    assert token.confidence.needs_review


def test_a_disputed_word_with_no_time_keeps_the_rating_its_text_gives() -> None:
    assert _untimed(Confidence.REVIEW_REQUIRED).confidence is Confidence.REVIEW_REQUIRED


def test_a_word_where_nothing_was_chosen_is_still_unresolved() -> None:
    """The words in square brackets are the ones "Unresolved" exists for."""
    assert _untimed(Confidence.UNRESOLVED).confidence is Confidence.UNRESOLVED


def test_a_doubtful_time_still_weighs_on_the_word_in_full() -> None:
    """Only a missing time is capped; a time that is there but doubtful is not."""
    token = _untimed(Confidence.HIGH)
    token.timing_confidence = Confidence.REVIEW_REQUIRED

    assert token.confidence is Confidence.REVIEW_REQUIRED


def test_a_time_a_person_rejected_is_not_capped() -> None:
    """The word keeps its numbers, so its time is rejected rather than missing."""
    token = _untimed(Confidence.HIGH)
    token.start = 30.0
    token.end = 30.5

    assert token.confidence is Confidence.UNRESOLVED


def test_the_counts_by_rating_follow_the_capped_rating() -> None:
    transcript = Transcript(
        recording_name="interview.wav",
        tokens=[
            _untimed(Confidence.HIGH),
            _untimed(Confidence.HIGH),
            _untimed(Confidence.UNRESOLVED),
        ]
    )

    counts = transcript.counts_by_confidence()

    assert counts[Confidence.REVIEW_SUGGESTED] == 2
    assert counts[Confidence.UNRESOLVED] == 1


# -- Which words belong in the word review list --------------------------


def test_a_word_in_doubt_only_for_its_speaker_needs_no_word_review():
    token = FinalToken(text="yes")
    token.flag(ReviewReason.SPEAKER_UNCERTAIN)

    assert token.needs_review is True
    assert token.needs_word_review is False
    assert token.has_speaker_doubt is True


def test_a_word_in_doubt_for_its_text_and_its_speaker_needs_word_review():
    token = FinalToken(text="fifteen")
    token.flag(ReviewReason.SPEAKER_UNCERTAIN)
    token.flag(ReviewReason.PROVIDER_DISAGREEMENT)

    assert token.needs_word_review is True
    assert token.has_speaker_doubt is True


def test_overlapping_speech_is_still_a_reason_to_review_the_word():
    """Two voices at once make the words harder to hear, not only their owner."""
    token = FinalToken(text="yes")
    token.flag(ReviewReason.OVERLAPPING_SPEECH)

    assert token.needs_word_review is True


def test_a_pending_word_with_no_reason_still_needs_word_review():
    token = FinalToken(text="yes", review_status=ReviewStatus.PENDING)

    assert token.needs_word_review is True
    assert FinalToken(text="settled").needs_word_review is False


def test_the_word_review_tokens_leave_out_speaker_only_doubts():
    speaker_only = FinalToken(text="yes")
    speaker_only.flag(ReviewReason.SPEAKER_UNCERTAIN)
    text_doubt = FinalToken(text="fifteen")
    text_doubt.flag(ReviewReason.NUMERIC_DISAGREEMENT)
    transcript = Transcript(recording_name="a", tokens=[speaker_only, text_doubt])

    assert transcript.review_tokens == [speaker_only, text_doubt]
    assert transcript.word_review_tokens == [text_doubt]
