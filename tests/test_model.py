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

from audio_transcriber.transcription.model import Confidence, FinalToken, Transcript
from audio_transcriber.transcription.normalise import normalise


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
