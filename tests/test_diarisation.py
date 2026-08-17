"""Tests for speaker attribution and its second opinion.

Everything here is data. Who was speaking is decided from labels and spans,
and the two questions worth being careful about are both answered without
any audio: whether two services' labels are matched by the audio they cover
rather than by the order they appear in, and whether changing a speaker
leaves the word itself completely alone.

That second one has a test of its own for a reason. Text, timing and speaker
are three separate answers with three separate sources, and a speaker
correction that quietly moved a word or altered its spelling would undo the
design the whole application rests on.
"""

from __future__ import annotations

from audio_transcriber.transcription.diarisation import (
    DiarisationConcern,
    DiarisationOptions,
    SpeakerDecision,
    SpeakerTurn,
    apply_speaker,
    apply_speakers,
    find_speaker_concerns,
    map_speaker_labels,
    reconcile_speakers,
    speaker_turns,
    speakers_to_expect,
    turns_from_result,
)
from audio_transcriber.transcription.escalation import EscalationReason
from audio_transcriber.transcription.model import (
    AudioSpan,
    Confidence,
    FinalToken,
    Provider,
    ProviderResult,
    ProviderToken,
    ReviewReason,
    TimingStatus,
)


def turn(speaker: str, start: float, end: float, *token_ids: str) -> SpeakerTurn:
    return SpeakerTurn(speaker, AudioSpan(start, end), token_ids)


def said(speaker: str, text: str, start: float, end: float, index: int = 0) -> ProviderToken:
    return ProviderToken(
        provider=Provider.ELEVENLABS,
        index=index,
        text=text,
        start=start,
        end=end,
        speaker=speaker,
    )


def final(text: str, start: float, end: float, speaker: str) -> FinalToken:
    return FinalToken(
        text=text,
        start=start,
        end=end,
        speaker=speaker,
        speaker_source=Provider.ELEVENLABS,
        speaker_confidence=Confidence.REVIEW_SUGGESTED,
        timing_status=TimingStatus.EXACT_PROVIDER_TIME,
        timing_source=Provider.ELEVENLABS,
        timing_confidence=Confidence.HIGH,
        text_confidence=Confidence.HIGH,
    )


# -- Reading the speakers out of what a service returned -----------------


def test_consecutive_words_by_one_person_become_one_turn():
    words = [
        said("speaker_0", "good", 0.0, 0.4, 0),
        said("speaker_0", "morning", 0.5, 1.1, 1),
        said("speaker_1", "morning", 1.4, 2.0, 2),
        said("speaker_0", "shall", 2.2, 2.5, 3),
    ]

    turns = speaker_turns(words)

    assert [(one.speaker, one.span.start, one.span.end) for one in turns] == [
        ("speaker_0", 0.0, 1.1),
        ("speaker_1", 1.4, 2.0),
        ("speaker_0", 2.2, 2.5),
    ]


def test_a_word_with_no_time_behind_it_is_not_part_of_any_turn():
    """A turn is a stretch of audio, and that word is not anywhere."""
    words = [
        said("speaker_0", "good", 0.0, 0.4, 0),
        ProviderToken(Provider.OPENAI, 1, "morning", speaker="speaker_0"),
    ]

    assert [one.span.end for one in speaker_turns(words)] == [0.4]


def test_turns_are_read_from_a_result_without_its_punctuation():
    result = ProviderResult(
        provider=Provider.ELEVENLABS,
        tokens=[
            said("speaker_0", "yes", 0.0, 0.4, 0),
            ProviderToken(
                Provider.ELEVENLABS, 1, ",", start=0.4, end=0.45,
                speaker="speaker_0", is_punctuation=True
            ),
        ],
    )

    assert [one.duration for one in turns_from_result(result)] == [0.4]


# -- Keeping the speaker space to a size the recording supports ----------


def test_the_expected_count_anchors_how_many_speakers_may_be_found():
    assert speakers_to_expect(2, ["A", "B"]) == 2
    assert speakers_to_expect(1, []) == 1


def test_the_expected_count_is_read_off_the_recording_configuration():
    from audio_transcriber.transcription.model import RecordingConfiguration

    options = DiarisationOptions.from_configuration(
        RecordingConfiguration(expected_speaker_count=3)
    )

    assert options.expected_speaker_count == 3
    assert DiarisationOptions.from_configuration().expected_speaker_count == 1


def test_the_count_rises_only_as_far_as_the_backbone_itself_heard():
    """Bounded by somebody's evidence rather than by a service's imagination."""
    assert speakers_to_expect(1, ["speaker_0", "speaker_1", "speaker_2"]) == 3
    assert speakers_to_expect(3, ["speaker_0"]) == 3


# -- Deciding when to ask again ------------------------------------------


def test_more_people_than_expected_are_flagged_and_the_quietest_one_is_the_surplus():
    """The invented speaker is almost always the one with the least speech."""
    turns = [
        turn("speaker_0", 0.0, 20.0),
        turn("speaker_1", 20.0, 38.0),
        turn("speaker_2", 38.0, 40.0),
    ]

    concerns = find_speaker_concerns(turns, options=DiarisationOptions(expected_speaker_count=2))

    assert [concern.concern for concern in concerns] == [
        DiarisationConcern.UNEXPECTED_SPEAKERS
    ]
    assert concerns[0].speakers == ("speaker_2",)


def test_two_people_where_two_are_expected_raise_nothing():
    turns = [turn("speaker_0", 0.0, 20.0), turn("speaker_1", 20.0, 40.0)]

    assert find_speaker_concerns(turns, options=DiarisationOptions(expected_speaker_count=2)) == []


def test_people_speaking_over_each_other_are_flagged():
    turns = [turn("speaker_0", 0.0, 10.0), turn("speaker_1", 9.2, 20.0)]

    concerns = find_speaker_concerns(turns, options=DiarisationOptions(expected_speaker_count=2))

    assert concerns[0].concern is DiarisationConcern.OVERLAPPING_SPEECH
    assert concerns[0].span == AudioSpan(9.2, 10.0)


def test_a_single_very_short_utterance_is_flagged_on_its_own():
    turns = [turn("speaker_0", 0.0, 10.0), turn("speaker_1", 10.2, 10.5),
             turn("speaker_0", 11.0, 20.0)]

    concerns = find_speaker_concerns(turns, options=DiarisationOptions(expected_speaker_count=2))

    assert [concern.concern for concern in concerns] == [
        DiarisationConcern.VERY_SHORT_UTTERANCE
    ]
    assert concerns[0].span == AudioSpan(10.2, 10.5)


def test_a_run_of_short_turns_is_one_question_about_the_flipping():
    """The identity is changing faster than people take turns."""
    turns = [
        turn("speaker_0", 0.0, 10.0),
        turn("speaker_1", 10.1, 10.4),
        turn("speaker_0", 10.5, 10.8),
        turn("speaker_1", 10.9, 11.2),
        turn("speaker_0", 11.5, 20.0),
    ]

    concerns = find_speaker_concerns(turns, options=DiarisationOptions(expected_speaker_count=2))

    assert [concern.concern for concern in concerns] == [
        DiarisationConcern.FREQUENT_SPEAKER_CHANGES
    ]
    assert concerns[0].span == AudioSpan(10.1, 11.2)


def test_a_disputed_word_on_a_speaker_boundary_is_flagged():
    """The same few seconds answer both doubts at once."""
    turns = [turn("speaker_0", 0.0, 10.0), turn("speaker_1", 10.1, 20.0)]

    concerns = find_speaker_concerns(
        turns,
        disputed_spans=[AudioSpan(9.9, 10.3)],
        options=DiarisationOptions(expected_speaker_count=2),
    )

    assert [concern.concern for concern in concerns] == [
        DiarisationConcern.DISPUTE_NEAR_BOUNDARY
    ]
    assert concerns[0].span == AudioSpan(9.9, 10.3)


def test_a_disputed_word_far_from_any_boundary_is_not():
    turns = [turn("speaker_0", 0.0, 10.0), turn("speaker_1", 10.1, 20.0)]

    concerns = find_speaker_concerns(
        turns,
        disputed_spans=[AudioSpan(4.0, 4.4)],
        options=DiarisationOptions(expected_speaker_count=2),
    )

    assert concerns == []


def test_a_concern_becomes_something_escalation_can_group_and_send():
    turns = [turn("speaker_0", 0.0, 10.0, "one"), turn("speaker_1", 9.2, 20.0, "two")]

    concern = find_speaker_concerns(
        turns, options=DiarisationOptions(expected_speaker_count=2)
    )[0]
    dispute = concern.as_dispute()

    assert dispute.span == concern.span
    assert dispute.reasons == (EscalationReason.OVERLAPPING_SPEECH,)
    assert dispute.token_ids == ("one", "two")


# -- Matching one service's labels onto the other's ----------------------


def test_labels_are_matched_by_the_audio_they_cover_and_not_by_their_order():
    """Matching by position would attribute every word to the wrong person.

    The second service labels the man who speaks first "B", because it
    happened to hear the woman's opening line and the backbone did not. The
    two lists are in opposite orders and describe exactly the same two
    people.
    """
    backbone = [
        turn("speaker_0", 0.0, 10.0),
        turn("speaker_1", 10.0, 20.0),
        turn("speaker_0", 20.0, 30.0),
    ]
    other = [turn("B", 0.0, 10.0), turn("A", 10.0, 20.0), turn("B", 20.0, 30.0)]

    mapping = map_speaker_labels(backbone, other)

    assert mapping.mapping == {"B": "speaker_0", "A": "speaker_1"}
    assert mapping.unmatched == ()
    assert mapping.coverage["B"] == 1.0


def test_a_label_that_covers_two_people_is_matched_to_the_one_it_covers_most():
    backbone = [turn("speaker_0", 0.0, 18.0), turn("speaker_1", 18.0, 20.0)]
    other = [turn("A", 0.0, 20.0)]

    mapping = map_speaker_labels(backbone, other)

    assert mapping.mapping == {"A": "speaker_0"}
    assert mapping.coverage["A"] == 0.9
    assert mapping.confidence_for("A") is Confidence.HIGH


def test_a_label_spread_across_everybody_is_matched_only_weakly():
    backbone = [turn("speaker_0", 0.0, 10.0), turn("speaker_1", 10.0, 20.0)]
    other = [turn("A", 5.0, 15.0)]

    mapping = map_speaker_labels(backbone, other)

    assert mapping.confidence_for("A") is Confidence.REVIEW_SUGGESTED


def test_a_speaker_only_one_service_heard_is_reported_rather_than_forced_onto_somebody():
    backbone = [turn("speaker_0", 0.0, 10.0)]
    other = [turn("A", 0.0, 10.0), turn("B", 12.0, 20.0)]

    mapping = map_speaker_labels(backbone, other)

    assert mapping.mapping == {"A": "speaker_0"}
    assert mapping.unmatched == ("B",)
    assert mapping.label_for("B") is None


# -- Changing a speaker, and only a speaker ------------------------------


def test_a_speaker_correction_leaves_the_text_and_the_timing_untouched():
    """The rule the whole module exists to keep.

    A person told that a word was reattributed must be able to trust that
    the word itself did not move and did not change.
    """
    token = final("Yes", 42.18, 42.41, "speaker_0")
    other = [turn("B", 42.0, 43.0)]
    mapping = map_speaker_labels([turn("speaker_1", 42.0, 43.0)], other)

    decision, = reconcile_speakers([token], other, mapping)
    apply_speaker(token, decision)

    assert decision.changed is True
    assert token.speaker == "speaker_1"
    assert token.speaker_source is Provider.ASSEMBLYAI
    assert token.text == "Yes"
    assert token.text_confidence is Confidence.HIGH
    assert (token.start, token.end) == (42.18, 42.41)
    assert token.timing_status is TimingStatus.EXACT_PROVIDER_TIME
    assert token.timing_source is Provider.ELEVENLABS
    assert token.timing_confidence is Confidence.HIGH
    assert ReviewReason.SPEAKER_UNCERTAIN in token.review_reasons


def test_two_services_agreeing_confirms_the_speaker_rather_than_changing_it():
    token = final("yes", 42.18, 42.41, "speaker_1")
    other = [turn("B", 42.0, 43.0)]
    mapping = map_speaker_labels([turn("speaker_1", 42.0, 43.0)], other)

    decision, = reconcile_speakers([token], other, mapping)
    apply_speaker(token, decision)

    assert decision.changed is False
    assert token.speaker == "speaker_1"
    assert token.speaker_confidence is Confidence.HIGH
    assert token.review_reasons == []


def test_a_speaker_the_backbone_never_had_flags_the_words_without_renaming_them():
    """An identifier from one service's world would explain nothing here."""
    token = final("yes", 42.18, 42.41, "speaker_0")
    other = [turn("A", 0.0, 10.0), turn("B", 42.0, 43.0)]
    mapping = map_speaker_labels([turn("speaker_0", 0.0, 10.0)], other)

    decision, = reconcile_speakers([token], other, mapping)
    apply_speaker(token, decision)

    assert token.speaker == "speaker_0"
    assert token.speaker_confidence is Confidence.REVIEW_REQUIRED
    assert ReviewReason.SPEAKER_UNCERTAIN in token.review_reasons


def test_a_word_the_second_opinion_did_not_cover_is_left_alone():
    token = final("yes", 5.0, 5.4, "speaker_0")
    other = [turn("A", 42.0, 43.0)]
    mapping = map_speaker_labels([turn("speaker_1", 42.0, 43.0)], other)

    assert reconcile_speakers([token], other, mapping) == []


def test_a_word_on_a_boundary_takes_the_turn_it_shares_the_most_audio_with():
    token = final("yes", 9.9, 10.4, "speaker_0")
    other = [turn("A", 0.0, 10.0), turn("B", 10.0, 20.0)]
    backbone = [turn("speaker_0", 0.0, 10.0), turn("speaker_1", 10.0, 20.0)]

    decision, = reconcile_speakers([token], other, map_speaker_labels(backbone, other))

    assert decision.speaker == "speaker_1"


def test_applying_a_set_of_decisions_counts_what_really_changed():
    first = final("yes", 1.0, 1.4, "speaker_0")
    second = final("no", 2.0, 2.4, "speaker_1")
    decisions = [
        SpeakerDecision(first.id, "speaker_1", Provider.ASSEMBLYAI,
                        Confidence.REVIEW_SUGGESTED, changed=True),
        SpeakerDecision(second.id, "speaker_1", Provider.ELEVENLABS,
                        Confidence.HIGH, changed=False),
    ]

    changed = apply_speakers([first, second], decisions)

    assert changed == 1
    assert first.speaker == "speaker_1"
    assert second.speaker == "speaker_1"
    assert (second.start, second.end) == (2.0, 2.4)
