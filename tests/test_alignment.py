"""Tests for mapping every service's words onto the timed backbone.

No audio and no network. Every test builds the provider words by hand,
because what is being tested is the matching, and a handful of words with
a known answer says far more about it than a real transcript would.

Two things get more attention than the rest. The first is that a
formatting difference must come out as one correspondence rather than as a
deleted word plus an inserted one, which is the failure the costs are tuned
around. The second is the windowing, because that is what stops a long
recording taking quadratic time, and a module that quietly aligns an hour
as one sequence would pass every other test in this file.
"""

from __future__ import annotations

import time

import pytest

from audio_transcriber.transcription.alignment import (
    AlignedTable,
    AlignmentOptions,
    ProviderAlignment,
    align_result,
    align_sequences,
    build_aligned_table,
    choose_backbone,
)

# The windowing is the most important decision in the module, so it is
# tested directly rather than only through its effect on the output.
from audio_transcriber.transcription.alignment import _build_stream, _windows

from audio_transcriber.transcription.model import (
    AlignmentStatus,
    Provider,
    ProviderResult,
    ProviderToken,
)


def words(
    provider: Provider,
    text: str,
    timed: bool = False,
    speaker: str | None = None,
    start_at: float = 0.0,
) -> list[ProviderToken]:
    """Turn a sentence into one token per word, the way a service would.

    ``timed`` gives every word half a second of its own, which is enough
    for the tests that care whether timing survives and harmless for the
    ones that do not.
    """
    tokens: list[ProviderToken] = []
    for index, word in enumerate(text.split()):
        moment = start_at + index * 0.5
        tokens.append(
            ProviderToken(
                provider=provider,
                index=index,
                text=word,
                start=moment if timed else None,
                end=moment + 0.4 if timed else None,
                speaker=speaker,
                is_punctuation=False,
            )
        )
    return tokens


def statuses(alignment: ProviderAlignment) -> list[AlignmentStatus]:
    return [column.status for column in alignment.columns]


def texts(alignment: ProviderAlignment) -> list[tuple[str, str]]:
    return [(column.backbone_text, column.text) for column in alignment.columns]


# -- The straightforward cases -------------------------------------------


def test_two_services_that_agree_line_up_word_for_word():
    backbone = words(Provider.ELEVENLABS, "we signed the contract today", timed=True)
    other = words(Provider.OPENAI, "we signed the contract today")

    alignment = align_sequences(backbone, other)

    assert statuses(alignment) == [AlignmentStatus.EXACT] * 5
    assert alignment.quality == pytest.approx(1.0)
    assert alignment.agreement == pytest.approx(1.0)
    assert [column.position for column in alignment.columns] == [0, 1, 2, 3, 4]


def test_one_word_heard_differently_is_a_substitution_in_the_same_place():
    backbone = words(Provider.ELEVENLABS, "we signed the contract today", timed=True)
    other = words(Provider.OPENAI, "we signed the contact today")

    alignment = align_sequences(backbone, other)

    assert statuses(alignment) == [
        AlignmentStatus.EXACT,
        AlignmentStatus.EXACT,
        AlignmentStatus.EXACT,
        AlignmentStatus.SUBSTITUTION,
        AlignmentStatus.EXACT,
    ]
    disputed = alignment.columns[3]
    assert disputed.backbone_token.text == "contract"
    assert disputed.aligned_token.text == "contact"
    # The words either side matched, so this is a confident disagreement
    # about one word rather than a sign that the alignment lost its place.
    assert disputed.quality > 0.4


def test_a_word_only_one_service_heard_becomes_an_insertion():
    backbone = words(Provider.ELEVENLABS, "we signed contract today", timed=True)
    other = words(Provider.OPENAI, "we signed the contract today")

    alignment = align_sequences(backbone, other)

    inserted = [column for column in alignment.columns if column.is_insertion]
    assert len(inserted) == 1
    assert inserted[0].text == "the"
    assert inserted[0].status is AlignmentStatus.INSERTION
    # It sits before the backbone word it came in front of.
    assert inserted[0].position == 2
    assert inserted[0].backbone_token is None


def test_a_word_a_service_missed_becomes_a_deletion():
    backbone = words(Provider.ELEVENLABS, "we signed the contract today", timed=True)
    other = words(Provider.OPENAI, "we signed contract today")

    alignment = align_sequences(backbone, other)

    deleted = [column for column in alignment.columns if column.is_deletion]
    assert len(deleted) == 1
    assert deleted[0].backbone_token.text == "the"
    assert deleted[0].status is AlignmentStatus.DELETION
    assert deleted[0].text == ""


# -- The cases the costs are tuned for -----------------------------------


@pytest.mark.parametrize(
    ("backbone_word", "other_word"),
    [
        ("Contract", "contract"),
        ("contract,", "contract"),
        ("twenty-five", "25"),
        ("Jürgen", "Jurgen"),
        ("dont", "don't"),
    ],
)
def test_a_formatting_difference_is_one_correspondence_and_not_two_errors(
    backbone_word, other_word
):
    """This is the failure the whole cost table is arranged to prevent.

    If a substitution cost more than an insertion plus a deletion, the
    cheapest path through a pair of words that are merely written
    differently would be to throw one away and invent the other, and the
    fact that they are the same word would be lost for good.
    """
    backbone = words(Provider.ELEVENLABS, f"we signed the {backbone_word} today", timed=True)
    other = words(Provider.OPENAI, f"we signed the {other_word} today")

    alignment = align_sequences(backbone, other)

    assert len(alignment.columns) == 5
    matched = alignment.columns[3]
    assert matched.status is AlignmentStatus.FORMAT_EQUIVALENT
    assert matched.equivalence.is_equivalent
    assert matched.backbone_token is not None and matched.aligned_token is not None


def test_two_words_against_one_are_a_merge_rather_than_a_stray_insertion():
    backbone = words(Provider.ELEVENLABS, "the data base was updated", timed=True)
    other = words(Provider.OPENAI, "the database was updated")

    alignment = align_sequences(backbone, other)

    assert statuses(alignment) == [
        AlignmentStatus.EXACT,
        AlignmentStatus.MERGE,
        AlignmentStatus.EXACT,
        AlignmentStatus.EXACT,
    ]
    merged = alignment.columns[1]
    assert merged.backbone_text == "data base"
    assert merged.text == "database"
    assert merged.backbone_indices == (1, 2)
    # Both backbone words are covered, so the timing rules can give the one
    # final word the span of both.
    assert [token.start for token in merged.backbone_tokens] == [0.5, 1.0]


def test_one_word_against_two_is_a_split():
    backbone = words(Provider.ELEVENLABS, "she cannot attend", timed=True)
    other = words(Provider.OPENAI, "she can not attend")

    alignment = align_sequences(backbone, other)

    assert statuses(alignment) == [
        AlignmentStatus.EXACT,
        AlignmentStatus.SPLIT,
        AlignmentStatus.EXACT,
    ]
    split = alignment.columns[1]
    assert split.backbone_text == "cannot"
    assert split.text == "can not"
    assert len(split.aligned_tokens) == 2


def test_a_number_spread_over_several_words_still_merges():
    backbone = words(Provider.ELEVENLABS, "it cost twenty five thousand euro", timed=True)
    other = words(Provider.OPENAI, "it cost 25000 euro")

    alignment = align_sequences(backbone, other)

    merged = [column for column in alignment.columns if column.status is AlignmentStatus.MERGE]
    assert len(merged) == 1
    assert merged[0].backbone_text == "twenty five thousand"
    assert merged[0].text == "25000"


def test_costs_that_would_lose_a_correspondence_are_refused():
    with pytest.raises(ValueError, match="insertion plus a deletion"):
        AlignmentOptions(maximum_substitution_cost=2.5)
    with pytest.raises(ValueError, match="equivalent match"):
        AlignmentOptions(minimum_substitution_cost=0.01)


# -- Punctuation ---------------------------------------------------------


def test_punctuation_takes_no_part_in_the_matching_but_is_not_thrown_away():
    """Services disagree about punctuation constantly.

    Aligning a full stop against a comma and calling it a substitution
    would fill the review queue with nothing. The entries are still kept,
    because they carry timing of their own.
    """
    backbone = [
        ProviderToken(Provider.ELEVENLABS, 0, "yes", 0.0, 0.4),
        ProviderToken(Provider.ELEVENLABS, 1, ",", 0.4, 0.5, is_punctuation=True),
        ProviderToken(Provider.ELEVENLABS, 2, "today", 0.5, 1.0),
        ProviderToken(Provider.ELEVENLABS, 3, ".", 1.0, 1.1, is_punctuation=True),
    ]
    other = [
        ProviderToken(Provider.OPENAI, 0, "yes"),
        ProviderToken(Provider.OPENAI, 1, "today"),
        ProviderToken(Provider.OPENAI, 2, "!", is_punctuation=True),
    ]

    alignment = align_sequences(backbone, other)

    assert statuses(alignment) == [AlignmentStatus.EXACT, AlignmentStatus.EXACT]
    assert [mark.text for mark in alignment.columns[1].punctuation] == ["!"]

    table = build_aligned_table(
        ProviderResult(provider=Provider.ELEVENLABS, tokens=backbone),
        [ProviderResult(provider=Provider.OPENAI, tokens=other)],
    )
    first_row, second_row = table.rows
    assert [mark.text for mark in first_row.punctuation] == [","]
    assert [mark.text for mark in second_row.punctuation] == ["."]


# -- A service that returned nothing -------------------------------------


def test_a_service_that_returned_nothing_leaves_every_word_a_deletion():
    """A failed service must not cost the user their transcript.

    Its alignment has the same shape as everyone else's, so nothing
    downstream has to know that it failed, and the review rules can see
    plainly that it had no opinion about any of these words.
    """
    backbone = ProviderResult(
        provider=Provider.ELEVENLABS,
        tokens=words(Provider.ELEVENLABS, "we signed the contract today", timed=True),
    )
    silent = ProviderResult(
        provider=Provider.MICROSOFT,
        tokens=[],
        error="The service did not answer.",
    )

    alignment = align_result(backbone, silent)

    assert statuses(alignment) == [AlignmentStatus.DELETION] * 5
    assert alignment.is_empty
    assert alignment.agreement == 0.0

    table = build_aligned_table(backbone, [silent])
    assert table.missing_providers == (Provider.MICROSOFT,)
    assert len(table.rows) == 5
    assert table.rows[0].column_for(Provider.MICROSOFT).text == ""


def test_the_provider_has_to_be_given_when_there_are_no_words_to_read_it_from():
    with pytest.raises(ValueError, match="has to be given"):
        align_sequences(words(Provider.ELEVENLABS, "a word"), [])


# -- Choosing the backbone -----------------------------------------------


def test_the_timed_service_is_the_backbone():
    results = [
        ProviderResult(
            provider=Provider.ELEVENLABS,
            tokens=words(Provider.ELEVENLABS, "we signed the contract", timed=True),
        ),
        ProviderResult(
            provider=Provider.OPENAI,
            tokens=words(Provider.OPENAI, "we signed the contract"),
        ),
    ]

    assert choose_backbone(results) is Provider.ELEVENLABS


def test_another_service_takes_over_when_the_usual_backbone_fails():
    """The specification hands timing and speakers to AssemblyAI here.

    Nothing in the alignment assumes ElevenLabs, so the whole pipeline
    carries on with a different service holding the timeline.
    """
    results = [
        ProviderResult(provider=Provider.ELEVENLABS, tokens=[], error="Scribe failed."),
        ProviderResult(
            provider=Provider.OPENAI,
            tokens=words(Provider.OPENAI, "we signed the contract"),
        ),
        ProviderResult(
            provider=Provider.ASSEMBLYAI,
            tokens=words(Provider.ASSEMBLYAI, "we signed the contract", timed=True),
        ),
    ]

    backbone = choose_backbone(results)
    assert backbone is Provider.ASSEMBLYAI

    table = build_aligned_table(results[2], [results[1], results[0]])
    assert table.backbone_provider is Provider.ASSEMBLYAI
    assert statuses(table.alignment_for(Provider.OPENAI)) == [AlignmentStatus.EXACT] * 4


def test_a_service_without_timings_can_still_hold_the_sequence_together():
    """Better a transcript with no spans than no transcript at all."""
    results = [
        ProviderResult(provider=Provider.ELEVENLABS, tokens=[], error="Scribe failed."),
        ProviderResult(
            provider=Provider.OPENAI,
            tokens=words(Provider.OPENAI, "we signed the contract today"),
        ),
        ProviderResult(
            provider=Provider.MICROSOFT,
            tokens=words(Provider.MICROSOFT, "we signed"),
        ),
    ]

    assert choose_backbone(results) is Provider.OPENAI


def test_nothing_can_be_the_backbone_when_nothing_answered():
    assert choose_backbone([]) is None
    assert (
        choose_backbone([ProviderResult(provider=Provider.OPENAI, tokens=[], error="No.")])
        is None
    )


# -- The table the reconciliation rules read -----------------------------


def test_the_table_puts_every_service_beside_the_backbone_word():
    backbone = ProviderResult(
        provider=Provider.ELEVENLABS,
        tokens=words(Provider.ELEVENLABS, "we signed the contract today", timed=True),
    )
    openai = ProviderResult(
        provider=Provider.OPENAI,
        tokens=words(Provider.OPENAI, "we signed the contact today"),
    )
    microsoft = ProviderResult(
        provider=Provider.MICROSOFT,
        tokens=words(Provider.MICROSOFT, "we signed the contract now"),
    )

    table = build_aligned_table(backbone, [openai, microsoft, backbone])

    assert isinstance(table, AlignedTable)
    assert table.providers == (Provider.OPENAI, Provider.MICROSOFT)
    assert len(table.rows) == 5

    disputed = table.row_for_backbone(3)
    assert disputed.backbone_token.text == "contract"
    assert disputed.texts == {Provider.OPENAI: "contact", Provider.MICROSOFT: "contract"}
    assert disputed.quality == min(column.quality for column in disputed.columns)

    # Both services disagree with the backbone in exactly one place, but
    # not equally convincingly. "contact" for "contract" is what a
    # mishearing looks like, so the words are confidently the same word;
    # "now" for "today" could as easily mean the alignment has put two
    # unrelated words together, and it scores lower for that.
    openai_column = table.row_for_backbone(3).column_for(Provider.OPENAI)
    microsoft_column = table.row_for_backbone(4).column_for(Provider.MICROSOFT)
    assert openai_column.status is AlignmentStatus.SUBSTITUTION
    assert microsoft_column.status is AlignmentStatus.SUBSTITUTION
    assert openai_column.quality > microsoft_column.quality


def test_an_inserted_word_gets_a_row_of_its_own_before_the_word_it_precedes():
    backbone = ProviderResult(
        provider=Provider.ELEVENLABS,
        tokens=words(Provider.ELEVENLABS, "we signed contract today", timed=True),
    )
    openai = ProviderResult(
        provider=Provider.OPENAI,
        tokens=words(Provider.OPENAI, "we signed the contract today"),
    )

    table = build_aligned_table(backbone, [openai])

    inserted = [row for row in table.rows if row.is_insertion_row]
    assert len(inserted) == 1
    assert inserted[0].texts == {Provider.OPENAI: "the"}
    assert table.rows.index(inserted[0]) == 2
    assert table.rows[3].backbone_token.text == "contract"


# -- Windowing -----------------------------------------------------------


def long_transcript(sentences: int) -> str:
    """A transcript with an ordinary mixture of repeated and rare words.

    Every sentence has the same shape and one word that occurs nowhere
    else, which is what real speech gives the anchor finder to work with.
    """
    return " ".join(
        f"and then the {name} was mentioned in the meeting"
        for name in (f"witness{number}" for number in range(sentences))
    )


def test_a_long_recording_is_aligned_in_pieces_and_not_as_one_sequence():
    """The single most important property in the module.

    Four hundred sentences is about four thousand words, which as one
    matrix would be sixteen million cells. Aligned between anchors it is a
    few thousand tiny ones, and the whole thing finishes in well under a
    second. The generous time limit is there to catch a return to
    quadratic behaviour, not to measure the machine.
    """
    text = long_transcript(400)
    backbone = words(Provider.ELEVENLABS, text, timed=True)
    other_text = text.replace("witness7 ", "witness7 clearly ").replace(
        "witness300", "witness301"
    )
    other = words(Provider.OPENAI, other_text)

    started = time.monotonic()
    alignment = align_sequences(backbone, other)
    elapsed = time.monotonic() - started

    assert elapsed < 20.0
    assert len(backbone) == 3600
    assert alignment.agreement > 0.99
    found = {
        column.status
        for column in alignment.columns
        if column.status is not AlignmentStatus.EXACT
    }
    assert found == {AlignmentStatus.INSERTION, AlignmentStatus.SUBSTITUTION}


def test_every_window_stays_within_its_size_limit():
    options = AlignmentOptions(maximum_window_words=50)
    text = long_transcript(60)
    backbone = _build_stream(words(Provider.ELEVENLABS, text, timed=True), options)
    other = _build_stream(words(Provider.OPENAI, text), options)

    windows = _windows(backbone, other, options)

    assert len(windows) > 1
    for window in windows:
        assert window.backbone_length <= options.maximum_window_words
        assert window.aligned_length <= options.maximum_window_words


def test_a_stretch_with_no_landmarks_at_all_is_still_cut_up():
    """Repeated words give the anchor finder nothing unique to pin to.

    When that happens the region has to be cut somewhere anyway, at a
    sentence end where there is one. Columns from such a cut are marked
    down, because a cut in the wrong place misaligns the words either side
    of the seam and the review rules should be told.
    """
    options = AlignmentOptions(maximum_window_words=20)
    text = " ".join(["the same words again and again."] * 30)
    backbone_tokens = words(Provider.ELEVENLABS, text, timed=True)
    other_tokens = words(Provider.OPENAI, text)
    backbone = _build_stream(backbone_tokens, options)
    other = _build_stream(other_tokens, options)

    windows = _windows(backbone, other, options)

    assert len(windows) > 1
    assert all(window.forced for window in windows)
    for window in windows:
        assert window.backbone_length <= options.maximum_window_words

    alignment = align_sequences(backbone_tokens, other_tokens, options=options)
    assert alignment.agreement == pytest.approx(1.0)
    # Every column came out of a forced cut, so none of them claims to be
    # perfectly trustworthy.
    assert all(column.quality < 1.0 for column in alignment.columns)


def test_a_silence_and_a_change_of_speaker_are_natural_places_to_cut():
    options = AlignmentOptions(maximum_window_words=6)
    first = words(Provider.ELEVENLABS, "the same words again", timed=True, speaker="a")
    second = words(
        Provider.ELEVENLABS, "the same words again", timed=True, speaker="b", start_at=5.0
    )
    tokens = list(first) + [
        ProviderToken(
            provider=token.provider,
            index=len(first) + index,
            text=token.text,
            start=token.start,
            end=token.end,
            speaker=token.speaker,
        )
        for index, token in enumerate(second)
    ]
    stream = _build_stream(tokens, options)

    assert stream.boundary_after[3] is True
    windows = _windows(stream, stream, options)
    assert [window.backbone_start for window in windows] == [0, 4]
