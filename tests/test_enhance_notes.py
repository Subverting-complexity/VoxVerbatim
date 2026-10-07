"""Tests for the wording that explains the Enhance Audio settings.

These check the shape of the writing rather than the words themselves. A
summary that has grown into three paragraphs, or a note that would be read
out as fifteen separate lines, is a real fault: the first is read aloud on
every focus and the second is what a screen reader has to work through.
"""

from __future__ import annotations

import pytest

from vox_verbatim.ui.enhance_notes import (
    CEILING,
    FILES,
    LIMITER,
    MAXIMUM_GAIN,
    NOTES,
    OUTPUT_FOLDER,
    OUTPUT_FORMAT,
    REPLACE_EXISTING,
    TARGET_LOUDNESS,
    guide_text,
    note_for,
    note_text,
    summary_of,
)

EVERY_SETTING = (
    FILES,
    OUTPUT_FOLDER,
    TARGET_LOUDNESS,
    CEILING,
    MAXIMUM_GAIN,
    LIMITER,
    OUTPUT_FORMAT,
    REPLACE_EXISTING,
)

#: Long enough to be read out on focus without becoming a speech.
_LONGEST_SUMMARY = 200

#: Short enough that a note this size is certainly a stub.
_SHORTEST_NOTE = 400


def test_every_setting_in_the_dialog_is_explained():
    assert set(NOTES) == set(EVERY_SETTING)


@pytest.mark.parametrize("key", EVERY_SETTING)
def test_a_summary_is_short_enough_to_hear_on_every_focus(key):
    summary = summary_of(key)

    assert summary
    assert len(summary) <= _LONGEST_SUMMARY, f"{key} summary is {len(summary)} characters"
    assert "\n" not in summary
    assert summary.endswith(".")


@pytest.mark.parametrize("key", EVERY_SETTING)
def test_a_note_actually_explains_something(key):
    note = note_text(key)

    assert len(note) >= _SHORTEST_NOTE, f"{key} note is only {len(note)} characters"
    # More than one paragraph, because every one of these settings has a
    # what, a why and a what-to-do, and running them together helps nobody.
    assert "\n\n" in note


@pytest.mark.parametrize("key", EVERY_SETTING)
def test_a_note_reflows_to_whatever_width_it_is_shown_at(key):
    """One line per paragraph, so the widget does the wrapping.

    Wrapping baked into the text cannot follow the width of the panel or
    the size of the user's font, and a screen reader reads a text box a
    line at a time, so a paragraph split across eight lines is read as
    eight lines with a pause in each gap.
    """
    paragraphs = note_text(key).split("\n\n")

    for paragraph in paragraphs:
        assert "\n" not in paragraph, f"{key} has a paragraph broken across lines"
        assert paragraph == paragraph.strip()


@pytest.mark.parametrize("key", EVERY_SETTING)
def test_a_note_is_titled_and_the_title_is_what_the_control_is_called(key):
    note = note_for(key)

    assert note.title
    assert note.title == note.title.strip()
    assert not note.title.endswith(".")


def test_the_limiter_note_answers_the_question_it_exists_for():
    """The question that prompted all of this: what does ticking it do?"""
    note = note_text(LIMITER)

    assert "ceiling is enforced" in note
    assert "cough" in note, "the worked example is what makes this one land"
    assert "trade-off" in note
    assert "In practice" in note


def test_the_guide_holds_every_note_in_full():
    guide = guide_text()

    for key in EVERY_SETTING:
        assert note_for(key).title in guide
        for paragraph in note_text(key).split("\n\n"):
            assert paragraph in guide


def test_asking_for_a_note_that_does_not_exist_is_a_mistake_in_the_code():
    with pytest.raises(KeyError):
        note_for("no-such-setting")
