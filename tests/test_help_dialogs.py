"""Tests for the keyboard shortcut list.

This text is the only place the application explains itself, and somebody
working by ear has nothing else to go on. A line describing behaviour the
application no longer has is therefore a functional defect rather than a
stale comment: it does not merely fail to help, it sends the reader off to
press a key that does something else and leave them thinking the window is
broken.

What is held down here is the handful of statements that were actually wrong
after the review became a thing done to a whole folder, plus the rule that
every key the review window binds is written down somewhere.
"""

from __future__ import annotations

import re
from pathlib import Path

from vox_verbatim.ui import review_window
from vox_verbatim.ui.help_dialogs import keyboard_shortcuts_text


def review_section() -> str:
    """The part of the list describing the review window, and nothing else."""
    text = keyboard_shortcuts_text()
    start = text.index("In the review window")
    end = text.index("Settings and closing")
    return text[start:end]


def test_reviewing_covers_the_folder_rather_than_the_highlighted_file():
    """The behaviour changed, so the sentence describing it had to.

    Somebody told that Ctrl+R reviews the highlighted file will arrow down to
    the recording they care about first, and be surprised to be shown words
    from four other recordings with no file name against them.
    """
    text = keyboard_shortcuts_text()

    assert "Review the whole folder" in text
    assert "Review the transcript of the highlighted file" not in text


def test_the_inner_loop_is_described_the_right_way_round():
    """F3 walks occurrences of one word; Ctrl+F3 walks words.

    Described the other way round, eight presses of F3 walk somebody through
    eight recordings of the same surname while they wait for the next word,
    and nothing tells them why.
    """
    section = review_section()

    assert "Go to the next word." in section
    assert "Ctrl+F3" in section
    assert "occurrence of that word" in section
    assert "previous word needing review" not in section


def review_window_shortcuts() -> set[str]:
    """Every key the review window binds, read out of its own source.

    Read from the text of the module rather than from a window that has been
    built, because building one needs a project store, a folder of
    recordings and an audio player, and this question is about what the
    module says rather than about what any particular window is doing. It
    also means a shortcut added tomorrow is caught by the test that already
    exists, which is the only way a list like this stays true.

    Both spellings Qt allows are collected: ``QKeySequence("Ctrl+L")`` and
    ``QKeySequence(Qt.Key.Key_F2)``.
    """
    source = (
        Path(review_window.__file__).read_text(encoding="utf-8")
    )
    found = set(re.findall(r'QKeySequence\("([^"]+)"\)', source))
    found.update(
        f"F{number}" for number in re.findall(r"QKeySequence\(Qt\.Key\.Key_F(\d+)\)", source)
    )
    return found


def test_every_key_the_review_window_binds_is_written_down():
    """A shortcut nobody has been told about is a shortcut nobody uses.

    Three of them were bound and never mentioned anywhere: the two that run
    the analysis and group the words again, and the one that takes an
    occurrence out of its word. Checking the whole set rather than those
    three means the next key somebody adds is caught here rather than by a
    reviewer noticing.

    Each key must appear exactly as Qt spells it, so that "Ctrl+Shift+F3" is
    not counted as documented merely because "F3" is written somewhere.
    """
    section = review_section()
    missing = sorted(key for key in review_window_shortcuts() if key not in section)

    assert missing == [], f"bound in the review window and not documented: {missing}"


def test_the_two_meanings_of_f5_are_told_apart():
    """It reads the folder again in one window and plays a word in the other.

    A reader who has learned the file list first will otherwise assume the
    review window works the same way, press F5 expecting nothing much, and be
    played audio they were not ready for.
    """
    section = review_section()

    assert "in the file list it reads" in section


def test_the_lists_are_called_words_rather_than_groups():
    """The menus say "word" and the screen reader says "word".

    Help text calling the same thing a group sends somebody looking for a
    control that does not exist under that name.
    """
    section = review_section()

    assert "group" not in section.replace("grouping tolerance", "").replace(
        "Group the words again", ""
    )


def test_no_user_facing_text_promises_a_later_phase():
    """Transcription shipped, so text saying it is coming later is wrong.

    The About box and the file list's tooltip both said so, and a screen
    reader reads both aloud.
    """
    package = Path(review_window.__file__).resolve().parent.parent
    # The whole text is searched, so a phrase split across a line break in
    # a docstring is still found.
    stale = [
        str(path.relative_to(package))
        for path in package.rglob("*.py")
        if re.search(r"later\s+(phase|version)", path.read_text(encoding="utf-8"))
    ]

    assert stale == []
