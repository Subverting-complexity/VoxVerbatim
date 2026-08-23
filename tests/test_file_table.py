"""Tests for the audio file table: its contents, its check boxes, and its keyboard."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from vox_verbatim.audio.library import AudioFile
from vox_verbatim.ui.file_table import (
    COLUMN_DURATION,
    COLUMN_NAME,
    COLUMN_SIZE,
    AudioFileTableModel,
    AudioFileTableView,
)


def make_files(*names: str) -> list[AudioFile]:
    return [AudioFile(path=Path("C:/Audio") / name, size_bytes=1024 * 1024) for name in names]


def test_the_table_has_the_three_required_columns_in_order(qapp):
    model = AudioFileTableModel()
    model.set_files(make_files("one.m4a"))

    assert model.columnCount() == 3
    headers = [
        model.headerData(column, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
        for column in range(3)
    ]
    assert headers == ["File name", "Duration", "File size"]


def test_a_duration_reads_as_pending_until_it_arrives(qapp):
    model = AudioFileTableModel()
    model.set_files(make_files("one.m4a"))
    index = model.index(0, COLUMN_DURATION)

    assert model.data(index, Qt.ItemDataRole.DisplayRole) == "Reading..."

    model.update_durations([("one.m4a", 92.0)])

    assert model.data(index, Qt.ItemDataRole.DisplayRole) == "1:32"
    assert model.data(index, Qt.ItemDataRole.AccessibleTextRole) == "1 minute 32 seconds"


def test_a_duration_that_cannot_be_read_says_so(qapp):
    model = AudioFileTableModel()
    model.set_files(make_files("one.m4a"))
    model.update_durations([("one.m4a", None)])
    index = model.index(0, COLUMN_DURATION)

    assert model.data(index, Qt.ItemDataRole.DisplayRole) == "Unknown"
    assert model.data(index, Qt.ItemDataRole.AccessibleTextRole) == "duration is not known"


def test_sizes_are_spoken_in_whole_words(qapp):
    model = AudioFileTableModel()
    model.set_files(make_files("one.m4a"))
    index = model.index(0, COLUMN_SIZE)

    assert model.data(index, Qt.ItemDataRole.DisplayRole) == "1.0 MB"
    assert model.data(index, Qt.ItemDataRole.AccessibleTextRole) == "1.0 megabytes"


def test_only_the_name_column_carries_the_check_box(qapp):
    model = AudioFileTableModel()
    model.set_files(make_files("one.m4a"))

    name_flags = model.flags(model.index(0, COLUMN_NAME))
    duration_flags = model.flags(model.index(0, COLUMN_DURATION))

    assert name_flags & Qt.ItemFlag.ItemIsUserCheckable
    assert not (duration_flags & Qt.ItemFlag.ItemIsUserCheckable)


def test_checking_a_file_reports_it_and_keeps_table_order(qapp):
    model = AudioFileTableModel()
    model.set_files(make_files("a.m4a", "b.m4a", "c.m4a"))
    reported: list[tuple[str, bool]] = []
    model.fileChecked.connect(lambda name, checked: reported.append((name, checked)))

    model.set_checked(2, True)
    model.set_checked(0, True)

    assert reported == [("c.m4a", True), ("a.m4a", True)]
    assert model.checked_names() == ["a.m4a", "c.m4a"]
    assert model.is_checked(0)
    assert not model.is_checked(1)


def test_toggling_a_row_turns_the_check_box_on_and_off(qapp):
    model = AudioFileTableModel()
    model.set_files(make_files("a.m4a"))

    model.toggle_checked(0)
    assert model.data(model.index(0, COLUMN_NAME), Qt.ItemDataRole.CheckStateRole) == (
        Qt.CheckState.Checked
    )

    model.toggle_checked(0)
    assert model.data(model.index(0, COLUMN_NAME), Qt.ItemDataRole.CheckStateRole) == (
        Qt.CheckState.Unchecked
    )


def test_a_check_box_can_be_set_through_the_model(qapp):
    model = AudioFileTableModel()
    model.set_files(make_files("a.m4a"))

    assert model.setData(
        model.index(0, COLUMN_NAME), Qt.CheckState.Checked, Qt.ItemDataRole.CheckStateRole
    )
    assert model.checked_names() == ["a.m4a"]


def test_files_that_disappear_are_no_longer_checked(qapp):
    model = AudioFileTableModel()
    model.set_files(make_files("a.m4a", "b.m4a"))
    model.set_checked_names(["a.m4a", "b.m4a"])

    model.set_files(make_files("b.m4a"))

    assert model.checked_names() == ["b.m4a"]


def test_restoring_check_boxes_ignores_names_that_are_not_there(qapp):
    model = AudioFileTableModel()
    model.set_files(make_files("a.m4a"))

    model.set_checked_names(["a.m4a", "deleted.m4a"])

    assert model.checked_names() == ["a.m4a"]


def test_the_space_bar_asks_to_toggle_the_highlighted_row(qapp):
    model = AudioFileTableModel()
    model.set_files(make_files("a.m4a", "b.m4a"))
    view = AudioFileTableView()
    view.setModel(model)
    view.toggleCheckRequested.connect(model.toggle_checked)
    view.select_row(1)

    QApplication.sendEvent(
        view,
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier),
    )

    assert model.checked_names() == ["b.m4a"]


def test_tab_moves_out_of_the_table_rather_than_across_its_cells(qapp):
    view = AudioFileTableView()

    assert not view.tabKeyNavigation()
