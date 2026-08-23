"""Tests for the small JSON files the application keeps for itself.

Most of these are about the one thing that goes wrong with a save in real
life, which is not the data and not the disk but another program holding the
old file open. On Windows the move into place then raises PermissionError,
and what matters is that the finished data is neither thrown away nor left
under a name nobody would look for.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from vox_verbatim import json_store
from vox_verbatim.json_store import (
    REPLACE_ATTEMPTS,
    read_json_object,
    unsaved_copy_path,
    write_json_object,
)


def _replace_that_refuses(times: int, monkeypatch):
    """An ``os.replace`` that raises PermissionError for its first ``times`` calls."""
    real = os.replace
    calls: list[int] = []

    def replace(source, destination):
        calls.append(1)
        if len(calls) <= times:
            raise PermissionError(13, "The process cannot access the file")
        return real(source, destination)

    monkeypatch.setattr(json_store.os, "replace", replace)
    return calls


def test_a_save_lands_and_reads_back(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    assert write_json_object(path, {"a": 1}) is True
    assert read_json_object(path) == {"a": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_a_file_held_open_for_a_moment_is_tried_again(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "settings.json"
    path.write_text('{"a": 0}', encoding="utf-8")
    calls = _replace_that_refuses(2, monkeypatch)
    waits: list[float] = []

    assert write_json_object(path, {"a": 1}, sleep=waits.append) is True

    assert len(calls) == 3
    assert read_json_object(path) == {"a": 1}
    # Waiting a little longer each time, and never so long that a save that
    # is going to work takes an age to do so.
    assert waits == [0.5, 1.0]
    assert list(tmp_path.iterdir()) == [path]


def test_a_file_that_stays_held_keeps_the_data_beside_it(tmp_path: Path, monkeypatch) -> None:
    """The old file is left as it was, and the new data is not thrown away."""
    path = tmp_path / "transcript.json"
    path.write_text('{"a": 0}', encoding="utf-8")
    calls = _replace_that_refuses(REPLACE_ATTEMPTS, monkeypatch)
    waits: list[float] = []

    assert write_json_object(path, {"a": 1}, sleep=waits.append) is False

    assert len(calls) == REPLACE_ATTEMPTS + 1  # the last one renames the copy
    assert read_json_object(path) == {"a": 0}
    kept = unsaved_copy_path(path)
    assert kept == tmp_path / "transcript.json.unsaved"
    assert json.loads(kept.read_text(encoding="utf-8")) == {"a": 1}
    assert sorted(item.name for item in tmp_path.iterdir()) == [
        "transcript.json",
        "transcript.json.unsaved",
    ]
    assert len(waits) == REPLACE_ATTEMPTS - 1
    assert max(waits) <= 2.0


def test_an_error_that_will_not_go_away_is_not_retried(tmp_path: Path, monkeypatch) -> None:
    calls: list[int] = []

    def replace(source, destination):
        calls.append(1)
        raise FileNotFoundError("The folder has gone.")

    monkeypatch.setattr(json_store.os, "replace", replace)
    waits: list[float] = []

    assert write_json_object(tmp_path / "x.json", {"a": 1}, sleep=waits.append) is False
    assert len(calls) == 1
    assert waits == []


def test_nothing_is_left_behind_when_the_move_fails_for_good(tmp_path: Path, monkeypatch) -> None:
    def replace(source, destination):
        raise FileNotFoundError("The folder has gone.")

    monkeypatch.setattr(json_store.os, "replace", replace)
    write_json_object(tmp_path / "x.json", {"a": 1}, sleep=lambda _seconds: None)

    assert list(tmp_path.iterdir()) == []
