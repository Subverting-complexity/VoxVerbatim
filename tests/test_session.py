"""Tests for saving and restoring the working session."""

from __future__ import annotations

import json

from audio_transcriber.session import SessionState, SessionStore


def test_a_saved_session_comes_back_unchanged(tmp_path):
    store = SessionStore(tmp_path / "session.json")
    state = SessionState(
        folder=r"C:\Recordings",
        checked_files=["one.m4a", "two.m4a"],
        selected_file="two.m4a",
        window_geometry="Z2VvbWV0cnk=",
        splitter_state="c3BsaXR0ZXI=",
    )

    assert store.save(state)
    loaded = store.load()

    assert loaded == state


def test_a_missing_file_gives_an_empty_session(tmp_path):
    store = SessionStore(tmp_path / "not-there.json")

    loaded = store.load()

    assert loaded == SessionState()


def test_a_damaged_file_gives_an_empty_session(tmp_path):
    path = tmp_path / "session.json"
    path.write_text("{ this is not json", encoding="utf-8")

    assert SessionStore(path).load() == SessionState()


def test_a_file_that_is_not_utf_8_gives_an_empty_session(tmp_path):
    """A session file with stray bytes must not stop the application starting."""
    path = tmp_path / "session.json"
    path.write_bytes(b'{"folder": "\xff\xfe not text"}')

    assert SessionStore(path).load() == SessionState()


def test_a_file_holding_the_wrong_types_falls_back_to_defaults(tmp_path):
    path = tmp_path / "session.json"
    path.write_text(
        json.dumps(
            {
                "folder": 42,
                "checked_files": ["good.m4a", 7, None],
                "selected_file": "",
                "window_geometry": ["not", "a", "string"],
            }
        ),
        encoding="utf-8",
    )

    loaded = SessionStore(path).load()

    assert loaded.folder is None
    assert loaded.checked_files == ["good.m4a"]
    assert loaded.selected_file is None
    assert loaded.window_geometry is None


def test_saving_creates_the_folder_it_needs(tmp_path):
    store = SessionStore(tmp_path / "nested" / "deeper" / "session.json")

    assert store.save(SessionState(folder="X:\\Audio"))
    assert store.load().folder == "X:\\Audio"


def test_saving_leaves_no_temporary_files_behind(tmp_path):
    store = SessionStore(tmp_path / "session.json")

    store.save(SessionState(folder="X:\\Audio"))
    store.save(SessionState(folder="Y:\\Audio"))

    assert sorted(item.name for item in tmp_path.iterdir()) == ["session.json"]
