"""The readable files follow every correction saved in the review window."""

from __future__ import annotations

from pathlib import Path

from tests.test_review_window import (
    RECORDING,
    FakePlayer,
    Folder,
    make_token,
    make_transcript,
    select_word,
    two_file_folder,
)
from vox_verbatim.transcription.exports import (
    REPORT_EXPORT_NAME,
    TEXT_EXPORT_NAME,
    write_exports,
)
from vox_verbatim.transcription.model import Confidence, ReviewReason
from vox_verbatim.transcription.project import ProjectStore
from vox_verbatim.transcription.store import TranscriptStore
from vox_verbatim.ui import review_window as review_window_module
from vox_verbatim.ui.review_window import ReviewWindow


class RefusingStore:
    """Stands in for an exports folder whose files another program holds open."""

    def __init__(self, refuse: set[str]) -> None:
        self.refuse = refuse
        self.written: dict[str, str] = {}

    def write_export(self, name: str, text: str):
        if name in self.refuse:
            return None
        self.written[name] = text
        return Path(name)


def _open(tmp_path, folder: Folder, write) -> ReviewWindow:
    window = ReviewWindow(
        tmp_path,
        folder.names,
        folder.load,
        {name: Path(f"C:/Audio/{name}") for name in folder.names},
        ProjectStore(tmp_path),
        FakePlayer(),
        folder.save,
        write_exports=write,
    )
    window.show()
    window.process_low_confidence_words()
    return window


def _real_files(tmp_path):
    """Write each recording's files into a real transcript folder under tmp_path."""
    audio = tmp_path / "audio"

    def write(name, transcript):
        return write_exports(transcript, TranscriptStore(audio / name))

    def read(name, export):
        return (TranscriptStore(audio / name).exports_folder / export).read_text("utf-8")

    return write, read


def test_write_exports_names_only_the_files_that_failed():
    store = RefusingStore({REPORT_EXPORT_NAME})
    failed = write_exports(make_transcript(), store)
    assert failed == [REPORT_EXPORT_NAME]
    assert TEXT_EXPORT_NAME in store.written


def test_a_replaced_word_reaches_transcript_txt_at_once(qapp, tmp_path):
    write, read = _real_files(tmp_path)
    window = _open(tmp_path, two_file_folder(), write)
    try:
        select_word(window, "Bosch")
        window._replacement_edit.setText("Boschoff")

        assert window.apply_replacement_to_word() is True

        assert "Boschoff" in read(RECORDING, TEXT_EXPORT_NAME)
    finally:
        window.close()


def test_a_settled_place_leaves_the_review_report(qapp, tmp_path):
    token = make_token("contract", confidence=Confidence.UNRESOLVED)
    folder = Folder({RECORDING: make_transcript([token])})
    write, read = _real_files(tmp_path)
    write(RECORDING, folder.transcripts[RECORDING])
    assert "contract" in read(RECORDING, REPORT_EXPORT_NAME)
    window = _open(tmp_path, folder, write)
    try:
        assert window.confirm_item() is True

        assert "contract" not in read(RECORDING, REPORT_EXPORT_NAME)
    finally:
        window.close()


def test_a_timing_decision_writes_the_files_again(qapp, tmp_path):
    token = make_token(
        "contract",
        reasons=(ReviewReason.WEAK_ALIGNMENT,),
        confidence=Confidence.REVIEW_SUGGESTED,
    )
    token.timing_confidence = Confidence.REVIEW_REQUIRED
    folder = Folder({RECORDING: make_transcript([token])})
    calls: list[str] = []

    def write(name, transcript):
        calls.append(name)
        return []

    window = _open(tmp_path, folder, write)
    try:
        assert window.decide_timing(True) is True
        assert calls == [RECORDING]
    finally:
        window.close()


def test_a_locked_file_is_announced_and_the_correction_is_kept(qapp, tmp_path, monkeypatch):
    said: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        review_window_module,
        "announce",
        lambda widget, message, urgent=False: said.append((message, urgent)),
    )
    token = make_token("contract", confidence=Confidence.UNRESOLVED)
    folder = Folder({RECORDING: make_transcript([token])})
    window = _open(tmp_path, folder, lambda name, transcript: [TEXT_EXPORT_NAME])
    try:
        said.clear()
        assert window.confirm_item() is True

        assert folder.token(RECORDING, token.id).needs_review is False
        message, urgent = said[-1]
        assert f"{TEXT_EXPORT_NAME} for {RECORDING} was not written again" in message
        assert urgent is True
        assert TEXT_EXPORT_NAME in window._status_label.text()
    finally:
        window.close()
