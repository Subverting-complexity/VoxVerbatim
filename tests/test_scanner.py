"""Tests for the background folder scanner."""

from __future__ import annotations

import threading
import time

from vox_verbatim.audio import scanner as scanner_module
from vox_verbatim.audio.scanner import FolderScanner

from tests.conftest import wait_until, write_fake_audio


def test_a_scan_reports_the_files_and_then_their_durations(qapp, tmp_path):
    write_fake_audio(tmp_path / "one.m4a")
    write_fake_audio(tmp_path / "two.m4a")
    write_fake_audio(tmp_path / "notes.txt")

    scanner = FolderScanner()
    found: list[str] = []
    durations: list[tuple[str, float | None]] = []
    finished: list[int] = []
    scanner.filesFound.connect(lambda files, _id: found.extend(f.name for f in files))
    scanner.durationsRead.connect(lambda batch, _id: durations.extend(batch))
    scanner.scanFinished.connect(finished.append)

    scan_id = scanner.start(tmp_path)

    assert wait_until(qapp, lambda: finished)
    assert found == ["one.m4a", "two.m4a"]
    assert sorted(name for name, _ in durations) == ["one.m4a", "two.m4a"]
    assert finished == [scan_id]
    scanner.stop()


def test_a_folder_that_is_not_there_is_reported_in_words(qapp, tmp_path):
    scanner = FolderScanner()
    failures: list[str] = []
    scanner.scanFailed.connect(lambda message, _id: failures.append(message))

    scanner.start(tmp_path / "no-such-folder")

    assert wait_until(qapp, lambda: failures)
    assert "no longer exists" in failures[0]
    scanner.stop()


def test_starting_a_new_scan_does_not_wait_for_a_slow_one(qapp, tmp_path, monkeypatch):
    """Switching folders must not block the window while a slow file is read.

    A scan can only notice that it was cancelled between files, so a file
    that takes a long time to read would hold up anyone waiting for the
    thread. Starting the next scan therefore cancels without waiting.
    """
    write_fake_audio(tmp_path / "slow.m4a")
    reading = threading.Event()
    release = threading.Event()

    def slow_read(_path):
        reading.set()
        release.wait(10.0)
        return None

    monkeypatch.setattr(scanner_module, "read_duration", slow_read)
    scanner = FolderScanner()
    scanner.start(tmp_path)
    assert reading.wait(5.0), "the scan never reached the slow file"

    started_at = time.monotonic()
    scanner.start(tmp_path)
    elapsed = time.monotonic() - started_at

    assert elapsed < 0.5, f"starting a scan waited {elapsed:.2f} seconds for the previous one"
    release.set()
    scanner.stop()


def test_each_scan_gets_its_own_number(qapp, tmp_path):
    scanner = FolderScanner()

    first = scanner.start(tmp_path)
    second = scanner.start(tmp_path)

    assert second == first + 1
    assert scanner.current_scan_id == second
    scanner.stop()
