"""Scanning a folder for audio files without freezing the user interface.

Listing a folder is quick, but reading the duration of every file means
opening each one in turn, which on a large folder or a network drive can
take seconds. Doing that on the main thread would lock up the window and
leave a screen reader with nothing to report.

The scanner therefore does its work on a background thread and reports back
through Qt signals in two stages. First the file names and sizes arrive, so
the list can be shown immediately. Then the durations arrive in small
batches and fill themselves in.

Each scan carries a number. When the user picks a different folder the old
scan is cancelled, and any results that were already on their way are
ignored because their number no longer matches.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QObject, Signal

from audio_transcriber.audio.library import list_audio_files, read_duration

_log = logging.getLogger(__name__)

#: Durations are sent back in batches so that a folder with hundreds of files
#: does not flood the main thread with one signal per file.
_BATCH_SIZE = 20
_BATCH_INTERVAL_SECONDS = 0.2


class FolderScanner(QObject):
    """Scans folders in the background and reports results through signals."""

    filesFound = Signal(object, int)
    """The files in the folder, as ``list[AudioFile]``, and the scan number."""

    durationsRead = Signal(object, int)
    """A batch of ``(file name, duration in seconds or None)`` and the scan number."""

    scanFailed = Signal(str, int)
    """A message explaining why the folder could not be read, and the scan number."""

    scanFinished = Signal(int)
    """Emitted once a scan has run to completion, with the scan number."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._scan_id = 0
        self._cancel = threading.Event()
        self._threads: list[threading.Thread] = []
        # A scan still running when the application quits would be emitting
        # signals at an object that is being taken apart, so it is stopped
        # first.
        application = QCoreApplication.instance()
        if application is not None:
            application.aboutToQuit.connect(self.stop)

    @property
    def current_scan_id(self) -> int:
        """The number of the most recently started scan."""
        return self._scan_id

    def start(self, folder: Path | str) -> int:
        """Cancel any scan in progress and start a new one. Returns its number.

        The old scan is told to stop but is not waited for. It can only
        notice between files, and reading one file on a slow network drive
        takes as long as it takes; waiting here would freeze the window at
        exactly the moment the user asked for something new. The cancelled
        scan ends quietly on its own, and its results are discarded.
        """
        self._cancel.set()
        self._threads = [thread for thread in self._threads if thread.is_alive()]
        self._scan_id += 1
        scan_id = self._scan_id
        cancel = threading.Event()
        self._cancel = cancel
        thread = threading.Thread(
            target=self._run,
            args=(Path(folder), scan_id, cancel),
            name=f"folder-scan-{scan_id}",
            daemon=True,
        )
        self._threads.append(thread)
        thread.start()
        return scan_id

    def stop(self, timeout_seconds: float = 2.0) -> None:
        """Cancel every running scan and wait briefly for the threads to end.

        Waiting matters when the window is closing. The threads emit signals
        on this object, so letting them end first avoids emitting into an
        object that is being destroyed.
        """
        self._cancel.set()
        deadline = time.monotonic() + timeout_seconds
        for thread in self._threads:
            if thread.is_alive():
                thread.join(max(0.0, deadline - time.monotonic()))
        self._threads = [thread for thread in self._threads if thread.is_alive()]

    def _run(self, folder: Path, scan_id: int, cancel: threading.Event) -> None:
        try:
            files = list_audio_files(folder)
        except OSError as error:
            self._emit(cancel, self.scanFailed, self._describe_error(folder, error), scan_id)
            return

        if not self._emit(cancel, self.filesFound, files, scan_id):
            return

        batch: list[tuple[str, float | None]] = []
        last_sent = time.monotonic()
        for audio_file in files:
            if cancel.is_set():
                return
            batch.append((audio_file.name, read_duration(audio_file.path)))
            now = time.monotonic()
            if len(batch) >= _BATCH_SIZE or (now - last_sent) >= _BATCH_INTERVAL_SECONDS:
                if not self._emit(cancel, self.durationsRead, batch, scan_id):
                    return
                batch = []
                last_sent = now

        if batch and not self._emit(cancel, self.durationsRead, batch, scan_id):
            return
        self._emit(cancel, self.scanFinished, scan_id)

    @staticmethod
    def _emit(cancel: threading.Event, signal, *args) -> bool:
        """Send a result back to the main thread, unless it is too late to bother.

        The scan runs on its own thread, so the window it reports to can be
        closed and destroyed while a result is on its way. Qt raises in that
        case, and the right answer is simply to stop scanning.
        """
        if cancel.is_set():
            return False
        try:
            signal.emit(*args)
        except RuntimeError:
            _log.debug("The scan result was dropped because its receiver has gone.")
            return False
        return True

    @staticmethod
    def _describe_error(folder: Path, error: OSError) -> str:
        """Turn an operating system error into something worth reading aloud."""
        if isinstance(error, FileNotFoundError):
            return f"The folder {folder} no longer exists."
        if isinstance(error, PermissionError):
            return f"You do not have permission to read the folder {folder}."
        if isinstance(error, NotADirectoryError):
            return f"{folder} is not a folder."
        reason = error.strerror or str(error)
        return f"The folder {folder} could not be read. {reason}."
