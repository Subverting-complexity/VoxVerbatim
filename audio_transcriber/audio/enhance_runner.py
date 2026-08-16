"""Running an enhancement over several files without freezing the window.

Enhancing a recording takes seconds to minutes depending on its length, so
it cannot happen on the thread that draws the window. Doing it there would
leave the dialog frozen, the progress bar stuck and a screen reader with
nothing to report, which is exactly the situation a progress bar exists to
avoid.

The work therefore runs on a background thread and reports back through Qt
signals, in the same shape as the folder scanner. Cancelling sets a flag
that the work checks between chunks of audio, so pressing Cancel stops
within a fraction of a second rather than at the end of the file.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QObject, Signal

from audio_transcriber.audio.enhance import (
    EnhanceOptions,
    FileResult,
    Outcome,
    enhance_files,
)

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunSummary:
    """What a whole run came to, once it has stopped."""

    results: list[FileResult] = field(default_factory=list)
    cancelled: bool = False

    def count(self, outcome: Outcome) -> int:
        return sum(1 for result in self.results if result.outcome == outcome)

    @property
    def enhanced(self) -> int:
        return self.count(Outcome.ENHANCED)

    @property
    def skipped(self) -> int:
        return self.count(Outcome.SKIPPED)

    @property
    def failed(self) -> int:
        return self.count(Outcome.FAILED)


class EnhanceRunner(QObject):
    """Enhances a list of files in the background and reports what it is doing."""

    fileStarted = Signal(str, int, int)
    """A file has been started: its name, its number, and how many there are."""

    fileFinished = Signal(object)
    """One file is done, carrying its :class:`FileResult`."""

    progressChanged = Signal(int)
    """How far the whole run has come, from 0 to 100."""

    runFinished = Signal(object)
    """The run has stopped, carrying its :class:`RunSummary`."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        # A run still going when the application quits would be emitting
        # signals at an object that is being taken apart, so it is stopped
        # first.
        application = QCoreApplication.instance()
        if application is not None:
            application.aboutToQuit.connect(self.stop)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, files: list[Path], options: EnhanceOptions) -> bool:
        """Begin a run. Returns whether one was started.

        A second run cannot be started while the first is going, because
        the two would write into the same folder at the same time.
        """
        if self.is_running:
            return False
        self._cancel = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            args=(list(files), options, self._cancel),
            name="enhance-audio",
            daemon=True,
        )
        self._thread.start()
        return True

    def cancel(self) -> None:
        """Ask the run to stop at the next chunk of audio."""
        self._cancel.set()

    def stop(self, timeout_seconds: float = 5.0) -> None:
        """Cancel the run and wait briefly for the thread to end.

        Waiting matters when the window is closing, because the thread
        emits signals on this object and must finish before it is
        destroyed.
        """
        self._cancel.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout_seconds)

    # -- The background thread -------------------------------------------

    def _run(self, files: list[Path], options: EnhanceOptions, cancel: threading.Event) -> None:
        """Enhance the files, turning what the engine reports into signals.

        Working through the list is the engine's job, not this class's. All
        that happens here is that each thing the engine says on the way is
        passed on to the main thread.
        """
        results = enhance_files(
            files,
            options,
            progress=lambda fraction: self._report(cancel, fraction),
            cancelled=cancel.is_set,
            on_start=lambda source, number, total: self._emit(
                cancel, self.fileStarted, source.name, number, total
            ),
            on_result=lambda result: self._emit(cancel, self.fileFinished, result),
        )
        # The summary goes out even when the run was cancelled, because the
        # dialog has to switch its controls back on and say what was done
        # before the user stopped it.
        summary = RunSummary(results=results, cancelled=cancel.is_set())
        self._emit(cancel, self.runFinished, summary, force=True)

    def _report(self, cancel: threading.Event, fraction: float) -> None:
        self._emit(cancel, self.progressChanged, int(round(max(0.0, min(1.0, fraction)) * 100)))

    @staticmethod
    def _emit(cancel: threading.Event, signal, *args, force: bool = False) -> bool:
        """Send a result to the main thread, unless it is too late to bother.

        The run has its own thread, so the dialog it reports to can be
        destroyed while a result is on its way. Qt raises in that case, and
        the right answer is simply to stop.
        """
        if cancel.is_set() and not force:
            return False
        try:
            signal.emit(*args)
        except RuntimeError:
            _log.debug("An enhancement result was dropped because its receiver has gone.")
            return False
        return True
