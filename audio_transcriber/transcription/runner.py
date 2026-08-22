"""Running the transcription pipeline over several recordings in the background.

Transcribing one recording means waiting on four or five network services,
each of which is sent minutes of audio. It takes minutes, sometimes tens of
minutes, and it plainly cannot happen on the thread that draws the window:
doing it there would leave the dialog frozen, the progress bar stuck and a
screen reader with nothing to report, which is exactly the situation a
progress bar exists to avoid.

So the work runs on a background thread and reports back through Qt signals,
in the same shape as the enhancement runner. Cancelling sets a flag, which is
checked between recordings and handed to the pipeline as its own cancel
check, so pressing Cancel stops at the next stage rather than at the end of
the queue.

Recordings are worked through one after another, never at once, and that is a
deliberate limit rather than an oversight. One recording already has several
services running in parallel inside it, and the user's rate limits are per
account rather than per recording, so two recordings at once would not go
twice as fast. They would collide with each other, be throttled, and fail in
a way that costs money for nothing.

Nothing here decides what to do about a failure. A recording that could not be
transcribed becomes a result saying so and the next one starts, because
losing one recording out of five is no reason to abandon the other four.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QCoreApplication, QObject, Signal

from audio_transcriber.transcription import pipeline
from audio_transcriber.transcription.model import Transcript
from audio_transcriber.transcription.store import transcript_folder_for

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecordingOutcome:
    """What became of one recording, whether or not it worked.

    This carries numbers and sentences rather than the transcript itself,
    and the difference is measured in hundreds of megabytes. A transcript
    holds every service's answer for every word and every candidate that
    was weighed, so a three-hour recording parses to something like 90 MB,
    and a run over a day's worth of recordings that kept each one would be
    holding the whole day in memory by the time it finished, for a dialog
    that only ever shows a word count. The transcript is saved to its folder
    the moment it is made, and anything that wants the words reads it from
    there.
    """

    path: Path
    transcribed: bool = False
    """Whether a transcript was made for this recording."""
    word_count: int = 0
    review_count: int = 0
    """How many words this recording is waiting for a person to settle."""
    warnings: tuple[str, ...] = ()
    transcript_folder: Path | None = None
    """Where the transcript and everything that explains it were written."""
    error: str = ""
    """Why there is no transcript, where there is none."""

    @classmethod
    def of_transcript(
        cls,
        path: Path,
        transcript: Transcript,
        transcript_folder: Path | None = None,
    ) -> "RecordingOutcome":
        """What there is to say about a finished transcript, without keeping it."""
        return cls(
            path=path,
            transcribed=True,
            word_count=len(transcript.tokens),
            review_count=len(transcript.review_tokens),
            warnings=tuple(transcript.warnings),
            transcript_folder=transcript_folder,
        )

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def succeeded(self) -> bool:
        return self.transcribed

    @property
    def message(self) -> str:
        """What happened to this recording, in sentences to be read out.

        The warnings are included in full rather than summarised. Every one
        of them is a service that did not answer or a decision that could not
        be made, and each changes what the transcript is worth, so hiding
        them behind a count would leave the user trusting a transcript they
        would have questioned.
        """
        if not self.transcribed:
            return f"{self.name} could not be transcribed. {self.error}".strip()
        words = _count(self.word_count, "word")
        if self.review_count:
            waiting = f"{_count(self.review_count, 'word')} waiting for review"
        else:
            waiting = "nothing waiting for review"
        lines = [f"{self.name}: {words}, with {waiting}."]
        lines.extend(self.warnings)
        return "\n".join(lines)


@dataclass(frozen=True)
class RunSummary:
    """What a whole run came to, once it has stopped."""

    results: list[RecordingOutcome] = field(default_factory=list)
    cancelled: bool = False

    @property
    def transcribed(self) -> int:
        return sum(1 for result in self.results if result.succeeded)

    @property
    def failed(self) -> int:
        return sum(1 for result in self.results if not result.succeeded)

    @property
    def review_count(self) -> int:
        """How many words the whole run left for a person to settle."""
        return sum(result.review_count for result in self.results)

    @property
    def transcripts(self) -> list[RecordingOutcome]:
        """The recordings that produced a transcript, in the order they ran."""
        return [result for result in self.results if result.succeeded]


class TranscriptionRunner(QObject):
    """Transcribes a list of recordings in the background and says how it goes."""

    recordingStarted = Signal(str, int, int)
    """A recording has been started: its name, its number, and how many there are."""

    recordingFinished = Signal(object)
    """One recording is done, carrying its :class:`RecordingOutcome`."""

    progressChanged = Signal(int, str)
    """How far the whole run has come, from 0 to 100, and what it is doing.

    The sentence travels with the number on purpose. A percentage alone tells
    a waiting user almost nothing over a run measured in minutes, whereas
    "Asking a second service about 12 places" tells them what their money is
    being spent on at that moment.
    """

    runFinished = Signal(object)
    """The run has stopped, carrying its :class:`RunSummary`."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        # Whether this runner currently holds a request for the machine to
        # stay awake. A run's request is withdrawn from two places -- the end
        # of the run and an explicit stop -- and both can happen for the same
        # run, so the flag makes sure one run's request is withdrawn once.
        self._holding_awake = False
        # A run still going when the application quits would be emitting
        # signals at an object that is being taken apart, so it is stopped
        # first.
        application = QCoreApplication.instance()
        if application is not None:
            application.aboutToQuit.connect(self.stop)
        # The request to stay awake is per thread, so it has to be withdrawn
        # on the thread that made it. This object lives on the main thread and
        # the summary arrives there through a queued signal, so a slot on the
        # object itself is the right place rather than the end of the worker.
        self.runFinished.connect(self._let_the_system_sleep)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, recordings: list[Path], options: pipeline.PipelineOptions) -> bool:
        """Begin a run. Returns whether one was started.

        A second run cannot be started while the first is going. Both would
        be talking to the same accounts at the same time, and the services
        count requests per account rather than per run.
        """
        if self.is_running:
            return False
        self._cancel = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            args=([Path(recording) for recording in recordings], options, self._cancel),
            name="transcribe",
            daemon=True,
        )
        self._thread.start()
        self._holding_awake = True
        keep_system_awake(True)
        return True

    def cancel(self) -> None:
        """Ask the run to stop at the next stage it checks."""
        self._cancel.set()

    def stop(self, timeout_seconds: float = 5.0) -> None:
        """Cancel the run and wait briefly for the thread to end.

        Waiting matters when the window is closing, because the thread emits
        signals on this object and must finish before it is destroyed. The
        wait is short and may expire: a request already with a service cannot
        be recalled, and holding the window open until it answers would be
        worse than letting a daemon thread finish on its own.
        """
        self._cancel.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout_seconds)
        self._let_the_system_sleep()

    def _let_the_system_sleep(self, _summary: object = None) -> None:
        """Withdraw this run's request to stay awake, once.

        A stopped run reaches here twice: from :meth:`stop` and again when the
        thread's summary arrives. Withdrawing twice would take another run's
        request with it, because the requests are counted across runners.
        """
        if not self._holding_awake:
            return
        self._holding_awake = False
        keep_system_awake(False)

    # -- The background thread -------------------------------------------

    def _run(
        self,
        recordings: list[Path],
        options: pipeline.PipelineOptions,
        cancel: threading.Event,
    ) -> None:
        """Work through the recordings one at a time, reporting as it goes."""
        total = len(recordings)
        results: list[RecordingOutcome] = []
        for number, recording in enumerate(recordings, start=1):
            # Checked here rather than only inside the pipeline, so that
            # cancelling stops the queue as well as the recording in hand.
            if cancel.is_set():
                break
            self._emit(cancel, self.recordingStarted, recording.name, number, total)
            outcome = self._transcribe_one(recording, options, cancel, number, total)
            results.append(outcome)
            self._emit(cancel, self.recordingFinished, outcome)
        # The summary goes out even when the run was cancelled, because the
        # dialog has to switch its controls back on and say what was done
        # before the user stopped it.
        summary = RunSummary(results=results, cancelled=cancel.is_set())
        self._emit(cancel, self.runFinished, summary, force=True)

    def _transcribe_one(
        self,
        recording: Path,
        options: pipeline.PipelineOptions,
        cancel: threading.Event,
        number: int,
        total: int,
    ) -> RecordingOutcome:
        """Transcribe one recording, turning any failure into a result.

        The pipeline promises to return a transcript for almost everything
        that can go wrong, and raises only when the recording itself cannot
        be read. Even so, everything is caught here. This thread has no
        window to show an error in, and an exception escaping it would end
        the run silently with the dialog still saying it was working.
        """
        try:
            transcript = pipeline.transcribe_recording(
                recording,
                options,
                progress=lambda report: self._report(cancel, number, total, report),
                cancelled=cancel.is_set,
            )
        except Exception as error:  # a bad recording must not end the run
            _log.exception("%s could not be transcribed.", recording)
            return RecordingOutcome(path=recording, error=_reason(error))
        # The transcript is let go here, on purpose. It has been saved to its
        # folder by the pipeline, and keeping it would hold the whole run's
        # worth of transcripts in memory for a dialog that shows a count.
        folder = transcript_folder_for(
            recording, options.settings.processing.transcript_folder_suffix
        )
        return RecordingOutcome.of_transcript(recording, transcript, folder)

    def _report(
        self,
        cancel: threading.Event,
        number: int,
        total: int,
        report: pipeline.PipelineProgress,
    ) -> None:
        """Turn one recording's progress into progress over the whole run.

        The recordings are given equal shares, which is a guess and is
        sometimes a poor one, since a five-minute recording and a two-hour
        one are not the same amount of work. It is still far better than a
        bar that runs to the end and starts again for each recording, which
        tells the user nothing about how long there is left.
        """
        share = max(1, total)
        fraction = max(0.0, min(1.0, report.fraction))
        overall = (number - 1 + fraction) / share
        self._emit(
            cancel,
            self.progressChanged,
            int(round(max(0.0, min(1.0, overall)) * 100)),
            report.stage,
        )

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
            _log.debug("A transcription result was dropped because its receiver has gone.")
            return False
        return True


# -- Keeping the machine awake -------------------------------------------

#: The two flags this application asks Windows for. ES_CONTINUOUS makes the
#: request hold until it is withdrawn; ES_SYSTEM_REQUIRED stops the machine
#: going to sleep. The display is deliberately not kept on: a run lasts
#: hours, nobody needs to watch it, and a screen that will not turn off is
#: a nuisance rather than a help.
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _windows_thread_execution_state(flags: int) -> None:
    """Ask Windows to hold the machine awake, or let it go. Windows only."""
    ctypes.windll.kernel32.SetThreadExecutionState(flags)  # type: ignore[attr-defined]


#: How the request reaches the operating system. A module-level name so that
#: a test can stand in for it and see what was asked.
set_thread_execution_state: Callable[[int], None] = _windows_thread_execution_state

#: How many runs currently want the machine kept awake. Windows keeps one
#: state per thread, not a count, so a second run that starts while the
#: first is still stopping would have its request cleared the moment the
#: first withdrew. Counting here means only the last withdrawal lets the
#: machine sleep.
_awake_requests = 0


def keep_system_awake(awake: bool) -> None:
    """Stop Windows sleeping while a run is going, and let it sleep again after.

    Requests are counted: each ``True`` is one request and each ``False``
    withdraws one, and the machine is only let go when no request remains.
    A withdrawal with nothing outstanding is ignored rather than clearing a
    request someone else holds.

    A transcription run is hours of waiting on services with nobody touching
    the keyboard, which is exactly what a laptop's power settings read as a
    machine nobody is using. Left to itself it sleeps, every request in
    flight dies, and the run is found the next morning stopped part way
    through with the early recordings paid for and the later ones untouched.

    Nothing here may raise. A machine that cannot be asked -- not Windows,
    or a Windows where the call fails -- transcribes exactly as before, and
    the worst that happens is the sleep this was meant to prevent.
    """
    global _awake_requests
    if awake:
        _awake_requests += 1
        if _awake_requests > 1:
            return
    else:
        if _awake_requests == 0:
            return
        _awake_requests -= 1
        if _awake_requests > 0:
            return
    if sys.platform != "win32":
        return
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED if awake else ES_CONTINUOUS
    try:
        set_thread_execution_state(flags)
    except Exception:  # staying awake is a convenience, never a reason to stop
        _log.debug("Could not change the system's sleep setting.", exc_info=True)


def summarise(summary: RunSummary) -> str:
    """One or two sentences saying how a finished run went."""
    total = len(summary.results)
    parts = [f"{summary.transcribed} of {total} recordings transcribed"]
    if summary.failed:
        parts.append(f"{summary.failed} could not be transcribed")
    counts = ", ".join(parts) + "."
    review = summary.review_count
    waiting = (
        f"{_count(review, 'word')} to review." if review else "Nothing is waiting for review."
    )
    if summary.cancelled:
        return f"The run was stopped. {counts} {waiting}"
    if total and summary.failed == 0:
        return f"Finished. {counts} {waiting}"
    return (
        f"Finished, with something to report. {counts} {waiting} See the list below for "
        "each recording."
    )


def _count(number: int, noun: str) -> str:
    """Say "1 word" or "12 words", so counts read as English rather than as data."""
    return f"{number:,} {noun}" if number == 1 else f"{number:,} {noun}s"


def _reason(error: Exception) -> str:
    """Say why something failed in words, falling back to the exception's type.

    Some exceptions carry a perfectly good sentence and some carry nothing at
    all. An empty message on screen would leave the user knowing only that
    something went wrong, so the class name is used rather than nothing.
    """
    message = str(error).strip()
    return message or f"The run stopped with a {type(error).__name__}."
