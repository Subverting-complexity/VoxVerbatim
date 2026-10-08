"""Making the smooth transcript again on request, and opening it.

The pipeline makes ``transcript-smooth.txt`` at the end of a run. After
that, the person corrects words in the review window, and the smooth copy
falls behind. The two commands here let them make it again from the
transcript as it now stands, and open it in their text editor.

Both the main window and the review window offer the commands, so the work
lives here rather than in either window. Smoothing sends the transcript to
the language model and can take a minute or more, so it runs on a
background thread and reports back through a Qt signal. The window stays
usable, and says out loud when the smooth copy is ready or why it is not.

The transcript itself is never changed here. The run records what it sent
in the transcript when it smooths at the end of a transcription; a
request made on demand is not recorded, because writing ``transcript.json``
from a background thread could overwrite a correction the review window
saved in the meantime.
"""

from __future__ import annotations

import copy
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtGui import QDesktopServices

from vox_verbatim.transcription.exports import SMOOTH_EXPORT_NAME
from vox_verbatim.transcription.model import Transcript
from vox_verbatim.transcription.smoothing import SmoothOutcome, Smoother, write_smooth_transcript
from vox_verbatim.transcription.store import TranscriptStore

_log = logging.getLogger(__name__)

#: The keys of the two commands, the same in both windows. Neither window
#: uses either key for anything else.
MAKE_AGAIN_KEY = "Ctrl+Shift+M"
OPEN_KEY = "Ctrl+Shift+O"


@dataclass(frozen=True)
class SmoothResult:
    """What one request to make the smooth transcript came to."""

    recording_name: str
    succeeded: bool
    message: str


def smooth_path(store: TranscriptStore) -> Path:
    """Where a recording's smooth transcript is, whether or not it exists."""
    return store.exports_folder / SMOOTH_EXPORT_NAME


def starting_message(recording_name: str) -> str:
    return (
        f"Making the smooth transcript of {recording_name} again with the language "
        "model. You can keep working. You will hear when it is done."
    )


def busy_message() -> str:
    return (
        "A smooth transcript is already being made. Wait for it to finish before "
        "asking for another."
    )


def make_smooth_transcript(
    recording_name: str,
    transcript: Transcript,
    store: TranscriptStore,
    smoother: Smoother,
) -> SmoothResult:
    """Make the smooth transcript and say what happened. Never raises.

    On a failure the older smooth file is removed, as the pipeline does: it
    was made from an earlier transcript, and a smooth copy that does not
    match the corrections beside it would mislead whoever reads it.
    """
    try:
        outcome = write_smooth_transcript(transcript, store, smoother)
    except Exception as error:  # noqa: BLE001 - reported, never raised on a thread
        _log.exception("Making the smooth transcript again failed.")
        outcome = SmoothOutcome(error=f"The smooth transcript was not made. {error}")
    if outcome.error:
        message = f"{recording_name}: {outcome.error}"
        if not message.endswith("."):
            message += "."
        if store.remove_export(SMOOTH_EXPORT_NAME):
            message += " Any older smooth transcript was removed, because it did not match."
        else:
            message += (
                f" An older {SMOOTH_EXPORT_NAME} could not be removed, and it does not "
                "match this transcript. Close any program that has it open, and delete it."
            )
        return SmoothResult(recording_name, False, message)
    message = f"The smooth transcript of {recording_name} is ready."
    problems: list[str] = []
    if outcome.unanswered_parts:
        problems.append(
            f"the language model did not answer for {_parts(outcome.unanswered_parts)}"
        )
    if outcome.unchecked_parts:
        problems.append(f"the edit of {_parts(outcome.unchecked_parts)} did not pass its check")
    if problems:
        message += (
            " But "
            + " and ".join(problems)
            + ". A warning line in the file marks each one."
        )
    return SmoothResult(recording_name, True, message)


def open_smooth_transcript(recording_name: str, store: TranscriptStore) -> tuple[bool, str]:
    """Open the smooth transcript in the default text editor.

    Returns whether it opened, and the sentence to say either way.
    """
    path = smooth_path(store)
    if not path.is_file():
        return False, (
            f"{recording_name} has no smooth transcript yet. Use Make Smooth Transcript "
            f"Again, {MAKE_AGAIN_KEY}, to make one."
        )
    if QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
        return True, f"Opened the smooth transcript of {recording_name}."
    return False, f"Windows could not open the smooth transcript, {path}."


class SmoothRunner(QObject):
    """Makes one smooth transcript at a time on a background thread.

    The main window owns the one runner and lends it to the review window,
    so the two windows can never smooth at the same time. Two runs at once
    could race on one file: a run that failed would remove the file the
    other had just written, after the person had heard that it was ready.
    """

    finished = Signal(object, object)
    """The run has stopped, carrying its :class:`SmoothResult` and whoever
    asked for it, so only that window announces it."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: threading.Thread | None = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        recording_name: str,
        transcript: Transcript,
        store: TranscriptStore,
        smoother: Smoother,
        requester: object = None,
    ) -> bool:
        """Start a run. Returns False, and starts nothing, if one is running.

        The transcript is copied first, so a correction the person makes while
        the request is out cannot change the words under the thread reading
        them. ``requester`` travels with the result.
        """
        if self.is_running():
            return False
        words = copy.deepcopy(transcript)
        self._thread = threading.Thread(
            target=self._run,
            args=(recording_name, words, store, smoother, requester),
            name="smooth-transcript",
            daemon=True,
        )
        self._thread.start()
        return True

    def wait(self, timeout_seconds: float | None = None) -> None:
        """Wait for the run to stop. For tests, and for a window closing."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout_seconds)

    def _run(
        self,
        recording_name: str,
        transcript: Transcript,
        store: TranscriptStore,
        smoother: Smoother,
        requester: object,
    ) -> None:
        result = make_smooth_transcript(recording_name, transcript, store, smoother)
        try:
            self.finished.emit(result, requester)
        except RuntimeError:
            # The window was closed while the request was out. The file is
            # written anyway; there is just nobody left to tell.
            _log.debug("A smooth transcript result was dropped because its window has gone.")


def _parts(number: int) -> str:
    return "1 part" if number == 1 else f"{number} parts"
