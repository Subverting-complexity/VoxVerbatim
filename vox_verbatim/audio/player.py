"""Audio playback, wrapped so the user interface never touches Qt Multimedia directly.

Everything here works in milliseconds, which is what Qt Multimedia uses.
The panel that drives it converts to and from seconds for display.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

_log = logging.getLogger(__name__)


class AudioPlayer(QObject):
    """Plays one audio file at a time and reports what it is doing."""

    positionChanged = Signal(int)
    """The playback position in milliseconds."""

    durationChanged = Signal(int)
    """The length of the loaded file in milliseconds, or zero while unknown."""

    playingChanged = Signal(bool)
    """Whether audio is currently coming out of the speakers."""

    playbackFinished = Signal()
    """The end of the file was reached."""

    errorOccurred = Signal(str)
    """Something went wrong, described in words a person can act on."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._audio_output = QAudioOutput(self)
        self._player = QMediaPlayer(self)
        self._player.setAudioOutput(self._audio_output)
        self._source_path: Path | None = None

        # Qt reports positions and durations as 64 bit values. They are passed
        # on through small methods rather than connected signal to signal,
        # because the two signal signatures do not match.
        self._player.positionChanged.connect(self._on_player_position_changed)
        self._player.durationChanged.connect(self._on_player_duration_changed)
        self._player.playbackStateChanged.connect(self._on_playback_state_changed)
        self._player.mediaStatusChanged.connect(self._on_media_status_changed)
        self._player.errorOccurred.connect(self._on_error)

    @property
    def source_path(self) -> Path | None:
        """The file that is loaded, or ``None`` if nothing is loaded."""
        return self._source_path

    @property
    def position(self) -> int:
        return self._player.position()

    @property
    def duration(self) -> int:
        return self._player.duration()

    @property
    def is_playing(self) -> bool:
        return self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

    @property
    def has_media(self) -> bool:
        return self._source_path is not None

    def load(self, path: Path | str | None) -> None:
        """Load a file ready for playback, or clear the player when given ``None``.

        Loading always stops whatever was playing. Selecting a different
        file in the list is not a request to start playing it.
        """
        self._player.stop()
        if path is None:
            self._source_path = None
            self._player.setSource(QUrl())
            self.durationChanged.emit(0)
            self.positionChanged.emit(0)
            return
        self._source_path = Path(path)
        self._player.setSource(QUrl.fromLocalFile(str(self._source_path)))

    def play(self) -> None:
        if self._source_path is None:
            return
        self._player.play()

    def pause(self) -> None:
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()

    def stop(self) -> None:
        self._player.stop()

    def toggle_play_pause(self) -> None:
        """Start playing, or pause if already playing."""
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def seek_to(self, milliseconds: int) -> None:
        """Move to an absolute position, clamped to the length of the file."""
        if self._source_path is None:
            return
        self._player.setPosition(self._clamp(milliseconds))

    def skip(self, delta_milliseconds: int) -> None:
        """Move forwards or backwards from the current position.

        The result is clamped, so pressing "Back 5 minutes" near the start
        goes to the beginning rather than doing nothing, which is what a
        listener expects.
        """
        if self._source_path is None:
            return
        self._player.setPosition(self._clamp(self._player.position() + delta_milliseconds))

    def _clamp(self, milliseconds: int) -> int:
        value = max(0, int(milliseconds))
        duration = self._player.duration()
        if duration > 0:
            value = min(value, duration)
        return value

    def _on_player_position_changed(self, milliseconds: int) -> None:
        self.positionChanged.emit(int(milliseconds))

    def _on_player_duration_changed(self, milliseconds: int) -> None:
        self.durationChanged.emit(int(milliseconds))

    def _on_playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        self.playingChanged.emit(state == QMediaPlayer.PlaybackState.PlayingState)

    def _on_media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.playbackFinished.emit()

    def _on_error(self, error: QMediaPlayer.Error, error_string: str) -> None:
        if error == QMediaPlayer.Error.NoError:
            return
        name = self._source_path.name if self._source_path else "the audio file"
        detail = error_string.strip()
        if error == QMediaPlayer.Error.FormatError:
            message = f"{name} is in a format this computer cannot play."
        elif error == QMediaPlayer.Error.ResourceError:
            message = f"{name} could not be opened. It may have been moved or deleted."
        elif error == QMediaPlayer.Error.AccessDeniedError:
            message = f"You do not have permission to play {name}."
        else:
            message = f"{name} could not be played."
        if detail:
            message = f"{message} {detail}"
        _log.warning("Playback error for %s: %s", self._source_path, error_string)
        self.errorOccurred.emit(message)
