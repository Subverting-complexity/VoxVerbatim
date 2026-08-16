"""The audio player: transport buttons, a seek bar, and a plain-words status.

The buttons are labelled compactly on screen, but every one of them carries
a full accessible name such as "Back 2 minutes", so a screen reader never
has to read out a row of angle brackets.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from audio_transcriber.audio.player import AudioPlayer
from audio_transcriber.formatting import format_duration, format_position, spoken_duration
from audio_transcriber.ui.accessibility import describe

#: The skip buttons, in the order they appear from left to right. Each entry
#: is the text on the button, the name a screen reader reads, and how far the
#: button moves in seconds.
SKIP_BUTTONS: tuple[tuple[str, str, int], ...] = (
    ("<<<", "Back 5 minutes", -300),
    ("<<", "Back 2 minutes", -120),
    ("<", "Back 15 seconds", -15),
)

FORWARD_BUTTONS: tuple[tuple[str, str, int], ...] = (
    (">", "Forward 15 seconds", 15),
    (">>", "Forward 2 minutes", 120),
    (">>>", "Forward 5 minutes", 300),
)

_SEEK_ARROW_STEP_SECONDS = 5
_SEEK_PAGE_STEP_SECONDS = 30

STATUS_NO_FILE = "No file loaded"
STATUS_STOPPED = "Stopped"
STATUS_PLAYING = "Playing"
STATUS_PAUSED = "Paused"
STATUS_FINISHED = "Finished playing"


class PlayerPanel(QGroupBox):
    """Drives an :class:`AudioPlayer` and shows what it is doing."""

    statusChanged = Signal(str)
    """The playback status changed, in words worth announcing."""

    def __init__(self, player: AudioPlayer, parent: QWidget | None = None) -> None:
        super().__init__("Audio player", parent)
        self._player = player
        self._syncing_slider = False
        self._duration_ms = 0

        self._buttons: list[QPushButton] = []
        controls = QHBoxLayout()
        controls.setSpacing(6)

        for text, name, delta_seconds in SKIP_BUTTONS:
            self._add_skip_button(controls, text, name, delta_seconds)

        # Alt+P belongs to the Playback menu, so Play answers to Alt+L.
        self._play_button = QPushButton("P&lay", self)
        describe(self._play_button, "Play", "Starts playing the selected file.")
        self._play_button.clicked.connect(self._player.play)
        self._add_button(controls, self._play_button)

        self._pause_button = QPushButton("Pa&use", self)
        describe(self._pause_button, "Pause", "Pauses playback at the current position.")
        self._pause_button.clicked.connect(self._player.pause)
        self._add_button(controls, self._pause_button)

        for text, name, delta_seconds in FORWARD_BUTTONS:
            self._add_skip_button(controls, text, name, delta_seconds)

        controls.addStretch(1)

        self._seek_label = QLabel("Playback position", self)
        self._seek_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._seek_slider.setRange(0, 0)
        self._seek_slider.setSingleStep(_SEEK_ARROW_STEP_SECONDS)
        self._seek_slider.setPageStep(_SEEK_PAGE_STEP_SECONDS)
        self._seek_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._seek_label.setBuddy(self._seek_slider)
        self._update_seek_description()
        self._seek_slider.valueChanged.connect(self._on_slider_value_changed)

        # The slider counts in whole seconds, so the number a screen reader
        # reads out for its value is a number of seconds into the recording
        # rather than a meaningless count of steps.
        self._time_label = QLabel(self)
        self._time_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._status_label = QLabel(STATUS_NO_FILE, self)
        describe(self._status_label, "Playback status")

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self._seek_label)
        layout.addWidget(self._seek_slider)
        layout.addWidget(self._time_label)
        layout.addWidget(self._status_label)

        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.playingChanged.connect(self._on_playing_changed)
        self._player.playbackFinished.connect(self._on_playback_finished)

        self.set_media_loaded(False)
        self._update_time_label(0)

    # -- Construction helpers -------------------------------------------

    def _add_skip_button(
        self,
        controls: QHBoxLayout,
        text: str,
        name: str,
        delta_seconds: int,
    ) -> QPushButton:
        button = QPushButton(text, self)
        describe(button, name, f"Moves playback {name.lower()}.")
        button.clicked.connect(lambda _checked=False, d=delta_seconds: self._player.skip(d * 1000))
        return self._add_button(controls, button)

    def _add_button(self, controls: QHBoxLayout, button: QPushButton) -> QPushButton:
        """Add a transport button, allowing it to narrow on a small window.

        Eight buttons at the width Windows gives a button by default would
        stop this panel from ever being narrower than about 700 pixels. The
        smallest width allowed here is worked out from the current font, so
        the buttons still grow when the user runs a large system font.
        """
        metrics = button.fontMetrics()
        text_width = metrics.horizontalAdvance(button.text().replace("&", ""))
        button.setMinimumWidth(text_width + metrics.height() * 2)
        self._buttons.append(button)
        controls.addWidget(button)
        return button

    # -- State ----------------------------------------------------------

    @property
    def status_widget(self) -> QWidget:
        """The widget holding the status text, for accessibility alerts."""
        return self._status_label

    def set_media_loaded(self, loaded: bool) -> None:
        """Enable or disable the controls depending on whether a file is loaded.

        Play and Pause are both left enabled whenever a file is loaded,
        rather than switching one off at a time. A control that switches
        itself off while it has focus takes the focus away with it, which
        leaves a keyboard or screen reader user stranded.
        """
        for button in self._buttons:
            button.setEnabled(loaded)
        self._seek_slider.setEnabled(loaded)
        if not loaded:
            self._duration_ms = 0
            self._set_slider_value(0)
            self._seek_slider.setRange(0, 0)
            self._update_time_label(0)
            self._set_status(STATUS_NO_FILE)
        else:
            self._set_status(STATUS_STOPPED)

    def focus_play_button(self) -> None:
        self._play_button.setFocus()

    # -- Player signals -------------------------------------------------

    def _on_position_changed(self, milliseconds: int) -> None:
        if not self._seek_slider.isSliderDown():
            self._set_slider_value(milliseconds // 1000)
        self._update_time_label(milliseconds)

    def _on_duration_changed(self, milliseconds: int) -> None:
        self._duration_ms = max(0, milliseconds)
        self._seek_slider.setRange(0, self._duration_ms // 1000)
        self._update_seek_description()
        self._update_time_label(self._player.position)

    def _on_playing_changed(self, playing: bool) -> None:
        if playing:
            self._set_status(STATUS_PLAYING)
        elif self._player.has_media:
            self._set_status(STATUS_PAUSED)
        else:
            self._set_status(STATUS_NO_FILE)

    def _on_playback_finished(self) -> None:
        self._set_status(STATUS_FINISHED)

    # -- Seek bar -------------------------------------------------------

    def _on_slider_value_changed(self, value: int) -> None:
        if self._syncing_slider:
            return
        self._player.seek_to(value * 1000)

    def _set_slider_value(self, seconds: int) -> None:
        if self._seek_slider.value() == seconds:
            return
        self._syncing_slider = True
        try:
            self._seek_slider.setValue(seconds)
        finally:
            self._syncing_slider = False

    def _update_seek_description(self) -> None:
        total = spoken_duration(self._duration_ms / 1000.0) if self._duration_ms else None
        description = (
            "Moves to another point in the recording. The value is the number of seconds "
            "from the start. Left and Right arrow move "
            f"{_SEEK_ARROW_STEP_SECONDS} seconds, Page Up and Page Down move "
            f"{_SEEK_PAGE_STEP_SECONDS} seconds, and Home and End move to the start and end."
        )
        if total:
            description = f"{description} The recording is {total} long."
        describe(self._seek_slider, "Playback position in seconds", description)

    def _update_time_label(self, position_ms: int) -> None:
        total = format_duration(self._duration_ms / 1000.0) if self._duration_ms else "--:--"
        self._time_label.setText(f"{format_position(position_ms)} of {total}")

    # -- Status ---------------------------------------------------------

    def _set_status(self, status: str) -> None:
        if self._status_label.text() == status:
            return
        self._status_label.setText(status)
        self.statusChanged.emit(status)
