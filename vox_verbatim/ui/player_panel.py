"""The audio player: transport buttons, a seek bar, and a plain-words status.

The buttons are labelled compactly on screen, and every one of them also
carries a full accessible name such as "Back 2 minutes". Both are worked
out from the skip intervals in the settings, so changing an interval
changes what the button says and what a screen reader reads.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from vox_verbatim.audio.player import AudioPlayer
from vox_verbatim.formatting import (
    compact_interval,
    format_duration,
    format_position,
    spoken_duration,
    spoken_position,
)
from vox_verbatim.settings import Settings
from vox_verbatim.ui.accessibility import announce, describe
from vox_verbatim.ui.flow_layout import FlowWidget


def skip_button_label(seconds: int) -> str:
    """The text on a skip button, such as ``- 2 min`` or ``+ 15 sec``.

    The buttons say how far they move in words. Arrow brackets would leave
    the difference between two minutes and five minutes to a count of how
    many brackets there are, and the difference between back and forward to
    which way they point, which is exactly the sort of thing someone using a
    magnifier should not have to work out.
    """
    sign = "-" if seconds < 0 else "+"
    return f"{sign} {compact_interval(abs(seconds))}"


def skip_button_name(seconds: int) -> str:
    """The name a screen reader reads, such as ``Back 2 minutes``."""
    direction = "Back" if seconds < 0 else "Forward"
    return f"{direction} {spoken_duration(abs(seconds))}"


_SEEK_ARROW_STEP_SECONDS = 5
_SEEK_PAGE_STEP_SECONDS = 30

#: A seek made from the keyboard is read out once the user stops moving,
#: rather than on every key press.
_SEEK_ANNOUNCE_DELAY_MS = 600

STATUS_NO_FILE = "No file loaded"
STATUS_STOPPED = "Stopped"
STATUS_PLAYING = "Playing"
STATUS_PAUSED = "Paused"
STATUS_FINISHED = "Finished playing"


class PlayerPanel(QGroupBox):
    """Drives an :class:`AudioPlayer` and shows what it is doing."""

    statusChanged = Signal(str)
    """The playback status changed, in words worth announcing."""

    # The transport controls ask for what they want rather than driving the
    # player themselves. The window loading the selected file is slightly
    # behind the highlight in the list, and it has to catch up before an
    # action is carried out, or the action would land on the previous file.
    playRequested = Signal()
    pauseRequested = Signal()
    skipRequested = Signal(int)
    """Move by this many milliseconds, forwards if positive."""

    seekRequested = Signal(int)
    """Move to this position, in milliseconds from the start."""

    focusReleased = Signal()
    """The controls are switching off while one of them holds the focus."""

    def __init__(
        self,
        player: AudioPlayer,
        settings: Settings | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("Audio player", parent)
        self._player = player
        self._syncing_slider = False
        self._duration_ms = 0
        self._settings = settings or Settings()

        self._buttons: list[QPushButton] = []
        self._skip_seconds_by_button: dict[QPushButton, int] = {}
        # The buttons wrap onto a second line when the window is narrow or
        # the system font is large, so the panel never forces the window
        # wider than the screen.
        controls = FlowWidget(self)

        # Longest skip first on the way back, so the row reads as one scale
        # running from the largest jump backwards to the largest forwards.
        self._back_buttons = [
            self._add_skip_button(controls, -seconds)
            for seconds in reversed(self._settings.skip_seconds)
        ]

        # Alt+P belongs to the Playback menu, so Play answers to Alt+L.
        self._play_button = QPushButton("P&lay", self)
        describe(self._play_button, "Play", "Starts playing the selected file.")
        self._play_button.clicked.connect(lambda: self.playRequested.emit())
        self._add_button(controls, self._play_button)

        self._pause_button = QPushButton("Pa&use", self)
        describe(self._pause_button, "Pause", "Pauses playback at the current position.")
        self._pause_button.clicked.connect(lambda: self.pauseRequested.emit())
        self._add_button(controls, self._pause_button)

        self._forward_buttons = [
            self._add_skip_button(controls, seconds)
            for seconds in self._settings.skip_seconds
        ]

        self._seek_label = QLabel("Playback position", self)
        self._seek_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._seek_slider.setRange(0, 0)
        self._seek_slider.setSingleStep(_SEEK_ARROW_STEP_SECONDS)
        self._seek_slider.setPageStep(_SEEK_PAGE_STEP_SECONDS)
        self._seek_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._seek_label.setBuddy(self._seek_slider)
        self._update_seek_description()
        self._seek_slider.valueChanged.connect(self._on_slider_value_changed)

        # A seek the user makes from the keyboard is read out in words once
        # they stop moving. The slider's own value is a count of seconds,
        # which is exact but hard to picture on a long recording.
        self._seek_announce_timer = QTimer(self)
        self._seek_announce_timer.setSingleShot(True)
        self._seek_announce_timer.setInterval(_SEEK_ANNOUNCE_DELAY_MS)
        self._seek_announce_timer.timeout.connect(self._announce_seek_position)

        self._time_label = QLabel(self)
        self._time_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        # These two labels carry messages, so they are left unnamed. A label
        # has no accessible value: naming it would hide what it says.
        self._status_label = QLabel(STATUS_NO_FILE, self)

        layout = QVBoxLayout(self)
        layout.addWidget(controls)
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

    def _add_skip_button(self, controls: FlowWidget, delta_seconds: int) -> QPushButton:
        button = QPushButton(self)
        # How far the button moves is read from the settings when it is
        # pressed, rather than captured now, so a change in the settings
        # takes effect without rebuilding the button.
        button.clicked.connect(
            lambda _checked=False, b=button: self.skipRequested.emit(
                self._skip_milliseconds_of(b)
            )
        )
        self._skip_seconds_by_button[button] = delta_seconds
        self._label_skip_button(button, delta_seconds)
        return self._add_button(controls, button)

    def _label_skip_button(self, button: QPushButton, delta_seconds: int) -> None:
        name = skip_button_name(delta_seconds)
        button.setText(skip_button_label(delta_seconds))
        describe(button, name, f"Moves playback {name.lower()}.")

    def _skip_milliseconds_of(self, button: QPushButton) -> int:
        return self._skip_seconds_by_button.get(button, 0) * 1000

    def apply_settings(self, settings: Settings) -> None:
        """Take new skip intervals, relabelling the buttons to match."""
        self._settings = settings
        backwards = list(reversed(settings.skip_seconds))
        for button, seconds in zip(self._back_buttons, backwards):
            self._skip_seconds_by_button[button] = -seconds
            self._label_skip_button(button, -seconds)
        for button, seconds in zip(self._forward_buttons, settings.skip_seconds):
            self._skip_seconds_by_button[button] = seconds
            self._label_skip_button(button, seconds)

    def _add_button(self, controls: FlowWidget, button: QPushButton) -> QPushButton:
        """Add a transport button to the row that wraps."""
        self._buttons.append(button)
        controls.add(button)
        return button

    # -- State ----------------------------------------------------------

    def set_media_loaded(self, loaded: bool) -> None:
        """Enable or disable the controls depending on whether a file is loaded.

        Play and Pause are both left enabled whenever a file is loaded,
        rather than switching one off at a time. A control that switches
        itself off while it has focus takes the focus away with it, which
        leaves a keyboard or screen reader user stranded.

        When all of them do have to switch off, and one of them holds the
        focus, Qt would drop the focus somewhere arbitrary. The window is
        told instead, so it can put the focus somewhere sensible and say
        where it went.
        """
        # Whether the focus was ours is worked out before anything is
        # switched off, and reported afterwards, so that the window has the
        # last word on both where the focus goes and what the status says.
        losing_focus = not loaded and self._holds_focus()
        for button in self._buttons:
            button.setEnabled(loaded)
        self._seek_slider.setEnabled(loaded)
        if not loaded:
            self._duration_ms = 0
            self._set_slider_value(0)
            self._set_slider_range(0)
            self._update_seek_description()
            self._update_time_label(0)
            self._set_status(STATUS_NO_FILE)
        else:
            self._set_status(STATUS_STOPPED)
        if losing_focus:
            self.focusReleased.emit()

    def focus_play_button(self) -> None:
        self._play_button.setFocus(Qt.FocusReason.TabFocusReason)

    def _holds_focus(self) -> bool:
        # The window is asked rather than the application, so this is right
        # even when the window is not the active one on the desktop.
        focused = self.window().focusWidget()
        return focused is not None and (focused is self or self.isAncestorOf(focused))

    # -- Player signals -------------------------------------------------

    def _on_position_changed(self, milliseconds: int) -> None:
        if not self._seek_slider.isSliderDown():
            self._set_slider_value(milliseconds // 1000)
        self._update_time_label(milliseconds)

    def _on_duration_changed(self, milliseconds: int) -> None:
        self._duration_ms = max(0, milliseconds)
        self._set_slider_range(self._duration_ms // 1000)
        self._set_slider_value(min(self._player.position // 1000, self._duration_ms // 1000))
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
        self.seekRequested.emit(value * 1000)
        # Only a seek the user made restarts this timer. Updates that come
        # from playback go through _set_slider_value, which marks them as
        # ours, so listening to a recording stays quiet.
        self._seek_announce_timer.start()

    def _announce_seek_position(self) -> None:
        announce(self._seek_slider, spoken_position(self._seek_slider.value() * 1000))

    def _set_slider_value(self, seconds: int) -> None:
        if self._seek_slider.value() == seconds:
            return
        self._syncing_slider = True
        try:
            self._seek_slider.setValue(seconds)
        finally:
            self._syncing_slider = False

    def _set_slider_range(self, maximum_seconds: int) -> None:
        """Change the length of the seek bar without it counting as a seek.

        A shorter recording than the last one drags the slider's value down
        with it, which looks exactly like the user moving the slider. Left
        unguarded, loading a short file after a long one would jump straight
        to the end of it.
        """
        self._syncing_slider = True
        try:
            self._seek_slider.setRange(0, max(0, maximum_seconds))
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
