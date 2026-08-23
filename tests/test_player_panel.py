"""Tests for the player panel, in particular the seek bar."""

from __future__ import annotations

from vox_verbatim.audio.player import AudioPlayer
from vox_verbatim.ui.player_panel import PlayerPanel


def test_the_seek_bar_asks_to_move_when_the_user_moves_it(qapp):
    player = AudioPlayer()
    panel = PlayerPanel(player)
    panel.set_media_loaded(True)
    requested: list[int] = []
    panel.seekRequested.connect(requested.append)

    player.durationChanged.emit(300_000)
    panel._seek_slider.setValue(75)

    assert requested == [75_000]


def test_loading_a_shorter_file_does_not_look_like_a_seek(qapp):
    """A shorter recording drags the slider value down, which is not a seek.

    Without a guard the value being clamped to the new, smaller maximum
    looks exactly like the user dragging the slider, and playback would
    jump to the end of the file the moment it loaded.
    """
    player = AudioPlayer()
    panel = PlayerPanel(player)
    panel.set_media_loaded(True)
    player.durationChanged.emit(300_000)
    panel._seek_slider.setValue(200)

    requested: list[int] = []
    panel.seekRequested.connect(requested.append)
    player.durationChanged.emit(10_000)

    assert requested == []
    assert panel._seek_slider.maximum() == 10


def test_the_transport_buttons_ask_rather_than_act(qapp):
    """The window has to load the selected file before a command is carried out."""
    player = AudioPlayer()
    panel = PlayerPanel(player)
    panel.set_media_loaded(True)
    skips: list[int] = []
    plays: list[bool] = []
    panel.skipRequested.connect(skips.append)
    panel.playRequested.connect(lambda: plays.append(True))

    panel._play_button.click()
    for button in panel._buttons:
        if button.accessibleName() == "Back 2 minutes":
            button.click()

    assert plays == [True]
    assert skips == [-120_000]
