"""The main window, which ties the folder, the file list and the player together."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from PySide6.QtCore import QByteArray, QModelIndex, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QCloseEvent, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from vox_verbatim import APPLICATION_NAME
from vox_verbatim.audio.library import AudioFile, DurationState
from vox_verbatim.audio.player import AudioPlayer
from vox_verbatim.audio.scanner import FolderScanner
from vox_verbatim.paths import log_file_path
from vox_verbatim.session import SessionState, SessionStore
from vox_verbatim.settings import (
    SETTINGS_FILE_NAME,
    EnhanceSettings,
    ExportSettings,
    Settings,
    SettingsStore,
)
from vox_verbatim.transcription import grouping, smoothing
from vox_verbatim.transcription.calibration import (
    CALIBRATION_FILE_NAME,
    CalibrationStore,
)
from vox_verbatim.transcription.exports import (
    REPORT_EXPORT_NAME,
    TEXT_EXPORT_NAME,
    write_exports,
)
from vox_verbatim.transcription.learning import user_terms_index
from vox_verbatim.transcription.model import SPEAKER_ONLY_REASONS, Transcript
from vox_verbatim.transcription.project import (
    FlaggedItem,
    Occurrence,
    ProjectState,
    ProjectStore,
    SpeakerDoubtItem,
)
from vox_verbatim.transcription.runner import RunSummary
from vox_verbatim.transcription.runner import summarise as summarise_transcription
from vox_verbatim.transcription.store import TranscriptStore
from vox_verbatim.transcription.vocabulary import (
    VOCABULARY_FILE_NAME,
    Vocabulary,
    VocabularyStore,
)
from vox_verbatim.ui.accessibility import announce, describe
from vox_verbatim.ui.enhance_dialog import EnhanceAudioDialog, summarise
from vox_verbatim.ui.export_flow import EXPORT_KEY, ExportFlow
from vox_verbatim.ui.file_info_panel import FileInfoPanel
from vox_verbatim.ui.file_table import AudioFileTableModel, AudioFileTableView
from vox_verbatim.ui.folder_panel import FolderPanel
from vox_verbatim.ui.help_dialogs import KeyboardShortcutsDialog, show_about
from vox_verbatim.ui.review_lists import flagged_in, speaker_doubts_in
from vox_verbatim.ui.review_window import ReviewWindow
from vox_verbatim.ui.settings_dialog import SettingsDialog
from vox_verbatim.ui.smooth_commands import (
    MAKE_AGAIN_KEY,
    OPEN_KEY,
    SmoothResult,
    SmoothRunner,
    busy_message,
    open_smooth_transcript,
    starting_message,
)
from vox_verbatim.ui.transcribe_dialog import TranscribeDialog
from vox_verbatim.ui.player_panel import (
    STATUS_FINISHED,
    STATUS_PAUSED,
    STATUS_PLAYING,
    PlayerPanel,
    skip_button_name,
)

_log = logging.getLogger(__name__)

#: Anything saved per recording and replaced whole when the recording is read
#: again: a flagged word, or a stretch whose speaker is in doubt.
_PerRecording = TypeVar("_PerRecording", FlaggedItem, SpeakerDoubtItem)

#: Moving through the file list with the arrow keys should not load a new
#: file into the player on every key press, so loading waits for the
#: highlight to settle for this long.
_MEDIA_LOAD_DELAY_MS = 250

#: Saving the session is cheap but pointless to repeat on every key press.
_SESSION_SAVE_DELAY_MS = 750

#: The playback statuses worth reading out. The others follow from moving
#: the highlight rather than from a playback command.
_ANNOUNCED_PLAYBACK_STATUSES = frozenset({STATUS_PLAYING, STATUS_PAUSED, STATUS_FINISHED})


class MainWindow(QMainWindow):
    """The one window of the application."""

    def __init__(
        self,
        session_store: SessionStore,
        settings_store: SettingsStore | None = None,
        parent: QWidget | None = None,
        vocabulary_store: VocabularyStore | None = None,
        calibration_store: CalibrationStore | None = None,
    ) -> None:
        super().__init__(parent)
        self._session_store = session_store
        self._session = session_store.load()
        # The settings sit beside the session by default, which keeps
        # everything the application writes for itself in one folder and
        # keeps a test's temporary folder self-contained.
        self._settings_store = settings_store or SettingsStore(
            session_store.path.parent / SETTINGS_FILE_NAME
        )
        self._settings = self._settings_store.load()
        # The vocabulary and the record of how each service has performed sit
        # beside the settings, and for the same two reasons: everything the
        # application keeps for itself belongs in one folder, and a test that
        # is given a temporary folder must not write into the real one. Both
        # are read when they are needed rather than held open, because the
        # review window adds to them while the main window is running.
        self._vocabulary_store = vocabulary_store or VocabularyStore(
            self._settings_store.path.parent / VOCABULARY_FILE_NAME
        )
        self._calibration_store = calibration_store or CalibrationStore(
            self._settings_store.path.parent / CALIBRATION_FILE_NAME
        )
        self._review_window: ReviewWindow | None = None
        # Whether a review is part way through being opened, which can take
        # a while and lets the event loop run in between; see open_project_review.
        self._opening_review = False
        # Whether this window has been asked to close. Opening a review lets
        # the event loop run between transcripts, and a close handled in that
        # time must stop the opening rather than let a window appear after
        # the application has gone; see _let_the_window_breathe_unless_closing.
        self._closing = False
        # A Transcribe dialog that was closed on a run still stopping. It is
        # kept so that a second run cannot be started on top of the first,
        # and so that the summary can be said out loud when the run finally
        # stops; it is let go when that summary arrives.
        self._stopping_dialog: TranscribeDialog | None = None
        # The review window's player, kept here only so that it can be
        # stopped the moment the window is asked to go. It belongs to the
        # window and Qt destroys it with the window; see _close_review_window.
        self._review_player: AudioPlayer | None = None
        self._folder: Path | None = None
        self._pending_checked: list[str] = []
        self._pending_selected: str | None = None
        self._needs_initial_split = True
        # Whether the file list on screen is a true reading of the folder.
        # It is not, until a scan has come back with the folder's contents.
        self._file_list_is_current = False

        self.setWindowTitle(APPLICATION_NAME)

        self._player = AudioPlayer(self)
        self._scanner = FolderScanner(self)
        self._model = AudioFileTableModel(self)
        # Makes the smooth transcript again on request, in the background.
        self._smooth_runner = SmoothRunner(self)
        self._smooth_runner.finished.connect(self._on_smooth_finished)

        self._media_load_timer = QTimer(self)
        self._media_load_timer.setSingleShot(True)
        self._media_load_timer.setInterval(_MEDIA_LOAD_DELAY_MS)
        self._media_load_timer.timeout.connect(self._load_selected_media)

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(_SESSION_SAVE_DELAY_MS)
        self._save_timer.timeout.connect(self._save_session)

        self._build_ui()
        self._build_menus()
        self._connect_signals()
        self._restore_window_layout()
        self._restore_session_folder()

    # -- Building the window --------------------------------------------

    def _build_ui(self) -> None:
        self._folder_panel = FolderPanel(self)

        self._table_label = QLabel("Audio file&s", self)
        self._table = AudioFileTableView(self)
        self._table.setModel(self._model)
        self._table_label.setBuddy(self._table)
        describe(
            self._table,
            "Audio files",
            "The audio files in the selected folder. Use the Up and Down arrow keys to "
            "choose the file to play, and the Space bar to check or clear a file for "
            "transcription.",
        )

        # This label and the status label below carry messages, so neither is
        # given an accessible name. A label has no accessible value of its
        # own: its text is its name, and naming it would hide what it says
        # behind the name.
        self._summary_label = QLabel(self)

        # Alt+F, Alt+O and Alt+L are already taken by the File menu, the
        # folder box and the Play button, so Enhance answers to Alt+N.
        self._enhance_button = QPushButton("E&nhance Audio...", self)
        describe(
            self._enhance_button,
            "Enhance Audio",
            "Makes quiet recordings louder, writing the result to a folder you "
            "choose. It works on the checked files, or on the highlighted file if "
            "none are checked.",
        )
        self._enhance_button.clicked.connect(self.show_enhance_audio)

        # Alt+N has just gone to Enhance Audio and Alt+E is part of it, so
        # Transcribe answers to Alt+T. Nothing else in this window uses it:
        # the folder box has Alt+O and Alt+B, the file list Alt+S, the player
        # Alt+L and Alt+U, and the details panel Alt+M, Alt+D, Alt+Z and Alt+I.
        self._transcribe_button = QPushButton("&Transcribe...", self)
        describe(
            self._transcribe_button,
            "Transcribe",
            "Sends the recordings to the transcription services and builds a "
            "transcript from what they all heard. It works on the checked files, or "
            "on the highlighted file if none are checked.",
        )
        self._transcribe_button.clicked.connect(self.show_transcribe)

        left = QWidget(self)
        # Wide enough for the three columns of the file list to be readable
        # without scrolling sideways. Measured in characters of the current
        # font, so that at the large text sizes a low-vision user runs, this
        # does not become a floor that pushes the window off the screen.
        left.setMinimumWidth(self.fontMetrics().averageCharWidth() * 30)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 4, 8)
        left_layout.addWidget(self._folder_panel)
        left_layout.addWidget(self._table_label)
        left_layout.addWidget(self._table, 1)
        left_layout.addWidget(self._summary_label)
        left_layout.addWidget(self._enhance_button)
        left_layout.addWidget(self._transcribe_button)

        self._player_panel = PlayerPanel(self._player, self._settings, self)
        self._info_panel = FileInfoPanel(self)

        right = QWidget(self)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 8, 8, 8)
        right_layout.addWidget(self._player_panel)
        right_layout.addWidget(self._info_panel)
        right_layout.addStretch(1)

        self._splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._splitter.addWidget(left)
        self._splitter.addWidget(right)
        self._splitter.setChildrenCollapsible(False)
        # Both sides share the extra width when the window grows: the file
        # list needs it for long file names, and the player needs it for the
        # seek bar.
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 1)
        describe(self._splitter, "Window divider")
        self.setCentralWidget(self._splitter)

        self._status_label = QLabel("Ready", self)
        status_bar = QStatusBar(self)
        status_bar.addWidget(self._status_label, 1)
        status_bar.setSizeGripEnabled(True)
        self.setStatusBar(status_bar)

        self.resize(1120, 720)
        self._update_summary()

    def _build_menus(self) -> None:
        # Each entry is a menu command, which of the three skip intervals it
        # uses, and whether it goes back or forward, so the six can be
        # relabelled when the intervals change.
        self._skip_actions: list[tuple[QAction, int, int]] = []
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        self._add_action(
            file_menu,
            "Select &Folder...",
            QKeySequence.StandardKey.Open,
            self._browse_for_folder,
        )
        self._refresh_action = self._add_action(
            file_menu, "&Refresh File List", QKeySequence(Qt.Key.Key_F5), self.refresh
        )
        file_menu.addSeparator()
        self._enhance_action = self._add_action(
            file_menu,
            "E&nhance Audio...",
            QKeySequence("Ctrl+E"),
            self.show_enhance_audio,
        )
        self._transcribe_action = self._add_action(
            file_menu,
            "&Transcribe...",
            QKeySequence("Ctrl+T"),
            self.show_transcribe,
        )
        self._review_action = self._add_action(
            file_menu,
            "&Review Transcript...",
            QKeySequence("Ctrl+R"),
            self.show_review,
        )
        self._export_action = self._add_action(
            file_menu,
            "Ex&port Transcripts...",
            QKeySequence(EXPORT_KEY),
            self.show_export,
        )
        self._make_smooth_action = self._add_action(
            file_menu,
            "&Make Smooth Transcript Again",
            QKeySequence(MAKE_AGAIN_KEY),
            self.make_smooth_transcript_again,
        )
        self._open_smooth_action = self._add_action(
            file_menu,
            "&Open Smooth Transcript",
            QKeySequence(OPEN_KEY),
            self.open_smooth_transcript,
        )
        file_menu.addSeparator()
        # Named outright rather than as StandardKey.Preferences: Qt gives
        # Preferences no key at all on Windows, so the action had no shortcut
        # there although the README and the F1 list both name this one.
        self._settings_action = self._add_action(
            file_menu,
            "&Settings...",
            QKeySequence("Ctrl+,"),
            self.show_settings,
        )
        self._add_action(file_menu, "Open &Log File", None, self.open_log_file)
        self._add_action(file_menu, "Open Settings Fol&der", None, self.open_settings_folder)
        file_menu.addSeparator()
        self._add_action(file_menu, "E&xit", QKeySequence("Ctrl+Q"), self.close)

        playback_menu = menu_bar.addMenu("&Playback")
        self._add_action(
            playback_menu,
            "Play or &Pause",
            QKeySequence("Ctrl+Space"),
            self.toggle_play_pause,
        )
        # These six read their distance from the settings, so their labels
        # are worked out rather than written down. They carry no Alt letter
        # of their own for that reason; each has a shortcut key instead, and
        # the arrow keys walk the menu.
        playback_menu.addSeparator()
        for interval, shortcut in enumerate(("Alt+Left", "Alt+Shift+Left", "Alt+Ctrl+Left")):
            self._add_skip_action(playback_menu, interval, -1, shortcut)
        playback_menu.addSeparator()
        for interval, shortcut in enumerate(("Alt+Right", "Alt+Shift+Right", "Alt+Ctrl+Right")):
            self._add_skip_action(playback_menu, interval, 1, shortcut)

        view_menu = menu_bar.addMenu("&View")
        self._add_action(
            view_menu, "&Next Panel", QKeySequence(Qt.Key.Key_F6), self.focus_next_panel
        )

        help_menu = menu_bar.addMenu("&Help")
        self._add_action(
            help_menu,
            "&Keyboard Shortcuts",
            QKeySequence(Qt.Key.Key_F1),
            self._show_keyboard_shortcuts,
        )
        self._add_action(help_menu, f"&About {APPLICATION_NAME}", None, lambda: show_about(self))

    def _add_action(self, menu, text: str, shortcut, slot) -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        menu.addAction(action)
        # Menu shortcuts must work wherever the focus is, including inside
        # the file list and the player controls.
        self.addAction(action)
        return action

    def _add_skip_action(self, menu, interval: int, direction: int, shortcut: str) -> QAction:
        """Add one of the six skip commands to the Playback menu.

        ``interval`` picks one of the three intervals from the settings and
        ``direction`` is -1 for back or 1 for forward. Both the distance and
        the wording follow the settings, so changing an interval changes the
        menu as well as the buttons.
        """
        action = self._add_action(
            menu,
            skip_button_name(direction * self._settings.skip_seconds[interval]),
            QKeySequence(shortcut),
            lambda _checked=False, i=interval, d=direction: self.skip(
                d * self._settings.skip_seconds[i] * 1000
            ),
        )
        self._skip_actions.append((action, interval, direction))
        return action

    def _connect_signals(self) -> None:
        self._folder_panel.folderChosen.connect(self._on_folder_chosen)

        self._table.toggleCheckRequested.connect(self._model.toggle_checked)
        selection_model = self._table.selectionModel()
        if selection_model is not None:
            selection_model.currentRowChanged.connect(self._on_current_row_changed)

        self._model.fileChecked.connect(self._on_file_checked)
        self._model.checkedFilesChanged.connect(self._update_summary)
        self._model.checkedFilesChanged.connect(self._schedule_save)

        self._scanner.filesFound.connect(self._on_files_found)
        self._scanner.durationsRead.connect(self._on_durations_read)
        self._scanner.scanFailed.connect(self._on_scan_failed)
        self._scanner.scanFinished.connect(self._on_scan_finished)

        self._player.positionChanged.connect(self._info_panel.set_position)
        self._player.durationChanged.connect(self._on_player_duration_changed)
        self._player.errorOccurred.connect(self._on_player_error)
        self._player_panel.statusChanged.connect(self._on_playback_status_changed)
        self._player_panel.playRequested.connect(self.play)
        self._player_panel.pauseRequested.connect(self.pause)
        self._player_panel.skipRequested.connect(self.skip)
        self._player_panel.seekRequested.connect(self.seek_to)
        self._player_panel.focusReleased.connect(self._on_player_focus_released)

    # -- Folders and files ----------------------------------------------

    def _restore_session_folder(self) -> None:
        """Reopen the folder from the last run, if it is still there."""
        folder_text = self._session.folder
        if not folder_text:
            self._set_status("Select a folder to begin.")
            return
        if not self._settings.reopen_last_folder:
            self._set_status(
                "Select a folder to begin. Reopening the last folder is switched off "
                "in the settings."
            )
            return
        folder = Path(folder_text)
        self._folder_panel.set_folder(folder)
        if not folder.is_dir():
            self._folder = folder
            # Held for F5 once the drive is back, the same as for a scan.
            self._pending_checked = list(self._session.checked_files)
            self._pending_selected = self._session.selected_file
            self._set_status(
                f"The folder {folder} is not available. It may be on a drive that is "
                "not connected.",
                alert=True,
                urgent=True,
            )
            return
        self._load_folder(
            folder,
            checked=self._session.checked_files,
            selected=self._session.selected_file,
        )

    def _browse_for_folder(self) -> None:
        """Ask for a folder, unless a review is part way through opening."""
        if self._refuse_folder_change_while_opening_review():
            return
        self._folder_panel.browse()

    def _on_folder_chosen(self, folder_text: str) -> None:
        if self._refuse_folder_change_while_opening_review():
            return
        folder = Path(folder_text)
        self._folder_panel.set_folder(folder)
        self._load_folder(folder)
        self._table.setFocus()

    def _refuse_folder_change_while_opening_review(self) -> bool:
        """Say no to a change of folder while a review is being opened.

        Opening a review reads the folder's transcripts on this thread and
        lets the event loop run between them, which is long enough for a
        folder change to arrive. Letting it through would change the folder
        under the opening: the review would be built from the old folder's
        transcripts and then shown under the new folder's name, with every
        decision in it saved to the wrong project. Returns whether the change
        was refused.
        """
        if not self._opening_review:
            return False
        self._set_status(
            "The review is still being opened, so the folder cannot be changed yet. "
            "Wait for the review to appear.",
            alert=True,
            urgent=True,
        )
        return True

    def refresh(self) -> None:
        """Read the current folder again, keeping the checked and selected files."""
        if self._folder is None:
            self._set_status("There is no folder to refresh.", alert=True)
            return
        if self._file_list_is_current:
            checked = self._model.checked_names()
            selected = self._selected_file_name()
        else:
            # The list has not been read yet: the folder was unavailable at
            # start-up, or its first scan has not finished. The empty list
            # says nothing about what was checked, and passing it on would
            # let the next save throw the remembered files away.
            checked = self._pending_checked
            selected = self._pending_selected
        self._load_folder(self._folder, checked=checked, selected=selected)

    def _load_folder(
        self,
        folder: Path,
        checked: list[str] | None = None,
        selected: str | None = None,
    ) -> None:
        # A review window belongs to the folder it was opened on: its project
        # file, its transcripts and its player all live there. Leaving it open
        # over a different folder would put two projects in front of the person
        # at once with one status bar between them, and no way to tell which
        # folder a message was about. Nothing is lost by closing it, because
        # the review window writes each decision away as it is made. Reading
        # the same folder again is not a change of folder and leaves it alone,
        # so refreshing the file list does not take a review away.
        if self._refuse_folder_change_while_opening_review():
            return
        if self._folder is not None and folder != self._folder:
            self._close_review_window()
        self._folder = folder
        self._pending_checked = list(checked or [])
        self._pending_selected = selected
        self.setWindowTitle(f"{folder.name} - {APPLICATION_NAME}")
        self._set_status(f"Reading {folder}...")
        self._file_list_is_current = False
        self._scanner.start(folder)
        self._schedule_save()

    def _on_files_found(self, files: list[AudioFile], scan_id: int) -> None:
        if scan_id != self._scanner.current_scan_id:
            return
        self._model.set_files(files)
        # The list now really is what the folder holds, empty or not, so it
        # is worth saving.
        self._file_list_is_current = True
        self._model.set_checked_names(self._pending_checked)
        self._update_summary()

        if not files:
            self._clear_selection()
            self._set_status(
                f"No audio files were found in {self._folder}.",
                alert=True,
            )
            return

        row = self._model.row_for_name(self._pending_selected)
        if row < 0:
            row = 0
        self._table.select_row(row)
        self._update_for_row(row)

    def _on_durations_read(self, durations: list[tuple[str, float | None]], scan_id: int) -> None:
        if scan_id != self._scanner.current_scan_id:
            return
        self._model.update_durations(durations)
        selected_name = self._selected_file_name()
        if selected_name and any(name == selected_name for name, _ in durations):
            self._info_panel.refresh_duration()

    def _on_scan_failed(self, message: str, scan_id: int) -> None:
        if scan_id != self._scanner.current_scan_id:
            return
        self._model.clear()
        self._clear_selection()
        self._update_summary()
        self._set_status(message, alert=True, urgent=True)

    def _on_scan_finished(self, scan_id: int) -> None:
        if scan_id != self._scanner.current_scan_id:
            return
        count = self._model.rowCount()
        if count:
            self._set_status(f"{self._summary_text()} in {self._folder}.", alert=True)

    # -- Selection ------------------------------------------------------

    def _on_current_row_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
        self._update_for_row(current.row() if current.isValid() else -1)

    def _update_for_row(self, row: int) -> None:
        audio_file = self._model.file_at(row)
        self._info_panel.set_file(audio_file)
        # Playback stops as soon as the highlight moves, before the new file
        # has finished loading. Otherwise the previous file would still be
        # heard while the panel already reports the new one as stopped.
        self._player.stop()
        self._player_panel.set_media_loaded(audio_file is not None)
        if audio_file is None:
            self._media_load_timer.stop()
            self._player.load(None)
        else:
            self._media_load_timer.start()
        self._schedule_save()

    def _clear_selection(self) -> None:
        self._table.clearSelection()
        self._table.setCurrentIndex(QModelIndex())
        self._update_for_row(-1)

    def _load_selected_media(self) -> None:
        self._media_load_timer.stop()
        audio_file = self._model.file_at(self._table.selected_row())
        if audio_file is None:
            self._player.load(None)
            return
        if self._player.source_path == audio_file.path:
            return
        self._player.load(audio_file.path)

    # -- Transport ------------------------------------------------------
    #
    # Every playback command goes through here first, so that the file the
    # user is looking at is the file the command acts on. Loading a file is
    # held back for a moment after the highlight moves, and without this a
    # command given in that moment would act on the file before it.

    def play(self) -> None:
        self._load_selected_media()
        self._player.play()

    def pause(self) -> None:
        self._load_selected_media()
        self._player.pause()

    def toggle_play_pause(self) -> None:
        self._load_selected_media()
        self._player.toggle_play_pause()

    def skip(self, milliseconds: int) -> None:
        self._load_selected_media()
        self._player.skip(milliseconds)

    def seek_to(self, milliseconds: int) -> None:
        self._load_selected_media()
        self._player.seek_to(milliseconds)

    def _selected_file_name(self) -> str | None:
        audio_file = self._model.file_at(self._table.selected_row())
        return audio_file.key if audio_file is not None else None

    # -- Player ---------------------------------------------------------

    def _on_player_duration_changed(self, milliseconds: int) -> None:
        """Take the player's word for the duration when the file header did not say.

        A file whose header could not be parsed still has a length once it
        is open for playback, so the list and the details panel are filled
        in from there rather than left reading "Unknown".
        """
        if milliseconds <= 0:
            return
        source = self._player.source_path
        if source is None:
            return
        row = self._model.row_for_name(source.name)
        audio_file = self._model.file_at(row)
        if audio_file is None or audio_file.duration_state == DurationState.KNOWN:
            return
        self._model.set_duration_for_name(source.name, milliseconds / 1000.0)
        if self._selected_file_name() == source.name:
            self._info_panel.refresh_duration()

    def _on_player_error(self, message: str) -> None:
        self._set_status(message, alert=True, urgent=True)

    def _on_playback_status_changed(self, status: str) -> None:
        # Only the statuses that follow from something the user did are read
        # out. "Stopped" and "No file loaded" come from moving the highlight
        # in the file list, and announcing those would interrupt the reading
        # of each file name on the way down the list.
        self._set_status(status, alert=status in _ANNOUNCED_PLAYBACK_STATUSES)

    def _on_player_focus_released(self) -> None:
        """Catch the focus when the transport controls switch off underneath it."""
        self._focus_file_table()
        self._set_status(
            "No file is loaded. The focus has moved to the audio file list.",
            alert=True,
            urgent=True,
        )

    # -- Checked files --------------------------------------------------

    def _on_file_checked(self, name: str, checked: bool) -> None:
        state = "checked" if checked else "cleared"
        self._set_status(f"{name} {state}. {self._summary_text()}.")
        # The check box does report its own new state, but only some screen
        # readers re-read a cell after the state behind it changes. Saying
        # it plainly costs a few words and removes the doubt.
        announce(self._status_label, f"{name} {state}")

    # -- Enhancing audio --------------------------------------------------

    def _chosen_files(self) -> list[AudioFile]:
        """The recordings a command in this window would work on.

        The checked files are what the check boxes are for, so they win.
        Where nothing is checked, the highlighted file is what the user is
        looking at and is taken to be what they mean.

        Every command answers this the same way, deliberately. A window where
        Enhance Audio acted on the checked files and Transcribe acted on the
        highlighted one would be a window nobody could predict.
        """
        checked = self._model.checked_files()
        if checked:
            return checked
        selected = self._model.file_at(self._table.selected_row())
        return [selected] if selected is not None else []

    def files_to_enhance(self) -> list[AudioFile]:
        """The recordings an enhancement run would work on."""
        return self._chosen_files()

    def show_enhance_audio(self) -> None:
        """Open the Enhance Audio dialog on the chosen files."""
        files = self.files_to_enhance()
        if not files:
            self._set_status(
                "There is nothing to enhance. Check the files you want, or highlight "
                "one in the file list.",
                alert=True,
                urgent=True,
            )
            return
        dialog = EnhanceAudioDialog(
            [audio_file.path for audio_file in files],
            self._settings.enhance,
            source_folder=self._folder,
            parent=self,
        )
        dialog.exec()
        chosen = dialog.chosen_settings()
        summary = dialog.summary
        # The dialog belongs to this window, so closing it does not get rid
        # of it. Left alone, every opening of Enhance Audio would add
        # another one that lives as long as the application does, each with
        # its own background runner attached. What it was left on is read
        # out first, because after this it is on its way out.
        dialog.deleteLater()
        self._remember_enhance_settings(chosen)
        if summary is not None:
            self._set_status(summarise(summary), alert=True)
        else:
            self._set_status("Enhance Audio was closed without enhancing anything.")

    def _remember_enhance_settings(self, enhance: EnhanceSettings) -> None:
        """Keep what the user chose, so the dialog opens the same way next time.

        This happens whether or not anything was enhanced, because setting
        the parameters up and then closing the dialog is a reasonable thing
        to do.

        A failure to save is only logged here, rather than announced. What
        the user is waiting to hear is how the run went, and burying that
        under a message about the settings file would serve them badly. The
        Settings dialog reports the same failure plainly.
        """
        self._settings = replace(self._settings, enhance=enhance)
        if not self._settings_store.save(self._settings):
            _log.warning("The Enhance Audio settings could not be saved.")

    # -- Transcribing -----------------------------------------------------

    def files_to_transcribe(self) -> list[AudioFile]:
        """The recordings a transcription run would work on."""
        return self._chosen_files()

    def show_transcribe(self) -> None:
        """Open the Transcribe dialog on the chosen recordings.

        The vocabulary is read fresh each time rather than held, because
        reviewing a transcript adds the corrections a person made to it, and a
        copy loaded when the window opened would be out of date by the second
        run.
        """
        files = self.files_to_transcribe()
        if not files:
            self._set_status(
                "There is nothing to transcribe. Check the files you want, or "
                "highlight one in the file list.",
                alert=True,
                urgent=True,
            )
            return
        if self._previous_run_is_still_stopping():
            # Two runs at once would both be writing transcripts beside the
            # same recordings, and the first to finish would withdraw the
            # request keeping the machine awake from under the second.
            self._set_status(
                "The previous run is still stopping. Wait for it to finish before "
                "starting another.",
                alert=True,
                urgent=True,
            )
            return
        dialog = TranscribeDialog(
            files,
            self._settings.transcription,
            vocabulary=self._vocabulary_store.load(),
            parent=self,
        )
        dialog.exec()
        if dialog.is_stopping_in_background:
            # Closed on a run that had been asked to stop and was still
            # waiting on a service. The dialog is left alive until the run
            # has stopped, because the runner inside it is what the run
            # reports to and what withdraws the request keeping the machine
            # awake; the dialog gets rid of itself once the summary arrives.
            # This window keeps hold of it until then, so that it can refuse
            # a second run and can say how the first went when it stops.
            # The person is told plainly, because a window that closed on a
            # running job and said nothing would leave them wondering whether
            # the recordings already done had been kept.
            self._stopping_dialog = dialog
            dialog.detachedRunFinished.connect(self._on_detached_run_finished)
            self._set_status(
                "Transcribe was closed while the run was still stopping. It stops by "
                "itself when the request it is waiting on comes back. Recordings "
                "already transcribed are saved beside their recordings.",
                alert=True,
            )
            return
        summary = dialog.summary
        review = dialog.review_request
        # The dialog belongs to this window, so closing it does not get rid of
        # it. What it was left holding is read out first, because after this
        # it is on its way out.
        dialog.deleteLater()
        if summary is not None:
            self._set_status(summarise_transcription(summary), alert=True)
        else:
            self._set_status("Transcribe was closed without transcribing anything.")
        if review is not None and review.succeeded:
            # Asked for from inside the dialog, and answered out here. The
            # dialog is modal, so a review window opened from within it would
            # appear behind something the user cannot dismiss.
            #
            # The review covers the whole folder, as it always does, and the
            # recording just transcribed is only where the person is put down
            # in it. That matters more than it sounds: a word in the new file
            # is very likely the same word as one already settled in an older
            # file, and opening on the new file alone would hide exactly the
            # decision that has already been taken.
            self.open_project_review(land_on=review.path.name)

    # -- Exporting transcripts ---------------------------------------------

    def show_export(self) -> None:
        """Export the chosen recordings' transcripts to a folder the person picks."""
        files = self._chosen_files()
        if not files:
            self._set_status(
                "There is nothing to export. Check the files you want, or highlight "
                "one in the file list.",
                alert=True,
                urgent=True,
            )
            return
        self.export_recordings([audio_file.path for audio_file in files])

    def export_recordings(
        self,
        recordings: list[Path],
        parent: QWidget | None = None,
        say: Callable[..., None] | None = None,
        scope_text: str | None = None,
    ) -> None:
        """Run one export of ``recordings``, with its boxes in front of ``parent``.

        The review window calls this for the recording it is on, passing
        itself and its own status line, so the dialog opens in front of it
        and what happened is said where the person is working. The settings
        and the transcript folders are this window's, so an export from
        either window remembers the same folder and check boxes.
        """
        flow = ExportFlow(
            parent if parent is not None else self,
            say if say is not None else self._set_status,
            settings=lambda: self._settings.export,
            save_settings=self._remember_export_settings,
            store_for=self._transcript_store,
        )
        flow.run(recordings, scope_text)

    def _remember_export_settings(self, export: ExportSettings) -> None:
        """Keep what was chosen, so the dialog opens the same way next time."""
        self._settings = replace(self._settings, export=export)
        if not self._settings_store.save(self._settings):
            _log.warning("The Export Transcripts settings could not be saved.")

    # -- The smooth transcript ---------------------------------------------

    def _highlighted_file(self) -> AudioFile | None:
        return self._model.file_at(self._table.selected_row())

    def make_smooth_transcript_again(self) -> None:
        """Make the highlighted recording's smooth transcript again, in the background.

        It is made from the transcript as it is saved now, so every correction
        made in the review window is in it. The folder's own smoothing prompt
        is used where it has one.
        """
        audio_file = self._highlighted_file()
        if audio_file is None:
            self._set_status(
                "Highlight a recording in the file list first.", alert=True, urgent=True
            )
            return
        if self._smooth_runner.is_running():
            self._set_status(busy_message(), alert=True, urgent=True)
            return
        name = audio_file.path.name
        store = self._transcript_store(audio_file.path)
        transcript = store.load()
        if transcript is None:
            self._set_status(
                f"{name} has no transcript yet, so there is nothing to smooth. "
                "Transcribe it first.",
                alert=True,
                urgent=True,
            )
            return
        smoother = smoothing.smoother_for(self._settings.transcription, audio_file.path.parent)
        self._smooth_runner.start(name, transcript, store, smoother, requester=self)
        self._set_status(starting_message(name), alert=True)

    def open_smooth_transcript(self) -> None:
        """Open the highlighted recording's smooth transcript in the text editor."""
        audio_file = self._highlighted_file()
        if audio_file is None:
            self._set_status(
                "Highlight a recording in the file list first.", alert=True, urgent=True
            )
            return
        opened, message = open_smooth_transcript(
            audio_file.path.name, self._transcript_store(audio_file.path)
        )
        self._set_status(message, alert=True, urgent=not opened)

    def _on_smooth_finished(self, result: SmoothResult, requester: object) -> None:
        """Say how a run went, unless the review window that asked will say it.

        A run the review window asked for is said here only when that window
        has closed since, so the result is never lost and never said twice.
        Closing the review window with its own close button hides it rather
        than deleting it, so it is still held here; whether it is on screen
        is what tells.
        """
        window = self._review_window
        if requester is not self and requester is window and window.isVisible():
            return
        self._set_status(result.message, alert=True, urgent=not result.succeeded)

    def _previous_run_is_still_stopping(self) -> bool:
        """Whether a run closed in the background has not yet stopped."""
        dialog = self._stopping_dialog
        if dialog is None:
            return False
        if dialog.is_stopping_in_background:
            return True
        # It stopped without this window hearing, which should not happen
        # but is not a reason to refuse every run from now on.
        self._stopping_dialog = None
        return False

    def _on_detached_run_finished(self, summary: RunSummary) -> None:
        """Say how a run closed in the background went, now that it has stopped.

        The dialog that would have shown the summary is hidden and about to
        delete itself, so this is the only place the person can hear it. It
        arrives minutes after they closed the dialog, so it is announced
        rather than merely written, and it says that it is about the earlier
        run rather than about whatever they are doing now.
        """
        self._stopping_dialog = None
        self._set_status(
            f"The run that was stopping has finished. {summarise_transcription(summary)}",
            alert=True,
        )

    # -- Reviewing --------------------------------------------------------
    #
    # A folder is a project, and reviewing is a thing done to the project
    # rather than to one recording. That is the whole shape of what follows,
    # and it is worth saying why, because the obvious design is the other one.
    #
    # The same surname comes out of the services spelled three ways across
    # four interviews. Reviewed one recording at a time, the person decides
    # that surname four times and has no way of knowing they are the same
    # decision. Reviewed as a folder, it is one decision that lands
    # everywhere, and what they accepted is remembered so that the interview
    # transcribed next week is answered without asking them again.
    #
    # Nothing crosses between folders. Everything the project remembers lives
    # in a file in the audio folder itself, so two folders are two projects
    # with nothing shared, and a correction accepted for one client's
    # interviews can never rewrite another client's. That is the point of the
    # design rather than a side effect of where the file sits.

    def show_review(self) -> None:
        """Open the review window on the current folder's project.

        Which recording the highlight happens to be on decides nothing here,
        and that is deliberate rather than an oversight. A person who has
        arrowed down to the third file has not thereby said that they want to
        review only the third file, and a review that covered one recording
        would make them take the same decision once per file.
        """
        self.open_project_review()

    def open_project_review(self, land_on: str | None = None) -> ReviewWindow | None:
        """Open the review window on this folder, or say why it cannot be opened.

        ``land_on`` is the file name of a recording to put the person on when
        the window appears, which is how finishing a transcription hands over.
        They asked to review what has just been transcribed, so they should
        arrive on a word from it rather than wherever they left off last week.
        Nothing is lost when that recording has no words waiting: the window
        opens where the project says the person had got to, and says so.

        Returns the window, or ``None`` when there was nothing to open it on.
        """
        if self._opening_review:
            # The window gives the event loop a turn between transcripts so
            # that it can repaint, and that turn is enough for a second
            # Ctrl+R to arrive. Starting a second opening inside the first
            # would read the folder twice and save the project twice, so the
            # second is refused and says why.
            self._set_status(
                "The review is still being opened. Wait for it to appear.",
                alert=True,
                urgent=True,
            )
            return None
        if self._folder is None:
            self._set_status(
                "Select a folder before reviewing. A review covers the whole folder "
                "rather than one recording.",
                alert=True,
                urgent=True,
            )
            return None
        if not self._file_list_is_current:
            # The folder has not been read this run, so the file list says
            # nothing about what is in it. Refusing on that basis would be
            # telling the person their folder is empty when the truth is only
            # that we have not looked yet.
            self._set_status(
                f"{self._folder} has not been read yet, so there is nothing to review "
                "from. Try again once the file list has filled in.",
                alert=True,
                urgent=True,
            )
            return None

        # The folder is taken once, here, and used for the whole opening.
        # Reading lets the event loop run between transcripts, and a change
        # of folder is refused in that time, but this is the belt to that
        # brace: whatever self._folder comes to say, the window is built on
        # the folder whose transcripts were read.
        folder = self._folder
        recording_names, recording_paths, stores = self._folder_recordings()
        if not recording_names:
            self._refuse_empty_review([])
            return None

        # Any review already open goes now, before this one reads the project
        # file, and the order is the whole point rather than tidiness. A review
        # window writes the project as it closes, to record where the person
        # had got to. Closing it after this opening had read, analysed and
        # saved the project would put the window's older copy back over the
        # top, and the new window would then load that: the analysis just run
        # would be silently undone, and a recording transcribed since the last
        # review would show none of its words. Closing first means the two
        # writes happen in the order they were caused in.
        self._close_review_window()
        # Said before the first transcript is read rather than after, because
        # reading is the slow part. A folder of long recordings is minutes of
        # parsing on this thread, during which the window cannot repaint and
        # a screen reader has nothing new to say, so the person is told what
        # is happening and roughly why it is taking a while. The cursor says
        # the same thing to anybody who can see it.
        self._set_status(
            f"Reading {_count(len(recording_names), 'transcript')} in this folder. "
            "This can take a while.",
            alert=True,
            urgent=True,
        )
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._opening_review = True
        try:
            return self._open_project_review_after_saying_so(
                folder, land_on, recording_names, recording_paths, stores
            )
        except _WindowClosing:
            # The window was closed while the transcripts were being read.
            # Nothing has been saved yet, so there is nothing to undo, and a
            # review window appearing after the application has gone is the
            # one thing this must not do.
            _log.info("The review was not opened because the window is closing.")
            return None
        finally:
            self._opening_review = False
            QApplication.restoreOverrideCursor()

    def _open_project_review_after_saying_so(
        self,
        folder: Path,
        land_on: str | None,
        recording_names: list[str],
        recording_paths: dict[str, Path],
        stores: dict[str, TranscriptStore],
    ) -> ReviewWindow | None:
        """The part of :meth:`open_project_review` that does the reading."""
        reader = _TranscriptReader(
            stores, between_recordings=self._let_the_window_breathe_unless_closing
        )
        project_store = ProjectStore(folder)
        # Loading this folder's own project is what resumes it: the groups,
        # the decisions, the rules, the settings and the marker saying where
        # the person had got to all come from the file in this folder and from
        # nowhere else. A folder that has never been reviewed comes back as an
        # empty project rather than as a failure.
        state = project_store.load()
        analysing = self._recordings_to_analyse(state, recording_names, stores)
        # Asked before a word of any of them is read, and the order is what
        # makes it safe rather than tidy. Something else may be writing a
        # transcript at this very moment. Noting its age first and reading it
        # afterwards means the note describes a file at least as old as what
        # was read, so a write that lands in between makes the next opening
        # read the recording once more, needlessly. Noting it afterwards would
        # make that same write disappear: the note would describe the new file
        # while the analysis had read the old one, and nothing would ever look
        # at that recording again.
        analysed_as = _transcript_times(analysing, stores)
        corrected, problems = self._answer_with_project_rules(state, analysing, reader)
        gathered: dict[str, list[FlaggedItem]] = {}
        doubts: dict[str, list[SpeakerDoubtItem]] = {}
        # The folder listing is handed over as well as the shorter list of
        # recordings to read, and the two are different questions. The second
        # says which transcripts have to be parsed on this run; the first says
        # which recordings the folder still holds at all, which is the only
        # way the analysis can tell a file somebody deleted from one that
        # could not be read just now. Both look identical to the loader.
        state = grouping.reprocess(
            state,
            analysing,
            self._reading_and_noting(reader, gathered, doubts),
            present_recordings=recording_names,
        )
        state.flagged = _flagged_after(state.flagged, analysing, gathered)
        state.speaker_doubts = _flagged_after(state.speaker_doubts, analysing, doubts)
        written, unwritten_problems = self._write_recorded_answers(state, analysing, reader)
        corrected += written
        problems.extend(unwritten_problems)

        damaged = set(reader.damaged)
        if damaged == set(recording_names):
            # Everything this folder had was tried and none of it could be
            # read. Opening a window on it would show an empty review of five
            # recordings and say nothing about why, which is worse than saying
            # plainly that there is nothing here.
            self._refuse_empty_review(reader.damaged)
            return None
        # The window is given the recordings whose transcripts can be read. A
        # file already known to be damaged is left out of the list rather than
        # offered and then refused, so that nothing in the window stands for a
        # recording there is nothing to show for. One that goes bad later is a
        # different matter and cannot be foreseen: the loader answers with
        # nothing and says so at the time.
        readable = [name for name in recording_names if name not in damaged]

        # What each transcript said its own age was, for the next opening to
        # compare against. A recording that could not be read is deliberately
        # left without a note, because a note here is a promise that the file
        # has been looked at, and the promise would spare it from ever being
        # looked at again.
        state.transcript_times = _times_after(
            state.transcript_times, recording_names, analysed_as, damaged
        )
        # Stamped here rather than left as the analysis wrote it, so that it
        # says when this opening finished rather than when the middle of it
        # ran. Nothing decides what to read from this; _recordings_to_analyse
        # explains why not, and what it uses instead. What this is for is
        # telling somebody, and telling the review window, when the folder was
        # last gone through.
        state.processed_at = datetime.now().astimezone().isoformat()
        if not project_store.save_keeping_learned_names(state, []):
            problems.append(
                f"The review of this folder could not be saved to {project_store.path}."
            )

        # The review window gets a player of its own rather than sharing this
        # window's. Two windows driving one player fight over what is loaded:
        # this window loads whatever the highlight is on, the review window
        # loads whichever recording the selected occurrence belongs to, and
        # whichever moved last wins, so the person hears the wrong recording
        # roughly half the time. It also keeps reviewing out of the player
        # panel, whose position, duration and transport buttons would
        # otherwise follow words the person is not looking at.
        if self._closing:
            raise _WindowClosing()
        # The reader stops giving the event loop a turn from here on. It gave
        # one between transcripts so that the window could repaint during the
        # opening, but in the review window's hands that turn would let a
        # key press ask for a second recording part way through reading the
        # first, and the window would then file one recording's transcript
        # under the other's name.
        reader.between_recordings = None
        player = AudioPlayer()
        window = ReviewWindow(
            folder,
            readable,
            reader.load,
            recording_paths,
            project_store,
            player,
            save_correction=(
                lambda name, changed: self._save_correction(reader, name, changed)
            ),
            parent=self,
            record_statistics=lambda change: self._calibration_store.apply(change.apply),
            vocabulary=user_terms_index(self._vocabulary_store.load()),
            app_smoothing_prompt=self._settings.transcription.smoothing.prompt,
            write_exports=reader.write_exports,
            transcript_store_for=(
                lambda name: self._transcript_store(recording_paths[name])
                if name in recording_paths
                else None
            ),
            build_smoother=(
                lambda: smoothing.smoother_for(self._settings.transcription, folder)
            ),
            smooth_runner=self._smooth_runner,
            export_recordings=self.export_recordings,
        )
        # Given its parent after the window exists rather than before, so that
        # Qt destroys the player along with the window it belongs to. A player
        # parented to this window instead would outlive every review and go on
        # holding an open media handle on a file the person has finished with.
        player.setParent(window)
        self._review_window = window
        self._review_player = player
        window.show()

        # Landed silently, because the sentence below has more to say than the
        # row does and used to be cut across by it. The window's own
        # announcement is the right thing everywhere else, and this is the one
        # caller with something better.
        landed = land_on is not None and window.select_recording(
            land_on, announce_arrival=False
        )
        message = self._review_opening_message(
            folder, len(readable), corrected, reader.damaged, land_on, landed, problems
        )
        # On the screen only. The main window is behind the review window by
        # now, and an announcement raised from a window that is not in front is
        # unreliable across screen readers -- and this one used to arrive
        # assertively, on top of the review window's own, so the two facts a
        # person most needed fought each other. The sentence is still written
        # here for anyone who can look at it.
        self._set_status(message)
        self._land_in_review(window, message, urgent=bool(problems))
        # Only now, so that damage found while opening is reported once, in the
        # sentence above, rather than twice. From here on a transcript that
        # turns out to be unreadable is found when the person reaches it, which
        # is the price of not reading the folder in advance, and it is said out
        # loud at that moment rather than being swallowed.
        reader.on_damaged = self._report_damaged_transcript
        return window

    @staticmethod
    def _land_in_review(window: ReviewWindow, message: str, urgent: bool) -> None:
        """Put the person into the review window, in one place and with one sentence.

        Two things happen here and the order between them is the whole point.

        The focus is placed first, on the first list, so that the screen
        reader reads out the word the person has been put on. Left to itself,
        the focus lands on whichever control Qt decides is first in a window
        that has just been shown, which happens to be that list today and is
        nobody's decision. A window whose starting point depends on the widget
        order is a window that will one day open somewhere else because
        somebody added a control.

        The sentence is announced second, so that it follows the row rather
        than talking over it. It carries every fact the person cannot see and
        cannot get anywhere else: how many recordings are in the review, how
        many words a rule they accepted earlier has already corrected on their
        behalf in files nobody has opened, which transcripts could not be read,
        and where they have been put down. It is raised on the review window
        because that is the window in front.

        It interrupts only when something went wrong. A problem is worth
        cutting a row reading short for; a count is not.
        """
        window.focus_word_list()
        announce(window, message, urgent=urgent)

    def _report_damaged_transcript(self, recording_name: str) -> None:
        """Say that a transcript could not be read, at the moment that is found.

        The last sentence is the important one and it is not padding. A
        transcript that could not be read is a transcript that could not be
        read, and nothing more: the words of that recording are still in the
        review, still in their groups, and still carrying every decision the
        person has made about them. Somebody hearing only that a file could not
        be read would reasonably conclude their work on it had gone, and would
        either redo it or stop trusting the window, so the message says plainly
        that it has not.

        This is deliberately different from what is said at the top of the
        opening, where a recording found unreadable really is left out of the
        review before the window is given anything. Here the window is already
        open and the recording is in it.
        """
        self._set_status(
            f"The transcript for {recording_name} could not be read just now, so its "
            "words cannot be shown or played. Nothing you have decided about them has "
            "been lost.",
            alert=True,
            urgent=True,
        )

    def _folder_recordings(
        self,
    ) -> tuple[list[str], dict[str, Path], dict[str, TranscriptStore]]:
        """Every recording in this folder that has a transcript, named not read.

        Asking whether a transcript file exists costs one look at the file
        system; reading it costs tens of megabytes. So this settles which
        recordings are in the review and reads none of them, and whether a
        transcript is actually readable is found out later, by whoever first
        needs its contents.

        That is a real loss and it is worth naming rather than glossing over.
        A damaged transcript used to be caught here, before the window opened,
        and now it cannot be, because catching it means parsing every file in
        the folder and parsing every file in the folder is the cost being
        removed. What replaces it is honesty about the order of events: the
        opening message reports the damage found while opening, and anything
        found afterwards is announced the moment it is found. Nothing claims
        the folder is sound.

        A recording that has never been transcribed is simply not reviewable
        content and is left out. It is emphatically not a reason to refuse:
        the person may well be standing on the one file nobody has transcribed
        while the other forty in the folder are full of words waiting.
        """
        recording_names: list[str] = []
        recording_paths: dict[str, Path] = {}
        stores: dict[str, TranscriptStore] = {}
        for audio_file in self._model.files():
            store = self._transcript_store(audio_file.path)
            if not store.has_transcript:
                continue
            # Keyed on the file name with its extension, which is what an
            # occurrence records as the recording it belongs to, and which
            # keeps meeting.m4a and meeting.wav apart.
            name = audio_file.path.name
            recording_names.append(name)
            recording_paths[name] = audio_file.path
            stores[name] = store
        return recording_names, recording_paths, stores

    def _recordings_to_analyse(
        self,
        state: ProjectState,
        recording_names: list[str],
        stores: dict[str, TranscriptStore],
    ) -> list[str]:
        """Which recordings this opening has to read, and why the rest are spared.

        Reading one transcript at a time made the memory affordable but did
        nothing about the time: analysing fifty recordings still means parsing
        fifty transcripts, which is most of a minute of somebody waiting for a
        window. What makes opening quick is not reading them faster but not
        reading the ones that cannot have changed, and the project file already
        holds every occurrence found in those.

        So the only recording spared is one whose transcript file is still,
        exactly, the file that was read. The project wrote down a mark for each
        transcript at the moment it was analysed, built from its age, its size
        and its file ID by :func:`_transcript_mark`, and the question here is
        whether the file still gives the same mark. If it does
        not, for any reason and in either direction, the recording is read
        again. A recording the project has no note of has never been analysed,
        or was analysed and could not be read, and is read as well. So is a
        recording whose speaker doubts were saved in the old way, as flagged
        words; see :func:`_without_speaker_stretches`.

        It is worth saying why this is a comparison for sameness rather than
        the obvious one, which is to ask whether the transcript is newer than
        the analysis. That question sounds equivalent and is not, because the
        two times in it come from two different clocks. The stamp saying when
        the analysis ran is read from the system clock, and the age a file
        carries is written by the filesystem, and on Windows those two do not
        agree: a file written immediately after the clock was read comes out
        with a time up to about ten milliseconds *before* the reading, which is
        what measuring it on Windows 11 gives. "Is the file newer" then answers
        no about a file that has genuinely changed, the recording is silently
        left out of the review, and its shaky words never reach the person at
        all, which is the one failure this whole feature exists to prevent.

        That gap is not hypothetical, and it is not only a matter of test
        timing. A correction the person makes in the review window is written
        into a transcript after this opening stamped the project, and so is a
        recording transcribed again the moment a review is closed. Both are
        exactly the case above.

        Comparing for sameness has a second thing going for it, which the
        newer-than question gets wrong even with a perfect clock: a transcript
        restored from a backup or copied back from another machine carries an
        old age and is nonetheless a different file from the one that was
        analysed. It differs, so it is read.

        The recordings written during this very opening, by a rule correcting a
        word, differ from what was noted before they were read and are
        therefore read once more on the next opening. That settles itself after
        that one run, because the next analysis notes them as they now are and
        writes nothing further to them.
        """
        wanted: list[str] = []
        old_speaker_doubts = _without_speaker_stretches(state)
        for name in recording_names:
            analysed_as = state.transcript_times.get(name)
            store = stores.get(name)
            if analysed_as is None or store is None or name in old_speaker_doubts:
                wanted.append(name)
                continue
            try:
                now = _transcript_mark(store.transcript_path)
            except OSError:
                # The file cannot even be asked about. Read it, so that the
                # trouble is reported by whoever tries rather than becoming a
                # recording quietly left out of the review.
                wanted.append(name)
                continue
            if now != analysed_as:
                wanted.append(name)
        return wanted

    @staticmethod
    def _reading_and_noting(
        reader: _TranscriptReader,
        gathered: dict[str, list[FlaggedItem]],
        doubts: dict[str, list[SpeakerDoubtItem]] | None = None,
    ) -> Callable[[str], Transcript | None]:
        """A loader that also takes the flagged words out of what it reads.

        The analysis walks the folder one recording at a time and lets each
        transcript go, which is the only pass over the folder anything makes.
        The words some rule in the pipeline flagged can be found nowhere but in
        a transcript, so they are taken here, on the way past, and saved into
        the project. It is worth being blunt about what the alternative costs,
        because the alternative looks so much simpler: reading the folder when
        the review window opens, to pick these words out, is 37 seconds and
        1.6 GB on fifty hour-long recordings, and it is paid every time
        somebody presses Ctrl+R rather than once when the folder is analysed.

        A recording that could not be read leaves nothing here, and
        :func:`_flagged_after` is what makes sure that means "keep what was
        already known about it" rather than "it has no flagged words".

        The loader must still read one recording only when it is asked for it.
        Anything that reads them all in advance has undone the whole
        arrangement, whatever it is spelled like.
        """

        def load(recording_name: str) -> Transcript | None:
            transcript = reader.load(recording_name)
            if transcript is not None:
                gathered[recording_name] = flagged_in(recording_name, transcript)
                if doubts is not None:
                    doubts[recording_name] = speaker_doubts_in(recording_name, transcript)
            return transcript

        return load

    def _answer_with_project_rules(
        self,
        state: ProjectState,
        recording_names: list[str],
        reader: _TranscriptReader,
    ) -> tuple[int, list[str]]:
        """Answer this folder's transcripts with the replacements already accepted.

        A file transcribed this morning is answered by every replacement the
        person accepted last week, without their being asked the same question
        twice. That is what this does, and it runs before the analysis rather
        than after it for one reason worth stating: a rule fires on a word the
        services were *confident* about as readily as on a weak one, and the
        analysis only ever looks at the weak ones.

        That is not an oversight in the analysis. The words a service gets
        confidently wrong are precisely the proper names, so a correction that
        only reached words already in doubt would miss the whole case for
        remembering corrections at all. What makes it safe to rewrite a word
        nobody has questioned is that nothing is lost: the word keeps what it
        originally said, the occurrence records which rule changed it, and
        rules never leave this folder.

        The occurrences that come back are put into the project before the
        analysis runs, so that the analysis knows those words are already
        spoken for and does not find them a second time.

        Only recordings the project has never analysed are offered here, and
        that restriction is the important part rather than an optimisation.
        This step judges a word on the transcript alone: it protects a word a
        person corrected by hand, because the transcript records that, and it
        can see nothing else the person decided. A recording the project
        already holds occurrences for is full of things it cannot see: a word
        somebody looked at and confirmed was right as detected, or one they
        took out of its group on purpose. Rewriting those would overrule a
        person's decision with a general rule, which is the wrong way round.
        They belong to the analysis, which leaves settled work alone. A
        recording with nothing saved against it is either genuinely new or held
        nothing worth saving, and in neither case is there a decision here to
        overrule.

        A folder with no rules yet is the ordinary case and is skipped
        outright, so that a first review does not walk every word of every
        recording to discover that there was nothing to say about any of them.
        """
        if not state.rules:
            return 0, []
        analysed = {occurrence.recording_name for occurrence in state.occurrences}
        answered_count = 0
        problems: list[str] = []
        for name in recording_names:
            if name in analysed:
                continue
            transcript = reader.load(name)
            if transcript is None:
                continue
            corrected, answered, _queued = grouping.apply_rules(state, transcript, name)
            if not answered:
                continue
            state.occurrences.extend(answered)
            answered_count += len(answered)
            # Written and let go one recording at a time, rather than gathered
            # up and saved at the end, because gathering them up is holding the
            # folder in memory by another name.
            if not reader.save(name, corrected):
                problems.append(self._unsaved_corrections(name, reader))
        return answered_count, problems

    def _write_recorded_answers(
        self,
        state: ProjectState,
        recording_names: list[str],
        reader: _TranscriptReader,
    ) -> tuple[int, list[str]]:
        """Make in the transcripts any correction the analysis only wrote down.

        The analysis records that a rule answered a word and deliberately stops
        there; writing the word is left to whoever called it. This is the
        easiest thing in the whole feature to get wrong, because an occurrence
        that says a rule answered it looks exactly like a job that has been
        done, and it is a job still to do. Read the other way round, the person
        ends up with a project insisting a word was corrected and a transcript
        that still says the old thing.

        Only the recordings just analysed are considered, and that is what
        keeps this cheap rather than a second pass over the folder. An answer
        that has not been written can only have come from an analysis, and the
        project is saved after the writing here, so a run that stopped part way
        through never reached the disk at all and left no contradiction behind
        it. What this does catch is a contradiction written by something else,
        and that recording is looked at again the next time anything analyses
        it.

        Whether the work has already been done is read from the transcript
        rather than from a flag, because the transcript is the only thing that
        really knows. A word that already reads as its replacement was
        corrected on some earlier opening, and correcting it a second time
        would record the correction itself as what the services originally
        said, which is the one piece of evidence that cannot be recovered.
        """
        waiting: dict[str, list[Occurrence]] = {}
        wanted = set(recording_names)
        for occurrence in state.occurrences:
            if not occurrence.auto_applied or occurrence.replacement is None:
                continue
            if occurrence.recording_name not in wanted:
                continue
            waiting.setdefault(occurrence.recording_name, []).append(occurrence)

        applied = 0
        problems: list[str] = []
        # In the order the recordings were named, so that two openings over the
        # same folder do the same things in the same order.
        for name in recording_names:
            outstanding = waiting.get(name)
            if not outstanding:
                continue
            transcript = reader.load(name)
            if transcript is None:
                continue
            changed = False
            for occurrence in outstanding:
                token = next(
                    (item for item in transcript.tokens if item.id == occurrence.token_id),
                    None,
                )
                if token is None or token.text == occurrence.replacement:
                    continue
                transcript = transcript.with_correction(
                    occurrence.token_id, text=occurrence.replacement
                )
                changed = True
                applied += 1
                # The rule's tally is raised here rather than where the answer
                # was decided, because here is where it took effect. A rule
                # credited with a correction that was never written would tell
                # the person it had done work it had not done.
                #
                # Counted through the grouping module rather than by hand,
                # because the Review window writes a rule's correction too and
                # the tally must say the same thing whichever of the two did
                # the work. It said different things until it was made one
                # function, and which answer the person saw depended on a path
                # through the code that nothing on the screen mentions.
                if occurrence.applied_rule_id is not None:
                    grouping.note_rule_applied(state, occurrence.applied_rule_id)
            if changed and not reader.save(name, transcript):
                problems.append(self._unsaved_corrections(name, reader))
        return applied, problems

    def _unsaved_corrections(self, recording_name: str, reader: _TranscriptReader) -> str:
        """How to say that a rule's correction did not reach the disk.

        This is the worst outcome available in the whole opening sequence,
        because the project file is about to record the correction as having
        happened. So it becomes a sentence the person hears rather than a line
        in a log nobody opens.
        """
        path = reader.transcript_path(recording_name)
        where = "its transcript file" if path is None else str(path)
        return f"The automatic corrections to {recording_name} could not be saved to {where}."

    def _review_opening_message(
        self,
        folder: Path,
        recordings: int,
        corrected: int,
        damaged: list[str],
        land_on: str | None,
        landed: bool,
        problems: list[str],
    ) -> str:
        """What to say about a review that has just opened.

        Every part of this is something the person cannot see for themselves
        and would have to be told. The automatic corrections matter most: words
        have been rewritten in files nobody has opened, and a change made on
        somebody's behalf that is never mentioned is indistinguishable from a
        transcript that was always wrong.

        This is the whole of what is said, rather than one of two sentences
        racing each other, so it also has to say where the person has been put
        down. Landing on the recording they asked about is silent now, and a
        review that opened somewhere without saying so would leave somebody
        working out where they are from the first row they hear.
        """
        parts = [f"Reviewing {_count(recordings, 'recording')} in {folder}."]
        if corrected:
            was = "was" if corrected == 1 else "were"
            parts.append(
                f"{_count(corrected, 'word')} {was} corrected automatically from "
                "replacements you have already accepted in this folder."
            )
        if damaged:
            is_are = "is" if len(damaged) == 1 else "are"
            # What was found while opening, not a verdict on the folder.
            # Transcripts are read one at a time as they are wanted, so a file
            # nothing has needed yet has not been looked at, and saying the
            # rest are sound would be a claim nobody has checked.
            parts.append(
                f"{_count(len(damaged), 'transcript')} could not be read and {is_are} "
                f"not in this review: {', '.join(damaged)}."
            )
        if land_on is not None and landed:
            parts.append(f"You are on the first word waiting in {land_on}.")
        elif land_on is not None:
            parts.append(
                f"There is nothing waiting in {land_on}, so the review opens where you "
                "last left it."
            )
        parts.extend(problems)
        return " ".join(parts)

    def _refuse_empty_review(self, damaged: list[str]) -> None:
        """Say that the folder holds nothing to review, and why.

        The wording leads with the project rather than with a file. The person
        did not ask about the recording the highlight happens to be on, so
        telling them that recording has not been transcribed would answer a
        question they did not ask and send them off to transcribe one file when
        the folder needs all of them.
        """
        if damaged:
            message = (
                "There is nothing in this project to review. Every transcript in "
                f"{self._folder} could not be read, so the files may be damaged: "
                f"{', '.join(damaged)}."
            )
        else:
            message = (
                "There is nothing in this project to review. Nothing in "
                f"{self._folder} has been transcribed yet."
            )
        self._set_status(message, alert=True, urgent=True)

    def _close_review_window(self) -> None:
        """Get rid of the review window and its player, in that order.

        Only one review window is kept. Two of them would be two views of one
        project file, each writing over the other's idea of what the person had
        decided, and the loser of that race is a decision that quietly did not
        happen.

        The window is asked to go explicitly rather than being left to the
        garbage collector, which would otherwise keep it, its player and their
        signal connections alive indefinitely. The player is stopped before the
        window is closed, because deletion only takes effect the next time the
        event loop runs and the audio would otherwise carry on playing until
        it did.
        """
        window = self._review_window
        player = self._review_player
        self._review_window = None
        self._review_player = None
        if player is not None:
            player.stop()
        if window is not None:
            window.close()
            # The player is a child of the window, so it goes with it.
            window.deleteLater()

    def _transcript_store(self, recording: Path) -> TranscriptStore:
        """The transcript folder beside one recording, named as the settings say."""
        return TranscriptStore(
            recording,
            self._settings.transcription.processing.transcript_folder_suffix,
        )

    def _save_correction(
        self,
        reader: _TranscriptReader,
        recording_name: str,
        transcript: Transcript,
    ) -> bool:
        """Write a corrected transcript back, and answer whether that worked.

        A correction the person made and believes is saved, but is not, is the
        one failure here that must never pass quietly, because they will only
        find out about it after closing the window that still held the work.

        The answer goes back to the review window, and it is the review window
        that says so out loud. This window used to be the only one told, and it
        announced the failure from its own status bar, which is behind the
        review window the person is working in. The review window meanwhile
        went on as if the save had worked: it kept the change, marked the word
        replaced in the project and announced success. When the person moved
        away and back, the old text came back from the file, and the project no
        longer flagged the word, so nothing would ever bring it back to them.

        The reason is still written into this window's status bar, with the
        path, so it is there to be read later. It is not announced from here
        as well: two urgent announcements raised one after the other cut
        across each other, and the one from the window in front is the one
        that matters.

        It goes through the same reader the review is reading from, so the
        corrected transcript is the one in hand afterwards. Saving straight to
        the file instead would leave the reader holding the version from before
        the correction, and the next glance at that recording would show the
        person the word they had just changed.

        A recording this review does not know is reported rather than turned
        into a path, because guessing a path from a name is how a correction
        ends up in the wrong folder.
        """
        path = reader.transcript_path(recording_name)
        if path is None:
            self._set_status(
                f"The correction to {recording_name} could not be saved, because that "
                "recording is not one of the ones being reviewed."
            )
            return False
        if not reader.save(recording_name, transcript):
            self._set_status(f"The correction could not be saved to {path}.")
            return False
        return True

    def _summary_text(self) -> str:
        total = self._model.rowCount()
        checked = len(self._model.checked_names())
        if total == 0:
            return "No audio files"
        files = "1 file" if total == 1 else f"{total} files"
        return f"{files}, {checked} checked"

    def _update_summary(self) -> None:
        self._summary_label.setText(self._summary_text())

    # -- Status ---------------------------------------------------------

    def _set_status(self, message: str, alert: bool = False, urgent: bool = False) -> None:
        """Show a message in the status bar, and read it out if it matters."""
        self._status_label.setText(message)
        if alert:
            announce(self._status_label, message, urgent=urgent)

    # -- Panels and focus -----------------------------------------------

    def focus_next_panel(self) -> None:
        """Move the focus to the next panel of the window, and say which one.

        Each panel names the control the focus should land on. Calling
        setFocus on the panel itself is no good: a plain container keeps the
        focus and answers no keys, while a group box passes it to a child
        that is not always the first one.
        """
        panels: list[tuple[str, QWidget, object]] = [
            ("Audio folder", self._folder_panel, self._folder_panel.focus_browse_button),
            ("Audio files", self._table, self._focus_file_table),
            ("Audio player", self._player_panel, self._player_panel.focus_play_button),
            ("Selected file", self._info_panel, self._info_panel.focus_first_field),
        ]
        focused = self.focusWidget()
        current_index = -1
        for index, (_name, panel, _focus) in enumerate(panels):
            if focused is not None and (focused is panel or panel.isAncestorOf(focused)):
                current_index = index
                break
        # A panel whose controls are all switched off answers False, and the
        # next one along is tried, so the name read out is always the panel
        # the focus actually reached.
        for step in range(1, len(panels) + 1):
            name, _panel, focus = panels[(current_index + step) % len(panels)]
            if focus() is not False:
                self._set_status(name, alert=True)
                return

    def _focus_file_table(self) -> None:
        self._table.setFocus(Qt.FocusReason.TabFocusReason)

    # -- Settings and the files the application keeps ---------------------

    @property
    def settings(self) -> Settings:
        return self._settings

    def show_settings(self) -> None:
        """Open the Settings dialog and take what the user chose.

        The vocabulary and the service statistics are handed in as well as the
        settings, and the vocabulary is taken back out again. Without that the
        Vocabulary page edits a blank list that is thrown away when the dialog
        closes, which is worse than not offering the page at all: the user
        types in a client's whole list of surnames, presses OK, and is given no
        reason to think anything went wrong.

        The statistics only go in. Nothing on that page changes them; they are
        a record of what the services have actually done, built up by the
        review window one correction at a time.
        """
        dialog = SettingsDialog(
            self._settings,
            self,
            vocabulary=self._vocabulary_store.load(),
            statistics=self._calibration_store.load(),
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.apply_settings(dialog.chosen_settings())
        self.save_vocabulary(dialog.chosen_vocabulary())

    def save_vocabulary(self, vocabulary: Vocabulary) -> None:
        """Keep the words the user has taught the application.

        A failure is announced rather than logged. The settings are saved in
        the same breath and report their own failure, and a user who was told
        their settings were saved would otherwise reasonably assume their
        vocabulary was too.
        """
        if not self._vocabulary_store.save(vocabulary):
            self._set_status(
                f"The vocabulary could not be saved to {self._vocabulary_store.path}.",
                alert=True,
                urgent=True,
            )

    def apply_settings(self, settings: Settings) -> None:
        """Put new settings to work everywhere that uses them, and save them."""
        self._settings = settings
        self._player_panel.apply_settings(settings)
        for action, interval, direction in self._skip_actions:
            action.setText(skip_button_name(direction * settings.skip_seconds[interval]))
        if self._settings_store.save(settings):
            self._set_status("Settings saved.", alert=True)
        else:
            self._set_status(
                f"The settings could not be saved to {self._settings_store.path}.",
                alert=True,
                urgent=True,
            )

    def open_log_file(self) -> None:
        """Open the log file in whatever the user reads text files with."""
        path = log_file_path()
        if not path.is_file():
            self._set_status(
                f"There is no log file yet. It will appear at {path}.",
                alert=True,
                urgent=True,
            )
            return
        self._open_with_windows(path, "log file")

    def open_settings_folder(self) -> None:
        """Open the folder holding the settings, the session and the log."""
        self._open_with_windows(self._settings_store.path.parent, "settings folder")

    def _open_with_windows(self, path: Path, description: str) -> None:
        if QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            self._set_status(f"Opened the {description}, {path}.", alert=True)
        else:
            self._set_status(
                f"Windows could not open the {description}, {path}.",
                alert=True,
                urgent=True,
            )

    def _show_keyboard_shortcuts(self) -> None:
        dialog = KeyboardShortcutsDialog(self._settings, self)
        dialog.exec()
        # Qt returns the focus to whatever had it before the dialog opened,
        # which is what a keyboard user expects.

    # -- Session --------------------------------------------------------

    def _restore_window_layout(self) -> None:
        geometry = _decode(self._session.window_geometry)
        if geometry is not None:
            self.restoreGeometry(geometry)
        splitter_state = _decode(self._session.splitter_state)
        self._needs_initial_split = splitter_state is None
        if splitter_state is not None:
            self._splitter.restoreState(splitter_state)

    def showEvent(self, event) -> None:
        """Split the window down the middle the first time it appears.

        The divider can only be placed sensibly once the window has a real
        width, which it does not have while it is still being built.
        """
        super().showEvent(event)
        if self._needs_initial_split:
            self._needs_initial_split = False
            width = self._splitter.width()
            if width > 0:
                left = max(self._splitter.widget(0).minimumWidth(), width // 2)
                self._splitter.setSizes([left, max(1, width - left)])

    def _schedule_save(self) -> None:
        self._save_timer.start()

    def _save_session(self) -> None:
        # A run that never opened a folder keeps the one already remembered
        # rather than writing an empty one over it. That happens whenever
        # reopening the last folder is switched off: the folder is not
        # opened, so there is nothing here to save, and saving nothing would
        # throw away the very folder the setting is meant to leave alone.
        folder = str(self._folder) if self._folder is not None else self._session.folder
        if self._file_list_is_current:
            checked = self._model.checked_names()
            selected = self._selected_file_name()
        else:
            # The file list has not been read this run, so it says nothing
            # about which files were checked. Saving it as empty would throw
            # that away. This happens when reopening the last folder is
            # switched off, and when the folder is on a drive that is not
            # connected: in both cases the checked files are still there,
            # waiting for the folder to come back.
            checked = self._session.checked_files
            selected = self._session.selected_file
        state = SessionState(
            folder=folder,
            checked_files=checked,
            selected_file=selected,
            window_geometry=_encode(self.saveGeometry()),
            splitter_state=_encode(self._splitter.saveState()),
        )
        self._session = state
        if not self._session_store.save(state):
            _log.warning("The session could not be saved.")

    def _let_the_window_breathe_unless_closing(self) -> None:
        """Give the event loop a turn between transcripts, and stop if it closed us.

        The turn is what lets a close arrive at all: the window's close event
        runs inside it. Once it has, reading on would be minutes of work
        towards a window that must not appear, so the opening is abandoned
        at the next transcript boundary.
        """
        _let_the_window_breathe()
        if self._closing:
            raise _WindowClosing()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._closing = True
        self._save_timer.stop()
        self._media_load_timer.stop()
        self._scanner.stop()
        self._player.stop()
        # The review window is a separate window rather than a dialog, so
        # closing this one does not close it. Left open it would keep the
        # application running with no main window, with a player of its own
        # still holding a file open.
        self._close_review_window()
        self._save_session()
        super().closeEvent(event)


class _WindowClosing(Exception):
    """Raised inside an opening review when the main window closes under it."""


class _TranscriptReader:
    """Reads a folder's transcripts one at a time, and never more than one.

    This exists because of a measurement. Handing the Review window every
    transcript of a folder at once was tried first, and on fifty hour-long
    recordings it took 37 seconds and held about 1.6 GB of live objects, every
    time somebody pressed Ctrl+R. A transcript carries every service's answer
    for every word and every candidate that was weighed, so tens of megabytes
    an hour is what it genuinely costs; the mistake was reading fifty of them
    to show a list the project file already describes.

    So nothing here reads anything until it is asked for a particular
    recording, and the whole point is lost the moment somebody finds that
    awkward and reads them all into a dictionary in advance. Whatever it is
    spelled like, that is the 1.6 GB coming back.

    Exactly one transcript is held, the one last read. That is not a cache in
    any interesting sense and it is not meant to grow into one: holding two
    doubles the peak, and holding the folder is where this started. What the
    one buys is the inner loop of reviewing, where a person moves between
    several occurrences of the same recording in a row and would otherwise
    re-read tens of megabytes on every arrow key.

    A transcript that cannot be read is reported rather than hidden, but only
    once and only when it is actually reached, because under this arrangement
    nobody knows a file is damaged until they try to read it. The name is kept
    in :attr:`damaged` for whoever is composing a message, and ``on_damaged``
    is called so that damage found long after the window opened can still be
    said out loud.
    """

    def __init__(
        self,
        stores: dict[str, TranscriptStore],
        on_damaged=None,
        between_recordings: Callable[[], None] | None = None,
    ) -> None:
        self._stores = stores
        self.on_damaged = on_damaged
        # Called just before each transcript is actually parsed, and only
        # then: a transcript answered from the one in hand costs nothing and
        # is the inner loop of reviewing. What the opening sequence passes
        # here lets the window repaint and a screen reader speak between one
        # recording's tens of megabytes and the next's.
        self.between_recordings = between_recordings
        self._name: str | None = None
        self._transcript: Transcript | None = None
        self._damaged: list[str] = []

    @property
    def damaged(self) -> list[str]:
        """The recordings whose transcripts turned out not to be readable.

        This is what has been found so far and nothing more, because a file is
        only known to be damaged once something has tried to read it. It must
        never be read as "and the rest are fine".
        """
        return list(self._damaged)

    def load(self, recording_name: str) -> Transcript | None:
        """The transcript of one recording, or ``None`` if it cannot be read."""
        if recording_name == self._name:
            return self._transcript
        # The one in hand is let go before the next is read rather than
        # after, so that two transcripts are never held at once. A three-hour
        # recording parses to something like 90 MB, and the peak is what
        # decides whether a folder of them fits.
        self._name = None
        self._transcript = None
        if self.between_recordings is not None:
            self.between_recordings()
        store = self._stores.get(recording_name)
        transcript = None if store is None else store.load()
        if transcript is None and recording_name not in self._damaged:
            self._damaged.append(recording_name)
            if self.on_damaged is not None:
                self.on_damaged(recording_name)
        self._name = recording_name
        self._transcript = transcript
        return transcript

    def save(self, recording_name: str, transcript: Transcript) -> bool:
        """Write a corrected transcript back, and hold it as the one in hand.

        Holding it matters as much as writing it. Whoever reads this recording
        next has to see the correction, and a reader that answered from the
        disk while the corrected copy sat in memory would show a person the
        word they had just changed.
        """
        store = self._stores.get(recording_name)
        if store is None or not store.save(transcript):
            return False
        self._name = recording_name
        self._transcript = transcript
        return True

    def write_exports(self, recording_name: str, transcript: Transcript) -> list[str]:
        """Write the readable files again from a saved transcript.

        Answers the names of the files that could not be written. A recording
        this reader does not know has nowhere to write to, so both are named.
        It writes once without waiting, because the review window calls it
        after every correction and waiting would freeze the window.
        """
        store = self._stores.get(recording_name)
        if store is None:
            return [TEXT_EXPORT_NAME, REPORT_EXPORT_NAME]
        return write_exports(transcript, store, patient=False)

    def transcript_path(self, recording_name: str) -> Path | None:
        store = self._stores.get(recording_name)
        return None if store is None else store.transcript_path


def _let_the_window_breathe() -> None:
    """Let the window repaint and the screen reader speak, then carry on.

    Reading a folder's transcripts is done on the thread that draws the
    window, one recording at a time, and each one can take seconds. Without
    this the window is marked "not responding" part way through and the
    sentence announced at the start is the last thing a screen reader says
    until the end. Called between recordings only, never inside one.
    """
    application = QApplication.instance()
    if application is not None:
        application.processEvents()


def _without_speaker_stretches(state: ProjectState) -> set[str]:
    """The recordings whose speaker doubts were saved in the old way.

    A project analysed before speaker doubts had their own list saved each
    doubted word as a flagged word. Those words now leave the word list as
    soon as the project is opened, but a recording whose transcript has not
    changed is not read again, so its stretches would never be found and the
    doubts would vanish from the window altogether. Such a recording has
    flagged words whose only reason is the speaker, and no saved stretch, so
    it is read once more. That reading saves its stretches, and the next
    opening reads nothing.
    """
    speaker_only = {reason.value for reason in SPEAKER_ONLY_REASONS}
    old_style = {
        item.recording_name
        for item in state.flagged
        if not item.settled and item.reasons and set(item.reasons) <= speaker_only
    }
    with_stretches = {item.recording_name for item in state.speaker_doubts}
    return old_style - with_stretches


def _flagged_after(
    remembered: list[_PerRecording],
    analysed: list[str],
    gathered: dict[str, list[_PerRecording]],
) -> list[_PerRecording]:
    """The project's flagged words after one analysis, kept honest both ways.

    The stretches whose speaker is in doubt are kept by the same rule, and
    pass through here too.

    A recording that was read is described entirely by what it was just found
    to say. Its old entries are dropped rather than merged with the new ones,
    and that is the whole answer to a word that has gone from a regenerated
    transcript: it is not carried forward, so it can never become a row
    pointing at a word nobody can play or correct. Merging would keep it for
    ever, because nothing later would ever have a reason to look for it again.

    A recording that was not read keeps exactly what the project already knew
    about it. That covers two quite different cases and is right for both. One
    was skipped because nothing about it can have changed since it was last
    analysed, so what is remembered is current. The other could not be read at
    all just now -- a file being written by another program, a disconnected
    drive -- and throwing its words away would tell the person a recording had
    nothing wrong in it when the truth is that nobody could look. That is the
    same rule the analysis itself follows for the occurrences of a recording it
    could not read, and for the same reason.

    It does not cover a recording that has left the folder, and it must not be
    asked to. That case is settled by the analysis, which is the only place
    that has been told what the folder now holds, so what arrives here has
    already had the deleted recordings taken out of it. ``remembered`` is
    therefore the analysis's own list rather than a copy taken before it ran.
    """
    read = set(gathered)
    kept = [item for item in remembered if item.recording_name not in read]
    # In the order the recordings were analysed, so that two openings over one
    # folder leave the file in the same order and it does not churn.
    fresh = [item for name in analysed for item in gathered.get(name, ())]
    return kept + fresh


def _transcript_times(
    recording_names: list[str],
    stores: dict[str, TranscriptStore],
) -> dict[str, int]:
    """The mark of each of these transcript files, as :func:`_transcript_mark` gives it.

    The number is never turned into a date or read for meaning. Its only use
    is being compared with the same file's mark on a later opening, and for
    that the exact number matters and its meaning does not.

    A file that cannot be asked leaves no entry, and neither does a recording
    with no store behind it. Both come back to the caller as "nothing is known
    about that one", which is read as a reason to look at it again rather than
    as a reason to trust anything.
    """
    times: dict[str, int] = {}
    for name in recording_names:
        store = stores.get(name)
        if store is None:
            continue
        try:
            times[name] = _transcript_mark(store.transcript_path)
        except OSError:
            continue
    return times


def _transcript_mark(path: Path) -> int:
    """One number that changes whenever this transcript file is written again.

    The modification time alone is not enough, and the gap is easy to hit. On
    Windows a file's time comes from a clock that moves in steps of about
    fifteen milliseconds, so two saves of one transcript inside a single step
    leave it with exactly the same time. Measured on Windows 11, more than half
    of a run of back-to-back saves did. A correction written straight after an
    analysis is that case, and the recording would never be read again.

    So the size and the file's identity go in as well. Every transcript is
    written to a temporary file that then takes the old one's place, which
    gives it a new file ID on every save, so a rewrite changes the mark even
    when its time and its size do not. A filesystem that has no file IDs
    reports 0 for every file, and the mark then rests on the time and the size.

    The three are folded into one whole number so that the project file keeps
    the shape it always had. A project written before the size and the ID were
    part of it holds bare times, which never match, so each recording in it is
    read once more and noted afresh.

    Raises :class:`OSError` when the file cannot be asked about.
    """
    facts = path.stat()
    key = f"{facts.st_mtime_ns}:{facts.st_size}:{facts.st_ino}".encode("ascii")
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")


def _times_after(
    remembered: dict[str, int],
    present: list[str],
    taken: dict[str, int],
    damaged: set[str],
) -> dict[str, int]:
    """The transcript ages the project should hold after one analysis.

    Three things happen here, and each is what keeps one kind of mistake out
    of the project file.

    A recording that has left the folder loses its entry, so this stays a note
    about the transcripts that are there rather than a growing record of files
    nobody has any more.

    A recording that was read this time is described by the age noted before
    it was read. A recording that was not read keeps whatever was noted about
    it before, which is still true of it, because not being read is precisely
    what it means for a file not to have changed.

    A recording that could not be read loses its entry altogether, even one
    that had a perfectly good entry a moment ago. It was in this run's list
    because something about it looked new, and nobody has managed to look at
    it yet, so the next opening must try again. Leaving the old note in place
    would say the file had been analysed as it now stands, and the recording
    would be passed over from then on.
    """
    kept = {name: age for name, age in remembered.items() if name in present}
    kept.update({name: age for name, age in taken.items() if name not in damaged})
    for name in damaged:
        kept.pop(name, None)
    return kept


def _count(number: int, noun: str) -> str:
    """Say "1 recording" or "3 recordings", so a count reads as English.

    A sentence that reads as though it were assembled by a machine invites
    the listener to distrust the number in it, and these numbers are read out
    to somebody who cannot see the window they describe.
    """
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _encode(data: QByteArray) -> str | None:
    """Turn Qt layout data into text that can live in a JSON file."""
    if data is None or data.isEmpty():
        return None
    return bytes(data.toBase64()).decode("ascii")


def _decode(text: str | None) -> QByteArray | None:
    if not text:
        return None
    try:
        return QByteArray.fromBase64(QByteArray(text.encode("ascii")))
    except (UnicodeEncodeError, ValueError):
        _log.debug("Ignoring unreadable saved window layout.")
        return None
