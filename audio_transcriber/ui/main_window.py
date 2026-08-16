"""The main window, which ties the folder, the file list and the player together."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QByteArray, QModelIndex, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QCloseEvent, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QMainWindow,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from audio_transcriber import APPLICATION_NAME
from audio_transcriber.audio.library import AudioFile, DurationState
from audio_transcriber.audio.player import AudioPlayer
from audio_transcriber.audio.scanner import FolderScanner
from audio_transcriber.paths import log_file_path
from audio_transcriber.session import SessionState, SessionStore
from audio_transcriber.settings import SETTINGS_FILE_NAME, Settings, SettingsStore
from audio_transcriber.ui.accessibility import announce, describe
from audio_transcriber.ui.file_info_panel import FileInfoPanel
from audio_transcriber.ui.file_table import AudioFileTableModel, AudioFileTableView
from audio_transcriber.ui.folder_panel import FolderPanel
from audio_transcriber.ui.help_dialogs import KeyboardShortcutsDialog, show_about
from audio_transcriber.ui.settings_dialog import SettingsDialog
from audio_transcriber.ui.player_panel import (
    STATUS_FINISHED,
    STATUS_PAUSED,
    STATUS_PLAYING,
    PlayerPanel,
    skip_button_name,
)

_log = logging.getLogger(__name__)

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
            self._folder_panel.browse,
        )
        self._refresh_action = self._add_action(
            file_menu, "&Refresh File List", QKeySequence(Qt.Key.Key_F5), self.refresh
        )
        file_menu.addSeparator()
        self._add_action(
            file_menu,
            "&Settings...",
            QKeySequence.StandardKey.Preferences,
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

    def _on_folder_chosen(self, folder_text: str) -> None:
        folder = Path(folder_text)
        self._folder_panel.set_folder(folder)
        self._load_folder(folder)
        self._table.setFocus()

    def refresh(self) -> None:
        """Read the current folder again, keeping the checked and selected files."""
        if self._folder is None:
            self._set_status("There is no folder to refresh.", alert=True)
            return
        self._load_folder(
            self._folder,
            checked=self._model.checked_names(),
            selected=self._selected_file_name(),
        )

    def _load_folder(
        self,
        folder: Path,
        checked: list[str] | None = None,
        selected: str | None = None,
    ) -> None:
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
        name, _panel, focus = panels[(current_index + 1) % len(panels)]
        focus()
        self._set_status(name, alert=True)

    def _focus_file_table(self) -> None:
        self._table.setFocus(Qt.FocusReason.TabFocusReason)

    # -- Settings and the files the application keeps ---------------------

    @property
    def settings(self) -> Settings:
        return self._settings

    def show_settings(self) -> None:
        """Open the Settings dialog and take what the user chose."""
        dialog = SettingsDialog(self._settings, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.apply_settings(dialog.chosen_settings())

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

    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_timer.stop()
        self._media_load_timer.stop()
        self._scanner.stop()
        self._player.stop()
        self._save_session()
        super().closeEvent(event)


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
