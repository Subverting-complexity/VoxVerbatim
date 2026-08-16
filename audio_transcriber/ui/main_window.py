"""The main window, which ties the folder, the file list and the player together."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QByteArray, QModelIndex, Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
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
from audio_transcriber.session import SessionState, SessionStore
from audio_transcriber.ui.accessibility import announce, describe
from audio_transcriber.ui.file_info_panel import FileInfoPanel
from audio_transcriber.ui.file_table import AudioFileTableModel, AudioFileTableView
from audio_transcriber.ui.folder_panel import FolderPanel
from audio_transcriber.ui.help_dialogs import KeyboardShortcutsDialog, show_about
from audio_transcriber.ui.player_panel import PlayerPanel

_log = logging.getLogger(__name__)

#: Moving through the file list with the arrow keys should not load a new
#: file into the player on every key press, so loading waits for the
#: highlight to settle for this long.
_MEDIA_LOAD_DELAY_MS = 250

#: Saving the session is cheap but pointless to repeat on every key press.
_SESSION_SAVE_DELAY_MS = 750


class MainWindow(QMainWindow):
    """The one window of the application."""

    def __init__(self, session_store: SessionStore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session_store = session_store
        self._session = session_store.load()
        self._folder: Path | None = None
        self._pending_checked: list[str] = []
        self._pending_selected: str | None = None
        self._needs_initial_split = True

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

        self._summary_label = QLabel(self)
        describe(self._summary_label, "File list summary")

        left = QWidget(self)
        # Wide enough that the three columns of the file list are readable
        # without scrolling sideways, even before the user resizes anything.
        left.setMinimumWidth(360)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 4, 8)
        left_layout.addWidget(self._folder_panel)
        left_layout.addWidget(self._table_label)
        left_layout.addWidget(self._table, 1)
        left_layout.addWidget(self._summary_label)

        self._player_panel = PlayerPanel(self._player, self)
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
        describe(self._status_label, "Status")
        status_bar = QStatusBar(self)
        status_bar.addWidget(self._status_label, 1)
        status_bar.setSizeGripEnabled(True)
        self.setStatusBar(status_bar)

        self.resize(1120, 720)
        self._update_summary()

    def _build_menus(self) -> None:
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
        self._add_action(file_menu, "E&xit", QKeySequence("Ctrl+Q"), self.close)

        playback_menu = menu_bar.addMenu("&Playback")
        self._add_action(
            playback_menu,
            "Play or &Pause",
            QKeySequence("Ctrl+Space"),
            self._player.toggle_play_pause,
        )
        playback_menu.addSeparator()
        for label, shortcut, delta in (
            ("Back 15 &seconds", "Alt+Left", -15),
            ("Back &2 minutes", "Alt+Shift+Left", -120),
            ("Back &5 minutes", "Alt+Ctrl+Left", -300),
        ):
            self._add_skip_action(playback_menu, label, shortcut, delta)
        playback_menu.addSeparator()
        for label, shortcut, delta in (
            ("Forward 15 se&conds", "Alt+Right", 15),
            ("Forward 2 &minutes", "Alt+Shift+Right", 120),
            ("Forward 5 m&inutes", "Alt+Ctrl+Right", 300),
        ):
            self._add_skip_action(playback_menu, label, shortcut, delta)

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

    def _add_skip_action(self, menu, text: str, shortcut: str, delta_seconds: int) -> QAction:
        return self._add_action(
            menu,
            text,
            QKeySequence(shortcut),
            lambda _checked=False, d=delta_seconds: self._player.skip(d * 1000),
        )

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

    # -- Folders and files ----------------------------------------------

    def _restore_session_folder(self) -> None:
        """Reopen the folder from the last run, if it is still there."""
        folder_text = self._session.folder
        if not folder_text:
            self._set_status("Select a folder to begin.")
            return
        folder = Path(folder_text)
        self._folder_panel.set_folder(folder)
        if not folder.is_dir():
            self._folder = folder
            self._set_status(
                f"The folder {folder} is not available. It may be on a drive that is "
                "not connected.",
                alert=True,
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
        self._scanner.start(folder)
        self._schedule_save()

    def _on_files_found(self, files: list[AudioFile], scan_id: int) -> None:
        if scan_id != self._scanner.current_scan_id:
            return
        self._model.set_files(files)
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
        self._set_status(message, alert=True)

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
        audio_file = self._model.file_at(self._table.selected_row())
        if audio_file is None:
            self._player.load(None)
            return
        if self._player.source_path == audio_file.path:
            return
        self._player.load(audio_file.path)

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
        self._set_status(message, alert=True)

    def _on_playback_status_changed(self, status: str) -> None:
        self._set_status(status, alert=True)

    # -- Checked files --------------------------------------------------

    def _on_file_checked(self, name: str, checked: bool) -> None:
        # No alert here: a screen reader announces the new state of the
        # check box itself, and saying it twice is worse than saying it once.
        state = "checked" if checked else "cleared"
        self._set_status(f"{name} {state}. {self._summary_text()}.")

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

    def _set_status(self, message: str, alert: bool = False) -> None:
        """Show a message in the status bar, and read it out if it matters."""
        self._status_label.setText(message)
        if alert:
            announce(self._status_label, message)

    # -- Panels and focus -----------------------------------------------

    def focus_next_panel(self) -> None:
        """Move the focus to the next panel of the window, and say which one."""
        panels: list[tuple[str, QWidget]] = [
            ("Audio folder", self._folder_panel),
            ("Audio files", self._table),
            ("Audio player", self._player_panel),
            ("Selected file", self._info_panel),
        ]
        focused = self.focusWidget()
        current_index = -1
        for index, (_name, widget) in enumerate(panels):
            if focused is not None and (focused is widget or widget.isAncestorOf(focused)):
                current_index = index
                break
        name, widget = panels[(current_index + 1) % len(panels)]
        widget.setFocus(Qt.FocusReason.TabFocusReason)
        if not widget.hasFocus():
            widget.focusNextChild()
        self._set_status(name, alert=True)

    def _show_keyboard_shortcuts(self) -> None:
        dialog = KeyboardShortcutsDialog(self)
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
        state = SessionState(
            folder=str(self._folder) if self._folder is not None else None,
            checked_files=self._model.checked_names(),
            selected_file=self._selected_file_name(),
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
