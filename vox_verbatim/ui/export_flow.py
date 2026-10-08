"""The steps of one export, shared by the main window and the review window.

Both windows offer Export Transcripts. The main window exports the checked
recordings, or the highlighted one; the review window exports the recording
of the selected occurrence. Everything after that choice is the same: the
dialog, remembering what was chosen, asking once about files already in the
folder, writing, and the summary with its button to open the folder. It
lives here so the two windows cannot drift apart.

Each export is one :class:`ExportFlow`, made for the window that asked. The
window is the parent of every box the flow shows, so a modal box sits in
front of the window the person is working in, and Qt gives the focus back to
that window when the box closes. The flow also puts the focus back on the
control that had it, because a run of two or three boxes one after another
can otherwise leave it on the window itself, where a screen reader says
nothing useful.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import shiboken6
from PySide6.QtCore import QObject, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QDialog, QMessageBox, QWidget

from vox_verbatim.settings import ExportSettings
from vox_verbatim.transcription import folder_export
from vox_verbatim.transcription.folder_export import StoreFor
from vox_verbatim.ui.export_dialog import ExportDialog

_log = logging.getLogger(__name__)

#: The key of Export Transcripts, the same in both windows. Neither window
#: uses it for anything else.
EXPORT_KEY = "Ctrl+Shift+E"

#: Shows a sentence in the asking window's status line, and reads it out
#: when ``alert`` is True. The same shape as each window's ``_set_status``.
Say = Callable[..., None]

#: What the review window is given to export with: the recordings, the
#: window the dialog opens in front of, how to say what happened, and the
#: sentence at the top of the dialog. The main window's export_recordings.
ExportRecordings = Callable[[list[Path], QWidget, Say, str | None], None]


class ExportFlow(QObject):
    """One export, from the dialog to the summary.

    ``settings`` answers the export settings as they are now, and
    ``save_settings`` keeps what the person chose, so the dialog opens the
    same way next time from either window. ``store_for`` gives the transcript
    folder of a recording.

    The three boxes are methods a test replaces: the question about files
    already in the folder, the summary, and opening the folder.
    """

    def __init__(
        self,
        parent: QWidget,
        say: Say,
        settings: Callable[[], ExportSettings],
        save_settings: Callable[[ExportSettings], None],
        store_for: StoreFor,
    ) -> None:
        super().__init__(parent)
        self._window = parent
        self._say = say
        self._settings = settings
        self._save_settings = save_settings
        self._store_for = store_for
        self._focus: QWidget | None = None

    def run(self, recordings: Sequence[Path], scope_text: str | None = None) -> None:
        """Export ``recordings``, asking the person how first.

        ``scope_text`` is the sentence at the top of the dialog saying what
        will be exported; nothing given means the main window's wording.
        """
        self._remember_focus()
        try:
            self._run(list(recordings), scope_text)
        finally:
            self._restore_focus()
            self.deleteLater()

    def _run(self, recordings: list[Path], scope_text: str | None) -> None:
        dialog = ExportDialog(
            len(recordings), self._settings(), self._window, scope_text=scope_text
        )
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        chosen = dialog.chosen_settings()
        kinds = dialog.chosen_kinds()
        dialog.deleteLater()
        self._save_settings(chosen)
        if not accepted:
            self._say("Export Transcripts was closed without exporting anything.")
            return
        folder = Path(chosen.folder)
        plan = folder_export.plan_export(recordings, folder, kinds, self._store_for)
        replace_existing = True
        existing = plan.existing()
        if existing:
            answer = self._ask_about_existing_exports(existing, folder)
            if answer == "cancel":
                self._say("The export was cancelled. Nothing was written.", alert=True)
                return
            replace_existing = answer == "replace"
        result = folder_export.run_export(plan, self._store_for, replace_existing)
        message = folder_export.summary_text(result)
        self._say(message)
        if self._show_export_summary(message):
            self._open_folder(folder)

    # -- Focus ---------------------------------------------------------------

    def _remember_focus(self) -> None:
        """Note the control that has the focus in the asking window.

        The window's own record rather than the application's, because the
        application has no focus widget while the window is not active, and
        the window still knows which of its controls had it.
        """
        self._focus = self._window.window().focusWidget()

    def _restore_focus(self) -> None:
        widget = self._focus
        self._focus = None
        if widget is None or not shiboken6.isValid(widget):
            return
        if widget.isVisible() and widget.isEnabled():
            widget.setFocus(Qt.FocusReason.OtherFocusReason)

    # -- The boxes -----------------------------------------------------------

    def _ask_about_existing_exports(self, existing: list[Path], folder: Path) -> str:
        """Ask once what to do with files already in the folder.

        Answers "replace", "skip" or "cancel". A test replaces this.
        """
        count = len(existing)
        names = [path.name for path in existing]
        shown = "\n".join(names[:10])
        if count > 10:
            shown += f"\nand {count - 10} more."
        one = count == 1
        box = QMessageBox(self._window)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Files Already Exist")
        box.setText(
            f"{count} {'file' if one else 'files'} with the same "
            f"{'name' if one else 'names'} already {'exists' if one else 'exist'} "
            f"in {folder}. Replace them, skip them, or cancel the export?"
        )
        box.setInformativeText(shown)
        replace_button = box.addButton("&Replace All", QMessageBox.ButtonRole.YesRole)
        skip_button = box.addButton("&Skip Them", QMessageBox.ButtonRole.NoRole)
        cancel_button = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(cancel_button)
        box.setEscapeButton(cancel_button)
        box.exec()
        clicked = box.clickedButton()
        box.deleteLater()
        if clicked is replace_button:
            return "replace"
        if clicked is skip_button:
            return "skip"
        return "cancel"

    def _show_export_summary(self, message: str) -> bool:
        """Show how the export went, and answer whether to open the folder.

        A message box rather than the status line alone, because it holds the
        button that opens the folder, and a screen reader reads its text when
        it appears. A test replaces this.
        """
        box = QMessageBox(self._window)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Export Finished")
        box.setText(message)
        open_button = box.addButton("&Open Folder", QMessageBox.ButtonRole.ActionRole)
        close_button = box.addButton(QMessageBox.StandardButton.Close)
        box.setDefaultButton(close_button)
        box.setEscapeButton(close_button)
        box.exec()
        opened = box.clickedButton() is open_button
        box.deleteLater()
        return opened

    def _open_folder(self, folder: Path) -> None:
        """Open the export folder in File Explorer. A test replaces this."""
        if QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
            self._say(f"Opened the export folder, {folder}.", alert=True)
        else:
            self._say(
                f"Windows could not open the export folder, {folder}.", alert=True, urgent=True
            )
