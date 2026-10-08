"""The Export Transcripts dialog: which files, and which folder.

It only asks. The export flow works out what to write, asks about files
already in the folder, and writes them, because those steps need the
recordings and the transcript folders, which this dialog knows nothing of;
see :mod:`vox_verbatim.ui.export_flow`.

The dialog refuses to close on Export with no file kind ticked or with no
usable folder. It says why in a line under the controls, reads that line
out, and puts the focus on the control to fix.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from vox_verbatim.settings import ExportSettings
from vox_verbatim.transcription.folder_export import ExportKind
from vox_verbatim.ui.accessibility import announce, describe

NO_KIND_CHOSEN = "Tick at least one kind of file to export."
NO_FOLDER_CHOSEN = "Choose the folder to export to."


class ExportDialog(QDialog):
    """Choose the files to export and the folder to export them to."""

    def __init__(
        self,
        recording_count: int,
        settings: ExportSettings,
        parent: QWidget | None = None,
        scope_text: str | None = None,
    ) -> None:
        """``scope_text`` says what will be exported. The main window's wording,
        about checked and highlighted files, is the default; the review window
        names its one recording instead."""
        super().__init__(parent)
        self.setWindowTitle("Export Transcripts")
        layout = QVBoxLayout(self)

        noun = "recording" if recording_count == 1 else "recordings"
        if scope_text is None:
            scope_text = (
                f"{recording_count} {noun} will be exported: the checked files, or the "
                "highlighted file if none are checked."
            )
        self._count_label = QLabel(scope_text, self)
        self._count_label.setWordWrap(True)
        layout.addWidget(self._count_label)

        kinds = QGroupBox("Files to export", self)
        kinds_layout = QVBoxLayout(kinds)
        self._transcript_box = QCheckBox("&Transcript", kinds)
        describe(
            self._transcript_box,
            "Transcript",
            "The literal transcript, with every correction made in the review window.",
        )
        self._smooth_box = QCheckBox("S&mooth transcript", kinds)
        describe(
            self._smooth_box,
            "Smooth transcript",
            "The copy edited for easy reading. A recording that has none is skipped.",
        )
        self._report_box = QCheckBox("Review &report", kinds)
        describe(
            self._report_box,
            "Review report",
            "The report on how the transcript was made and what was uncertain.",
        )
        for box in (self._transcript_box, self._smooth_box, self._report_box):
            kinds_layout.addWidget(box)
        layout.addWidget(kinds)
        self._transcript_box.setChecked(settings.transcript)
        self._smooth_box.setChecked(settings.smooth_transcript)
        self._report_box.setChecked(settings.review_report)

        folder_label = QLabel("&Folder:", self)
        self._folder_edit = QLineEdit(settings.folder, self)
        folder_label.setBuddy(self._folder_edit)
        describe(
            self._folder_edit,
            "Export folder",
            "The folder the files are written to. Each file is named after its recording.",
        )
        self._browse_button = QPushButton("&Browse...", self)
        describe(self._browse_button, "Browse for the export folder")
        self._browse_button.clicked.connect(self._browse)
        folder_row = QHBoxLayout()
        folder_row.addWidget(folder_label)
        folder_row.addWidget(self._folder_edit, 1)
        folder_row.addWidget(self._browse_button)
        layout.addLayout(folder_row)

        # Carries the reason Export was refused. Not given an accessible
        # name, because a label's text is its name; see describe.
        self._problem_label = QLabel("", self)
        self._problem_label.setWordWrap(True)
        layout.addWidget(self._problem_label)

        self._buttons = QDialogButtonBox(self)
        self._export_button = self._buttons.addButton(
            "&Export", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self._buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self.setTabOrder(self._transcript_box, self._smooth_box)
        self.setTabOrder(self._smooth_box, self._report_box)
        self.setTabOrder(self._report_box, self._folder_edit)
        self.setTabOrder(self._folder_edit, self._browse_button)
        self._transcript_box.setFocus()

    # -- What was chosen ------------------------------------------------

    def chosen_settings(self) -> ExportSettings:
        return ExportSettings(
            folder=self._folder_edit.text().strip(),
            transcript=self._transcript_box.isChecked(),
            smooth_transcript=self._smooth_box.isChecked(),
            review_report=self._report_box.isChecked(),
        )

    def chosen_kinds(self) -> list[ExportKind]:
        chosen = []
        if self._transcript_box.isChecked():
            chosen.append(ExportKind.TRANSCRIPT)
        if self._smooth_box.isChecked():
            chosen.append(ExportKind.SMOOTH_TRANSCRIPT)
        if self._report_box.isChecked():
            chosen.append(ExportKind.REVIEW_REPORT)
        return chosen

    def chosen_folder(self) -> Path:
        return Path(self._folder_edit.text().strip())

    def problem(self) -> str | None:
        """Why Export cannot go ahead, or None when it can."""
        if not self.chosen_kinds():
            return NO_KIND_CHOSEN
        text = self._folder_edit.text().strip()
        if not text:
            return NO_FOLDER_CHOSEN
        if not Path(text).is_dir():
            return f"The folder {text} does not exist. Choose a folder that does."
        return None

    # -- Closing --------------------------------------------------------

    def accept(self) -> None:
        problem = self.problem()
        if problem is None:
            self._problem_label.setText("")
            super().accept()
            return
        self._problem_label.setText(problem)
        if problem == NO_KIND_CHOSEN:
            self._transcript_box.setFocus()
        else:
            self._folder_edit.setFocus()
        announce(self._problem_label, problem, urgent=True)

    # -- Browsing -------------------------------------------------------

    def _browse(self) -> None:
        folder = self._ask_for_folder(self._folder_edit.text().strip())
        if folder:
            self._folder_edit.setText(folder)
        self._folder_edit.setFocus()

    def _ask_for_folder(self, start: str) -> str:
        """Ask Windows for a folder. A test replaces this."""
        return QFileDialog.getExistingDirectory(self, "Export to Folder", start)
