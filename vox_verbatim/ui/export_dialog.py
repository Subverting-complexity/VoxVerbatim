"""The Export Transcripts dialog: which files, and which folder.

It only asks. The export flow works out what to write, asks about files
already in the folder, and writes them, because those steps need the
recordings and the transcript folders, which this dialog knows nothing of;
see :mod:`vox_verbatim.ui.export_flow`. The one thing it is told about the
recordings is which smooth transcripts are out of date, so it can ask what
to do with them in the same place as everything else.

The dialog refuses to close on Export with no file kind ticked or with no
usable folder. It says why in a line under the controls, reads that line
out, and puts the focus on the control to fix.
"""

from __future__ import annotations

from collections.abc import Sequence
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
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from vox_verbatim.settings import ExportSettings
from vox_verbatim.transcription.folder_export import ExportKind, names_text
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
        out_of_date_names: Sequence[str] = (),
    ) -> None:
        """``scope_text`` says what will be exported. The main window's wording,
        about checked and highlighted files, is the default; the review window
        names its one recording instead.

        ``out_of_date_names`` are the recordings whose smooth transcript is
        older than their corrections. When there are any, the dialog names
        them and asks whether to make them again first or export them as
        they are; Cancel is the third answer.
        """
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

        self._out_of_date_names = list(out_of_date_names)
        self._out_of_date_group: QGroupBox | None = None
        self._remake_radio: QRadioButton | None = None
        self._as_is_radio: QRadioButton | None = None
        if self._out_of_date_names:
            self._build_out_of_date_choice(layout)

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
        if self._remake_radio is not None and self._as_is_radio is not None:
            self.setTabOrder(self._report_box, self._remake_radio)
            self.setTabOrder(self._remake_radio, self._as_is_radio)
            self.setTabOrder(self._as_is_radio, self._folder_edit)
        else:
            self.setTabOrder(self._report_box, self._folder_edit)
        self.setTabOrder(self._folder_edit, self._browse_button)
        self._transcript_box.setFocus()

    def _build_out_of_date_choice(self, layout: QVBoxLayout) -> None:
        """Name the out-of-date smooth transcripts, and ask what to do with them.

        Two radio buttons rather than a second box after Export, so the whole
        decision is made in one place and can be read through with Tab. The
        names are in the label, and in each button's description as well,
        because a screen reader moving by Tab reads the buttons and not the
        label above them. The group follows the smooth transcript box: it is
        switched off while that box is clear, because the question does not
        arise.
        """
        names = names_text(self._out_of_date_names)
        one = len(self._out_of_date_names) == 1
        them = "it" if one else "them"
        group = QGroupBox(
            f"Smooth {'transcript' if one else 'transcripts'} that {'is' if one else 'are'} "
            "out of date",
            self,
        )
        group_layout = QVBoxLayout(group)
        sentence = (
            f"The smooth {'transcript' if one else 'transcripts'} of {names} "
            f"{'was' if one else 'were'} made before the latest corrections."
        )
        label = QLabel(sentence, group)
        label.setWordWrap(True)
        group_layout.addWidget(label)
        self._remake_radio = QRadioButton(f"Make {them} a&gain first", group)
        describe(
            self._remake_radio,
            f"Make {them} again first",
            f"{sentence} Make {them} again with the language model, with every "
            "correction, and then export. This can take a minute or more for each.",
        )
        self._as_is_radio = QRadioButton(f"E&xport {them} as {'it is' if one else 'they are'}", group)
        describe(
            self._as_is_radio,
            f"Export {them} as {'it is' if one else 'they are'}",
            f"{sentence} Export {them} without the latest corrections.",
        )
        self._remake_radio.setChecked(True)
        group_layout.addWidget(self._remake_radio)
        group_layout.addWidget(self._as_is_radio)
        layout.addWidget(group)
        self._out_of_date_group = group
        # The smooth box says so too, because it is where a person deciding
        # whether to export the smooth transcript is.
        describe(
            self._smooth_box,
            "Smooth transcript",
            "The copy edited for easy reading. A recording that has none is skipped. "
            f"{sentence} The choice below says what to do with "
            f"{'it' if one else 'them'}.",
        )
        group.setEnabled(self._smooth_box.isChecked())
        self._smooth_box.toggled.connect(group.setEnabled)

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

    def chosen_remake(self) -> bool:
        """Whether to make the out-of-date smooth transcripts again first.

        False when there are none, or the smooth transcript is not being
        exported, or the person chose to export them as they are.
        """
        return (
            self._remake_radio is not None
            and self._smooth_box.isChecked()
            and self._remake_radio.isChecked()
        )

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
