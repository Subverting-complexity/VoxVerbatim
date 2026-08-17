"""The Enhance Audio dialog.

The dialog opens with the files the user picked already listed, asks where
the enhanced copies should go and how loud they should be, and then runs
the work in the background while showing how far it has come.

Three things about it matter more than they look:

* While a run is going, every control that would change what the run is
  doing is switched off, leaving only Cancel. Rather than letting Qt drop
  the focus wherever it lands when the control holding it is disabled, the
  focus is moved to Cancel first, deliberately, and the user is told.
* Progress is not left to the progress bar alone. A bar has a value a
  screen reader can read, but only if the user goes and looks. The name of
  each file is therefore announced as it starts, which is the pace at
  which something actually changes.
* Everything the run did is written out in words afterwards, in a text box
  that can be read line by line and copied, as well as being summarised in
  a message box.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut, QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from audio_transcriber.audio.enhance import (
    MAXIMUM_CEILING_DBTP,
    MAXIMUM_MAXIMUM_GAIN_DB,
    MAXIMUM_TARGET_LUFS,
    MINIMUM_CEILING_DBTP,
    MINIMUM_MAXIMUM_GAIN_DB,
    MINIMUM_TARGET_LUFS,
    OUTPUT_FORMATS,
    EnhanceOptions,
    FileResult,
    output_format_for,
)
from audio_transcriber.audio.enhance_runner import EnhanceRunner, RunSummary
from audio_transcriber.settings import EnhanceSettings
from audio_transcriber.ui.accessibility import announce, describe
from audio_transcriber.ui.enhance_notes import (
    CEILING,
    FILES,
    LIMITER,
    MAXIMUM_GAIN,
    OUTPUT_FOLDER,
    OUTPUT_FORMAT,
    REPLACE_EXISTING,
    TARGET_LOUDNESS,
    guide_text,
    note_for,
)

#: The name of the folder suggested the first time, alongside the
#: recordings themselves. A sub-folder keeps the enhanced copies out of the
#: file list, which only reads the folder it is given.
DEFAULT_OUTPUT_FOLDER_NAME = "Enhanced"

#: Enough rows to see a normal selection without the dialog growing taller
#: than a small screen at a large font size. Longer lists scroll.
_VISIBLE_FILE_ROWS = 6

#: How much of a note is on show at once. Enough to read a paragraph and
#: see that there is more, without the panel dominating the dialog.
_NOTE_LINES = 7

#: Used when Qt cannot say which screen the dialog is on, which happens in
#: a test run with no desktop.
_FALLBACK_MAXIMUM_HEIGHT = 900

_log = logging.getLogger(__name__)


class EnhancementGuideDialog(QDialog):
    """Every setting explained, one after another, in one readable window.

    The panel in the Enhance Audio dialog shows one note at a time, which
    suits somebody working down the settings. This is the same words laid
    out for somebody who would rather read the lot before touching
    anything, or copy them somewhere else.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Guide to the Enhance Audio settings")

        self._text = QPlainTextEdit(self)
        self._text.setPlainText(guide_text())
        self._text.setReadOnly(True)
        describe(
            self._text,
            "Guide to the Enhance Audio settings",
            "Explains every setting in the Enhance Audio dialog. Use the arrow keys "
            "to read through it.",
        )

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(self._text)
        layout.addWidget(buttons)

        self.resize(640, 620)
        # The text is the point of this window, so it starts with the focus
        # and a screen reader lands on something worth reading.
        self._text.setFocus()


class EnhanceAudioDialog(QDialog):
    """Asks how to enhance the chosen recordings, then does it."""

    def __init__(
        self,
        files: list[Path],
        settings: EnhanceSettings,
        source_folder: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Enhance Audio")
        self._files = list(files)
        self._settings = settings
        self._source_folder = source_folder
        self._runner = EnhanceRunner(self)
        self._summary: RunSummary | None = None
        self._results: list[FileResult] = []
        # Which note belongs to which control, and which is on show. Both
        # are filled in as the controls are built.
        self._note_keys: dict[QWidget, str] = {}
        self._showing_note: str | None = None

        self._build_ui()
        self._connect_signals()
        self._show_settings(settings)
        self._open_at_a_sensible_size()

        # The output folder is the one thing that must be right before
        # anything happens, so it starts with the focus.
        self._show_note(OUTPUT_FOLDER)
        self._folder_edit.setFocus(Qt.FocusReason.TabFocusReason)

    # -- Building the dialog ---------------------------------------------

    def _build_ui(self) -> None:
        contents = QWidget(self)
        inner = QVBoxLayout(contents)
        inner.setContentsMargins(0, 0, 0, 0)
        # Each of these asks for exactly the height its contents need, and
        # that height grows with the system font. Without saying so they
        # would share out whatever the window has spare, which spreads the
        # controls apart and puts a lot of empty space between a label and
        # the field it belongs to.
        for group in (
            self._build_file_list(),
            self._build_output_folder(),
            self._build_parameters(),
            self._build_notes(),
            self._build_progress(),
        ):
            group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            inner.addWidget(group)
        # The report is the one part worth making bigger, so making the
        # window taller gives the room to it.
        inner.addWidget(self._build_report(), 3)
        inner.addStretch(1)

        # At a large Windows text size, or under heavy magnification, this
        # dialog is taller than the screen it has to fit on. Rather than
        # putting the bottom of it out of reach, it scrolls. Qt brings
        # whatever takes focus into view by itself, so tabbing through
        # still works exactly as it reads.
        self._scroll = QScrollArea(self)
        self._scroll.setWidget(contents)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # The scroll area is a container, not something to land on. Naming
        # it would put a stop on the way through that answers no keys of
        # its own.
        self._scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        layout = QVBoxLayout(self)
        layout.addWidget(self._scroll, 1)
        # The buttons stay outside the scrolling part, so Start and Cancel
        # are always where the user left them.
        layout.addWidget(self._build_buttons())

    def _open_at_a_sensible_size(self) -> None:
        """Start out as tall as the dialog's contents, within reason."""
        self.resize(
            max(620, self.sizeHint().width()),
            min(self._preferred_height(), self._room_to_grow()),
        )

    def _grow_to_fit(self) -> None:
        """Make room for something that has just appeared, without moving it.

        The progress bar and the report both appear part way through, and
        the dialog has to find room for them. It grows downwards from
        wherever it is, so that everything already on screen stays exactly
        where the user last saw it, which matters under a magnifier. It
        never shrinks, so a dialog the user has made bigger stays that way,
        and it never grows past the screen, because past the screen is out
        of reach. Anything that does not fit is reached by scrolling.
        """
        wanted = min(self._preferred_height(), self._room_to_grow())
        self.resize(self.width(), max(self.height(), wanted))

    def _preferred_height(self) -> int:
        """How tall the dialog would be if everything on it were shown at once.

        A scrolling area asks for a modest height whatever is inside it,
        because scrolling is exactly what it is for. That is no use for
        deciding how big to open: it would put a scroll bar on a dialog
        that had all the room it needed. So the contents are asked instead,
        and the difference between the two accounts for the buttons and the
        margins around them.
        """
        contents = self._scroll.widget()
        if contents is None:
            return self.sizeHint().height()
        around_it = self.sizeHint().height() - self._scroll.sizeHint().height()
        return contents.sizeHint().height() + around_it

    def _room_to_grow(self) -> int:
        """The tallest this dialog may sensibly be on the screen it is on."""
        screen = self.screen()
        if screen is None:
            return _FALLBACK_MAXIMUM_HEIGHT
        # A little short of the working area, so the title bar and the edge
        # of the screen are not fighting for the same pixels.
        return int(screen.availableGeometry().height() * 0.9)

    def _file_count_text(self) -> str:
        """The heading over the file list, which says how many there are.

        The count is in the heading rather than left to be worked out from
        the length of the list, so it is heard on the way in rather than
        counted on the way through.
        """
        count = len(self._files)
        if count == 1:
            return "1 file to enhance"
        return f"{count} files to enhance"

    def _build_file_list(self) -> QWidget:
        group = QGroupBox(self._file_count_text(), self)
        self._file_group = group

        self._file_list = QListWidget(group)
        self._file_list.addItems([path.name for path in self._files])
        # Nothing here is chosen or acted on; the list is what the run will
        # work through. It still takes focus, so it can be read line by
        # line before the run starts.
        self._file_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        # As tall as the list needs, up to a few rows, after which it
        # scrolls. The height is worked out from the height of a real row,
        # so it follows the system font rather than assuming a size.
        row_height = self._file_list.sizeHintForRow(0)
        if row_height > 0:
            rows = min(max(1, len(self._files)), _VISIBLE_FILE_ROWS)
            self._file_list.setFixedHeight(
                rows * row_height + 2 * self._file_list.frameWidth() + 4
            )
        self._explain(self._file_list, FILES)

        inner = QVBoxLayout(group)
        inner.addWidget(self._file_list)
        return group

    def _build_output_folder(self) -> QWidget:
        group = QGroupBox("Output folder", self)
        self._folder_group = group

        label = QLabel("Write the enhanced copies t&o", group)
        # Read-only rather than a plain label, so the path can take focus,
        # shows a real caret for ZoomText to follow, and can be copied.
        self._folder_edit = QLineEdit(group)
        self._folder_edit.setReadOnly(True)
        self._explain(self._folder_edit, OUTPUT_FOLDER)
        label.setBuddy(self._folder_edit)

        self._browse_button = QPushButton("B&rowse...", group)
        describe(
            self._browse_button,
            "Browse for output folder",
            "Opens a dialog for choosing where the enhanced recordings are written.",
        )
        # Standing on Browse is standing on the output folder, as far as
        # the explanation goes, so the note stays put rather than clearing.
        self._note_keys[self._browse_button] = OUTPUT_FOLDER

        row = QHBoxLayout()
        row.addWidget(self._folder_edit, 1)
        row.addWidget(self._browse_button, 0)

        inner = QVBoxLayout(group)
        inner.addWidget(label)
        inner.addLayout(row)
        return group

    def _build_parameters(self) -> QWidget:
        group = QGroupBox("Loudness", self)
        self._parameters_group = group
        form = QFormLayout(group)
        # Long rows wrap onto a second line rather than forcing the dialog
        # wider than the screen at a large font size.
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self._target_box = self._add_number_row(
            form, "&Target loudness", " LUFS", MINIMUM_TARGET_LUFS, MAXIMUM_TARGET_LUFS,
            TARGET_LOUDNESS,
        )
        self._ceiling_box = self._add_number_row(
            # Alt+C belongs to the Cancel button, so the ceiling takes Alt+P.
            form, "True-&peak ceiling", " dBTP", MINIMUM_CEILING_DBTP, MAXIMUM_CEILING_DBTP,
            CEILING,
        )
        self._gain_box = self._add_number_row(
            form, "&Maximum gain", " dB", MINIMUM_MAXIMUM_GAIN_DB, MAXIMUM_MAXIMUM_GAIN_DB,
            MAXIMUM_GAIN,
        )

        self._limiter_box = QCheckBox("Use a &limiter to reach the target loudness", group)
        self._explain(self._limiter_box, LIMITER)
        form.addRow(self._limiter_box)

        format_label = QLabel("Output &format", group)
        self._format_box = QComboBox(group)
        for output_format in OUTPUT_FORMATS:
            self._format_box.addItem(output_format.label, output_format.key)
        self._explain(self._format_box, OUTPUT_FORMAT)
        format_label.setBuddy(self._format_box)
        form.addRow(format_label, self._format_box)

        self._replace_box = QCheckBox("R&eplace files that are already in the folder", group)
        self._explain(self._replace_box, REPLACE_EXISTING)
        form.addRow(self._replace_box)
        return group

    def _add_number_row(
        self,
        form: QFormLayout,
        label_text: str,
        suffix: str,
        lowest: float,
        highest: float,
        note_key: str,
    ) -> QDoubleSpinBox:
        label = QLabel(label_text, self)
        box = QDoubleSpinBox(self)
        box.setRange(lowest, highest)
        box.setDecimals(1)
        box.setSingleStep(0.5)
        box.setSuffix(suffix)
        # The box will not accept a value outside its range, so there is no
        # error state to report and nothing for the user to correct.
        self._explain(box, note_key)
        label.setBuddy(box)
        form.addRow(label, box)
        return box

    def _build_notes(self) -> QWidget:
        """The panel that explains whichever setting the focus is on.

        Every setting here needs more explaining than fits on its label, and
        more than is bearable to hear read out on every focus. A tooltip
        cannot carry it either: it needs a mouse, it times out, and it
        cannot be scrolled or copied.

        So the explanation goes in a plain read-only text box that follows
        the focus. Tab onto a setting and its note appears; tab into the box
        to read the note at leisure, where it stays put rather than
        changing under you.
        """
        group = QGroupBox("About this setting", self)
        self._notes_group = group

        self._notes_text = QPlainTextEdit(group)
        self._notes_text.setReadOnly(True)
        self._notes_text.setFixedHeight(self.fontMetrics().lineSpacing() * _NOTE_LINES)
        describe(
            self._notes_text,
            "About this setting",
            "Explains the setting the focus is on. Use the arrow keys to read "
            "through it. It stays on the last setting while you read, and the "
            "Guide button shows all of them together.",
        )

        self._guide_button = QPushButton("&Guide to these settings...", group)
        describe(
            self._guide_button,
            "Guide to these settings",
            "Opens a window explaining every setting in this dialog, one after "
            "another, so they can be read straight through. F1 does the same.",
        )
        self._guide_button.setAutoDefault(False)

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self._guide_button, 0)

        inner = QVBoxLayout(group)
        inner.addWidget(self._notes_text)
        inner.addLayout(row)
        return group

    def _explain(self, widget: QWidget, key: str) -> QWidget:
        """Name a control from its note, and have the panel follow it.

        The short summary becomes the accessible description and the
        tooltip, so a screen reader reads it on focus and a mouse user sees
        it on hover. The full note is left to the panel, because hearing
        several paragraphs every time the focus moves would be unusable.
        """
        note = note_for(key)
        describe(widget, note.title, note.summary)
        self._note_keys[widget] = key
        return widget

    def _on_focus_changed(self, _old: QWidget | None, new: QWidget | None) -> None:
        """Show the note for whatever now has the focus, if it has one.

        The search walks up from the focused widget rather than looking it
        up directly, because the widget that actually takes focus is often
        a part of the control rather than the control itself: a spin box
        hands it to the text field inside it. Walking up finds the control
        the user thinks they are on.

        Focus that belongs to nothing in particular, including the note
        panel itself, leaves the note where it is. That is what lets the
        note be read: moving into it to read it must not change it.
        """
        widget = new
        while widget is not None and widget is not self:
            key = self._note_keys.get(widget)
            if key is not None:
                self._show_note(key)
                return
            widget = widget.parentWidget()

    def _show_note(self, key: str) -> None:
        note = note_for(key)
        if self._showing_note == key:
            return
        self._showing_note = key
        self._notes_group.setTitle(f"About: {note.title}")
        self._notes_text.setPlainText(note.note)
        # Back to the top, so reading starts at the first line rather than
        # wherever the last note was left.
        self._notes_text.moveCursor(QTextCursor.MoveOperation.Start)
        self._notes_text.verticalScrollBar().setValue(0)

    def show_guide(self) -> None:
        """Open the window that explains every setting one after another."""
        EnhancementGuideDialog(self).exec()

    def _build_progress(self) -> QWidget:
        self._progress_group = QGroupBox("Progress", self)
        self._progress_bar = QProgressBar(self._progress_group)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        describe(
            self._progress_bar,
            "Overall progress",
            "How far the run has come through all the files, as a percentage.",
        )

        # This label carries messages, so it is left unnamed. A label has no
        # accessible value of its own: its text is its name, and naming it
        # would hide what it says.
        self._progress_label = QLabel("Starting...", self._progress_group)
        self._progress_label.setWordWrap(True)

        inner = QVBoxLayout(self._progress_group)
        inner.addWidget(self._progress_bar)
        inner.addWidget(self._progress_label)
        # Nothing has started, so there is nothing to show yet.
        self._progress_group.setVisible(False)
        return self._progress_group

    def _build_report(self) -> QWidget:
        self._report_group = QGroupBox("What was done", self)
        self._report_text = QPlainTextEdit(self._report_group)
        self._report_text.setReadOnly(True)
        # Enough lines to read a report on one file without scrolling,
        # measured in lines of the current font rather than in pixels.
        self._report_text.setMinimumHeight(self.fontMetrics().lineSpacing() * 5)
        describe(
            self._report_text,
            "What was done",
            "One line for each recording, saying how loud it was, how far it was "
            "moved, and where it was written. Use the arrow keys to read through it.",
        )
        inner = QVBoxLayout(self._report_group)
        inner.addWidget(self._report_text)
        self._report_group.setVisible(False)
        return self._report_group

    def _build_buttons(self) -> QWidget:
        self._buttons = QDialogButtonBox(self)
        self._start_button = self._buttons.addButton(
            "&Start", QDialogButtonBox.ButtonRole.ActionRole
        )
        describe(
            self._start_button,
            "Start enhancing",
            "Begins working through the files. Everything else is switched off until "
            "the run has finished or you cancel it.",
        )
        self._start_button.setDefault(True)
        self._close_button = self._buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        self._close_button.setAutoDefault(False)
        self._describe_close_button()
        return self._buttons

    def _connect_signals(self) -> None:
        self._browse_button.clicked.connect(self.browse_for_output_folder)
        self._guide_button.clicked.connect(self.show_guide)
        self._start_button.clicked.connect(self.start)
        self._close_button.clicked.connect(self.reject)

        # F1 asks for help on what is in front of you, everywhere else in
        # Windows, so it does here too.
        self._guide_shortcut = QShortcut(QKeySequence(Qt.Key.Key_F1), self)
        self._guide_shortcut.activated.connect(self.show_guide)

        # The application is asked about the focus rather than each control
        # being watched, because the widget that actually takes focus is
        # often a part of a control rather than the control itself.
        application = QApplication.instance()
        if application is not None:
            application.focusChanged.connect(self._on_focus_changed)

        self._runner.fileStarted.connect(self._on_file_started)
        self._runner.fileFinished.connect(self._on_file_finished)
        self._runner.progressChanged.connect(self._progress_bar.setValue)
        self._runner.runFinished.connect(self._on_run_finished)

    # -- Settings in and out ---------------------------------------------

    def _show_settings(self, settings: EnhanceSettings) -> None:
        self._folder_edit.setText(str(self._starting_output_folder(settings)))
        self._folder_edit.setCursorPosition(0)
        self._target_box.setValue(settings.target_lufs)
        self._ceiling_box.setValue(settings.ceiling_dbtp)
        self._gain_box.setValue(settings.maximum_gain_db)
        self._limiter_box.setChecked(settings.use_limiter)
        self._replace_box.setChecked(settings.replace_existing)
        chosen = self._format_box.findData(output_format_for(settings.output_format).key)
        self._format_box.setCurrentIndex(max(0, chosen))

    def _starting_output_folder(self, settings: EnhanceSettings) -> Path:
        """Where to suggest writing to.

        The folder from last time is offered again. Failing that, a
        sub-folder of the recordings themselves, which keeps the enhanced
        copies together with the originals without them turning up in the
        file list.
        """
        if settings.output_folder:
            return Path(settings.output_folder)
        if self._source_folder is not None:
            return self._source_folder / DEFAULT_OUTPUT_FOLDER_NAME
        return Path.home() / DEFAULT_OUTPUT_FOLDER_NAME

    def chosen_settings(self) -> EnhanceSettings:
        """The settings as the user left them, to be remembered for next time."""
        return EnhanceSettings(
            output_folder=self._folder_edit.text() or None,
            target_lufs=self._target_box.value(),
            ceiling_dbtp=self._ceiling_box.value(),
            maximum_gain_db=self._gain_box.value(),
            use_limiter=self._limiter_box.isChecked(),
            output_format=str(self._format_box.currentData()),
            replace_existing=self._replace_box.isChecked(),
        )

    def chosen_options(self) -> EnhanceOptions:
        """What the user chose, in the form the enhancement engine wants."""
        settings = self.chosen_settings()
        return EnhanceOptions(
            output_folder=Path(settings.output_folder or "."),
            target_lufs=settings.target_lufs,
            ceiling_dbtp=settings.ceiling_dbtp,
            maximum_gain_db=settings.maximum_gain_db,
            use_limiter=settings.use_limiter,
            output_format=output_format_for(settings.output_format),
            replace_existing=settings.replace_existing,
        )

    def browse_for_output_folder(self) -> None:
        """Ask for a folder, starting from the one already shown."""
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Select the folder for the enhanced recordings",
            self._folder_edit.text(),
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )
        if not chosen:
            return
        self._folder_edit.setText(str(Path(chosen)))
        self._folder_edit.setCursorPosition(0)
        announce(self._folder_edit, f"Output folder set to {chosen}")

    # -- Running ----------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._runner.is_running

    @property
    def summary(self) -> RunSummary | None:
        """What the last run came to, or ``None`` if none has finished."""
        return self._summary

    def start(self) -> bool:
        """Begin the run, unless something is missing. Returns whether it began."""
        if self.is_running:
            return False
        if not self._files:
            self._refuse("There are no files to enhance.")
            return False
        if not self._folder_edit.text().strip():
            self._refuse(
                "No output folder has been chosen. Use the Browse button to choose "
                "where the enhanced recordings should go."
            )
            self._browse_button.setFocus(Qt.FocusReason.OtherFocusReason)
            return False

        options = self.chosen_options()
        try:
            options.output_folder.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            reason = error.strerror or str(error)
            self._refuse(
                f"The output folder {options.output_folder} could not be created. {reason}."
            )
            return False

        self._results = []
        self._summary = None
        self._report_text.clear()
        self._report_group.setVisible(False)
        self._progress_bar.setValue(0)
        self._progress_group.setVisible(True)
        self._grow_to_fit()
        self._set_progress_text(f"Starting on {len(self._files)} files.")
        # The focus is moved off the controls before they are switched off,
        # rather than after. A control that is disabled while it holds the
        # focus takes the focus with it, and the user is left nowhere.
        self._close_button.setFocus(Qt.FocusReason.OtherFocusReason)
        self._set_controls_enabled(False)

        if not self._runner.start(self._files, options):
            self._set_controls_enabled(True)
            self._describe_close_button()
            return False
        # Only now is the run really going, so only now does the Cancel
        # button mean "stop the run" rather than "close the dialog".
        self._describe_close_button()
        announce(self._progress_bar, f"Enhancing {len(self._files)} files.", urgent=True)
        return True

    def cancel(self) -> None:
        """Ask a running enhancement to stop as soon as it can."""
        if not self.is_running:
            return
        self._runner.cancel()
        self._set_progress_text("Stopping...")
        announce(self._progress_bar, "Stopping.", urgent=True)

    def _set_controls_enabled(self, enabled: bool) -> None:
        """Switch everything except Cancel off while a run is going."""
        for widget in (
            self._file_group,
            self._folder_group,
            self._parameters_group,
            self._start_button,
        ):
            widget.setEnabled(enabled)

    def _describe_close_button(self) -> None:
        """Name the button for what it does now, which changes as the run does."""
        if self.is_running:
            text, name = "&Cancel", "Cancel"
            description = (
                "Stops the run. Files already written are kept, and the one in "
                "progress is thrown away."
            )
        elif self._summary is not None:
            text, name = "&Close", "Close"
            description = "Closes this dialog."
        else:
            text, name = "&Cancel", "Cancel"
            description = "Closes this dialog without enhancing anything."
        self._close_button.setText(text)
        # The tooltip is cleared first, because describe only fills an empty
        # one in and this button is described more than once.
        self._close_button.setToolTip("")
        describe(self._close_button, name, description)

    # -- What the run reports ---------------------------------------------

    def _on_file_started(self, name: str, number: int, total: int) -> None:
        message = f"Enhancing {name}. File {number} of {total}."
        self._set_progress_text(message)
        # Each file is announced as it starts. The progress bar has a value
        # a screen reader can read, but only if the user goes and asks for
        # it; this is the pace at which something actually changes.
        announce(self._progress_bar, message)

    def _on_file_finished(self, result: FileResult) -> None:
        self._results.append(result)
        self._show_report()

    def _on_run_finished(self, summary: RunSummary) -> None:
        self._summary = summary
        self._results = list(summary.results)
        self._show_report()
        if not summary.cancelled:
            self._progress_bar.setValue(100)

        message = summarise(summary)
        self._set_progress_text(message)
        self._set_controls_enabled(True)
        self._describe_close_button()
        self._close_button.setFocus(Qt.FocusReason.OtherFocusReason)
        self._show_completion_message(message)

    def _show_completion_message(self, message: str) -> None:
        """Say plainly that the run is over, and how it went.

        A message box is used because it takes the focus and every screen
        reader reads it without being asked. The same words stay on the
        dialog behind it, and the detail stays in the report, so nothing is
        lost by dismissing it.
        """
        box = QMessageBox(self)
        box.setWindowTitle("Enhance Audio")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText(message)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

    def _show_report(self) -> None:
        """Write out what has happened so far, one paragraph per file.

        The whole report is rewritten each time rather than added to, so
        that it says the same thing whether it is read part way through a
        run or after the summary has arrived.
        """
        self._report_text.setPlainText(
            "\n\n".join(result.message for result in self._results) or "Nothing was enhanced."
        )
        self._report_group.setVisible(True)
        self._grow_to_fit()

    def _set_progress_text(self, message: str) -> None:
        self._progress_label.setText(message)

    def _refuse(self, message: str) -> None:
        """Report why the run cannot start, on screen and out loud."""
        self._progress_group.setVisible(True)
        self._grow_to_fit()
        self._set_progress_text(message)
        announce(self._progress_bar, message, urgent=True)

    # -- Closing ----------------------------------------------------------

    def reject(self) -> None:
        """Cancel a run rather than closing on top of it.

        The dialog owns the thread doing the work, so closing while it runs
        would leave that thread reporting into a window that is being taken
        apart. Escape and the Cancel button therefore stop the run first,
        and closing is left to a second press once it has stopped.
        """
        if self.is_running:
            self.cancel()
            return
        super().reject()

    def closeEvent(self, event) -> None:
        """Refuse to close on a running job, and never close on a live thread.

        Closing the window is the same decision as pressing Cancel, so it
        is answered the same way. Once nothing is running, the thread is
        stopped and waited for before the dialog goes, because it reports
        through signals on an object that is about to be destroyed.
        """
        if self.is_running:
            self.cancel()
            event.ignore()
            return
        self._runner.stop()
        # The focus watch is on the application, which outlives this
        # dialog, so it is let go of deliberately rather than left to be
        # tidied up.
        application = QApplication.instance()
        if application is not None:
            try:
                application.focusChanged.disconnect(self._on_focus_changed)
            except (RuntimeError, TypeError):
                _log.debug("The focus watch was already disconnected.")
        super().closeEvent(event)


def summarise(summary: RunSummary) -> str:
    """One or two sentences saying how a finished run went."""
    total = len(summary.results)
    parts = [f"{summary.enhanced} of {total} files enhanced"]
    if summary.skipped:
        parts.append(f"{summary.skipped} skipped")
    if summary.failed:
        parts.append(f"{summary.failed} could not be enhanced")
    counts = ", ".join(parts) + "."
    if summary.cancelled:
        return f"The run was cancelled. {counts}"
    if total and summary.enhanced == total:
        return f"Finished. {counts}"
    return f"Finished, with something to report. {counts} See the list below for each file."
