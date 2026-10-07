"""The Transcribe dialog.

The dialog opens with the recordings the user picked already listed, asks
the few things about them that no service can work out for itself, says what
the run is expected to cost, and then runs it in the background while showing
how far it has come.

It is the same shape as the Enhance Audio dialog, and deliberately so, but
three things about it are different because transcription is different.

The first is money. Enhancing a recording costs nothing but time; sending it
to four services costs real money, and the person pressing Start is paying.
So the estimate is on screen before the run rather than in next month's
invoice, and where the user has asked to be, they are asked to agree to it.
The estimate is honest about being an estimate: the rates behind it are
typed into Settings by hand and may be out of date, and the parts that
depend on how much the services disagree cannot be known before they have
answered.

The second is that a half-finished run is expensive. A missing API key found
half way through means the services that did answer have been paid for a
transcript nobody can use. So everything that would stop a run is checked
before anything is sent, and a run that cannot succeed is refused rather
than started.

The third is that the interesting answer arrives after the run, not during
it. A finished transcript usually has words in it that no service could
settle, and those words are what the user actually has to do something
about. The dialog therefore ends by saying how many there are and offering
the two things worth doing next: opening the transcript folder, and opening
the review window at the recording that needs it.

That last one is worth being precise about, because it changed. The review
covers the whole folder rather than the one recording just transcribed, and
the recording named here is only where the person is put down in it. A shaky
word in a file transcribed this morning is very often the same word as one
already settled in a file transcribed last week, and a review that could see
only the new file would hide the very decision that has already been taken.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PySide6.QtCore import QEvent, QObject, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut, QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from vox_verbatim.audio.library import AudioFile
from vox_verbatim.json_store import JsonReadStatus
from vox_verbatim.paths import calibration_file_path
from vox_verbatim.settings import (
    MAXIMUM_EXPECTED_SPEAKER_COUNT,
    MINIMUM_EXPECTED_SPEAKER_COUNT,
    TranscriptionSettings,
)
from vox_verbatim.transcription.cost import (
    CostEstimate,
    describe_duration,
    describe_money,
    estimate_cost,
)
from vox_verbatim.transcription.model import Provider, RecordingConfiguration
from vox_verbatim.transcription.pipeline import PipelineOptions
from vox_verbatim.transcription.project import (
    PROJECT_FILE_NAME,
    LearnedName,
    read_learned_names,
    remove_learned_name,
)
from vox_verbatim.transcription.providers.registry import DEFAULT_FULL_PASS_PROVIDERS
from vox_verbatim.transcription.runner import (
    RecordingOutcome,
    RunSummary,
    TranscriptionRunner,
    summarise,
)
from vox_verbatim.transcription.store import TranscriptStore
from vox_verbatim.transcription.vocabulary import (
    Vocabulary,
    VocabularyTerm,
    resolve_terms,
    terms_from_learned_names,
)
from vox_verbatim.ui.accessibility import announce, describe
from vox_verbatim.ui.review_lists import NO_LANGUAGE, language_name

_log = logging.getLogger(__name__)

#: Enough rows to see a normal selection without the dialog growing taller
#: than a small screen at a large font size. Longer lists scroll.
_VISIBLE_ROWS = 6

#: How much of a note is on show at once. Enough to read a paragraph and see
#: that there is more, without the panel dominating the dialog.
_NOTE_LINES = 7

#: Used when Qt cannot say which screen the dialog is on, which happens in a
#: test run with no desktop.
_FALLBACK_MAXIMUM_HEIGHT = 900


# -- What each setting means ---------------------------------------------
#
# Every setting here carries two pieces of writing. The summary is one or two
# sentences and becomes the control's accessible description and its tooltip,
# so a screen reader reads it on focus. The note is the full explanation,
# which is far too long to hear on every focus, so it goes in a panel that
# follows the focus and can be read at leisure.

RECORDINGS = "recordings"
AFRIKAANS = "afrikaans"
SPEAKER_COUNT = "speaker_count"
KNOWN_SPEAKERS = "known_speakers"
CONTEXT = "context"
PROFILES = "profiles"
LEARNED_NAMES = "learned_names"
COST = "cost"


@dataclass(frozen=True)
class Note:
    """What one setting is called, and what there is to say about it."""

    key: str
    title: str
    summary: str
    note: str


def _reflowable(text: str) -> str:
    """Put each paragraph on one line, so it can be wrapped to fit.

    The notes are written wrapped in the source below, because that is where
    they are read and reviewed. On screen the wrapping has to follow the
    width of the panel and the size of the user's font instead, and a screen
    reader reads a text box a line at a time, so a paragraph broken across
    eight lines is read as eight lines with a pause in each gap.
    """
    paragraphs = text.strip().split("\n\n")
    return "\n\n".join(
        " ".join(line.strip() for line in paragraph.splitlines()).strip()
        for paragraph in paragraphs
    )


_WRITTEN_NOTES: tuple[Note, ...] = (
    Note(
        key=RECORDINGS,
        title="Recordings to transcribe",
        summary=(
            "The recordings this run will work through, one after another. To change "
            "them, close this dialog, check the ones you want in the file list, and "
            "open it again."
        ),
        note="""\
These are the recordings this run will work through.

They are the files you checked in the main window, or the highlighted file if
you checked nothing. To change the list, close this dialog, check the files
you want, and open it again.

They are transcribed one after another rather than all at once. Each one
already has several services running at the same time inside it, and the
limits those services impose are per account rather than per recording, so
running two recordings together would not be faster. They would simply get
in each other's way.

Everything below applies to all of them. If one recording needs different
settings from the rest, transcribe it on its own.
""",
    ),
    Note(
        key=AFRIKAANS,
        title="Afrikaans may occur",
        summary=(
            "Switch this on only where Afrikaans may genuinely be spoken. Off means "
            "the application does not look for Afrikaans at all, which makes a "
            "recording without it both cheaper and more accurate."
        ),
        note="""\
This is not a preference about which language you would rather have. It
decides whether the application looks for Afrikaans at all.

Left off, no part of the recording is ever considered to be Afrikaans. There
is no Afrikaans detection and no Afrikaans fallback for any service, so a
German passage cannot be mistaken for Afrikaans, because Afrikaans is not one
of the available answers. That is why leaving it off makes a recording
without Afrikaans both cheaper and more accurate: a whole class of wrong
language guesses disappears, and no service is sent down a path that assumes
it can hear a language it cannot.

Switching it on narrows the evidence rather than widening it, and that is
worth knowing before you do. Of the five services, only ElevenLabs and OpenAI
handle Afrikaans well. Microsoft and Deepgram do not support it at all.
AssemblyAI can only hear it on its older model, at a documented word error
rate of between 25 and 50 per cent, so a second opinion on an Afrikaans
passage is a far weaker second opinion than on an English one.

So switch it on where Afrikaans may genuinely occur in these recordings, and
leave it off where it may not. There is nothing to be gained by switching it
on just in case.
""",
    ),
    Note(
        key=SPEAKER_COUNT,
        title="Expected number of speakers",
        summary=(
            "How many people you expect to hear. It is a hint to the services that "
            "tell speakers apart, not a limit, and getting it roughly right helps "
            "more than getting it exactly right."
        ),
        note="""\
How many different people you expect to hear in these recordings.

The services that tell speakers apart do far better when they are told
roughly how many to look for. Left to themselves they tend to split one
person into several when the line is noisy, or to merge two people who sound
alike.

It is a hint rather than a rule. A number that turns out to be wrong does not
break anything: the transcript reports the speakers actually found, and the
review window lets you correct and rename them afterwards.

One is the honest default, because most recordings are one person talking. If
you are not sure, count the people you expect to hear rather than the people
who were in the room.
""",
    ),
    Note(
        key=KNOWN_SPEAKERS,
        title="Known speaker names",
        summary=(
            "The names of the people you expect to hear, separated by commas, in the "
            "order they are likely to speak first. They are a starting guess you can "
            "correct afterwards."
        ),
        note="""\
The names of the people you expect to hear, separated by commas.

The services do not recognise voices, so they cannot work out who is talking.
They label speakers as "speaker 0", "speaker 1" and so on. These names are
matched to those labels in the order each speaker first says something.

That order is a guess, and it is treated as one. It saves you renaming
everybody by hand in the usual case where the person who calls the meeting to
order speaks first, and it is wrong whenever somebody else happens to speak
first. The review window lets you rename any speaker, so a wrong guess costs
a moment rather than the transcript.

The names are also given to the services that accept a list of words to
listen out for, which is worth more than the labelling: a surname the service
has been told to expect is far more likely to come back spelled correctly.
""",
    ),
    Note(
        key=CONTEXT,
        title="Recording context",
        summary=(
            "A sentence or two about what these recordings are, which is given to the "
            "services that accept context and to the model that settles disputes."
        ),
        note="""\
A sentence or two saying what these recordings are about.

Something like "a quarterly review meeting between the finance team and their
auditors, discussing the year-end close" is enough. It is passed to the
services that accept a description of the audio, and to the language model
that decides the disputes the other rules could not settle.

It earns its place on the words that sound alike. A model deciding between
two readings of a muffled word has almost nothing to go on except what the
conversation is about, and one sentence of context is often the difference
between the right industry term and an everyday word that sounds like it.

Write it in plain sentences rather than as keywords. Names, products and
jargon belong in a vocabulary profile instead, where they are sent to the
services as terms to listen out for.
""",
    ),
    Note(
        key=PROFILES,
        title="Vocabulary profiles",
        summary=(
            "The lists of names and specialist words to send to the services for "
            "these recordings. Use the Space bar to switch one on or off."
        ),
        note="""\
The lists of words the services should listen out for in these recordings.

Speech services are good at ordinary words and weak at exactly the words that
matter most: a client's surname, a product nobody else sells, an acronym that
sounds like an everyday word. Telling them beforehand makes a real difference
to whether those words come back spelled correctly.

Choose only the lists that apply. This is the part people get wrong. A
surname that is almost certain in one client's meeting is a distraction in
somebody else's, and several services cap how many terms they will accept, so
an unnecessary list does not simply sit there unused. It pushes the terms
that do matter off the end.

The corrections you have made while reviewing transcripts are always sent,
whichever profiles are chosen, because they are the only evidence here that
came from real audio rather than from a list typed out in advance.

Profiles are created and edited in Settings, under Vocabulary.
""",
    ),
    Note(
        key=LEARNED_NAMES,
        title="Names learned in this folder",
        summary=(
            "The names your reviews in this folder have taught. Select a wrong one "
            "and press Delete, or the Remove name button, to forget it."
        ),
        note="""The names that your corrections in the review window have taught this folder.

When you correct a word the services got wrong, and the correct word looks
like a name, the folder remembers it, together with the spellings the services
wrote instead. Each line says the name, its language, and how the services
heard it.

These names belong to this folder only. They are kept in the folder's project
file, not in the vocabulary profiles in Settings, so one client's surname is
never suggested for another client's recordings.

A slip in a review can teach a wrong name. To forget one, select it and press
Delete, or use the Remove name button. Removing a name only forgets it: if you
correct the same word again in a review, the folder learns it again.
""",
    ),
    Note(
        key=COST,
        title="What this run will cost",
        summary=(
            "What the services are expected to charge for this run, worked out from "
            "the length of the audio and the rates in Settings. It is an estimate, "
            "and it does not cover everything."
        ),
        note="""\
What this run is expected to cost, and how that figure was arrived at.

Every service listed is sent the whole of every recording, and each charges by
the minute of audio, so that part can be worked out exactly from the lengths
above. What cannot be worked out is everything that depends on how much the
services turn out to disagree with each other: the second opinions on
disputed passages, and asking a language model to decide between them. Both
are charged on top of the figure shown, and neither can be known until the
services have answered.

The rates themselves are numbers typed into Settings by hand. Prices change
without telling us, so treat the figure as being as up to date as those
rates, and check them against what each service charges today before relying
on it.

A service with no rate entered is reported as unpriced rather than as free,
and is left out of the total. Anything named that way is money the estimate
does not account for.
""",
    ),
)

_NOTES: dict[str, Note] = {
    note.key: Note(
        key=note.key,
        title=note.title,
        summary=" ".join(note.summary.split()),
        note=_reflowable(note.note),
    )
    for note in _WRITTEN_NOTES
}


def note_for(key: str) -> Note:
    return _NOTES[key]


def guide_text() -> str:
    """Every note, one after another, for somebody who would rather read them all."""
    return "\n\n".join(
        f"{note.title}\n\n{note.note}" for note in (_NOTES[note.key] for note in _WRITTEN_NOTES)
    )


class TranscriptionGuideDialog(QDialog):
    """Every setting explained, one after another, in one readable window.

    The panel in the Transcribe dialog shows one note at a time, which suits
    somebody working down the settings. This is the same words laid out for
    somebody who would rather read the lot before touching anything, or copy
    them somewhere else.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Guide to the transcription settings")

        self._text = QPlainTextEdit(self)
        self._text.setPlainText(guide_text())
        self._text.setReadOnly(True)
        describe(
            self._text,
            "Guide to the transcription settings",
            "Explains every setting in the Transcribe dialog. Use the arrow keys to "
            "read through it.",
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


class TranscribeDialog(QDialog):
    """Asks how to transcribe the chosen recordings, says what it will cost, then does it."""

    detachedRunFinished = Signal(object)
    """A run the dialog was closed on has stopped, carrying its :class:`RunSummary`.

    Sent only for a run that was left to stop in the background. The dialog
    is hidden by then and nobody can read what it would show, so whoever
    opened it is handed the summary to say out loud where the person now is.
    """

    def __init__(
        self,
        recordings: Sequence[AudioFile],
        settings: TranscriptionSettings,
        vocabulary: Vocabulary | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Transcribe")
        self._recordings = list(recordings)
        # The folder the recordings sit in, which is where the project file
        # holding the names learned in its reviews lives.
        self._folder: Path | None = self._recordings[0].path.parent if self._recordings else None
        # The folder's learned names as terms. Read now for the cost estimate,
        # and read again when the run starts, which is what the run sends.
        self._folder_terms: tuple[VocabularyTerm, ...] = self._read_folder_terms()[0]
        self._settings = settings
        self._vocabulary = vocabulary if vocabulary is not None else Vocabulary()
        self._runner = TranscriptionRunner(self)
        self._summary: RunSummary | None = None
        self._results: list[RecordingOutcome] = []
        self._review_request: RecordingOutcome | None = None
        # Whether Cancel has been pressed on the run now going, and whether
        # the dialog was then closed on top of it. Both are reset at start.
        self._cancel_requested = False
        self._detached = False
        # The last stage sentence spoken, so the same one is not read out
        # again every time the progress bar moves.
        self._announced_stage = ""
        # Which note belongs to which control, and which is on show. Both are
        # filled in as the controls are built.
        self._note_keys: dict[QWidget, str] = {}
        self._showing_note: str | None = None
        self._watching_focus = False

        self._build_ui()
        self._connect_signals()
        self._show_defaults()
        self._show_cost()
        self._open_at_a_sensible_size()

        # Whether Afrikaans may occur is the one choice that changes what the
        # run does rather than how well it does it, so it starts with the
        # focus and its explanation is the one on show.
        self._show_note(AFRIKAANS)
        self._afrikaans_box.setFocus(Qt.FocusReason.TabFocusReason)

    # -- Building the dialog ---------------------------------------------

    def _build_ui(self) -> None:
        contents = QWidget(self)
        inner = QVBoxLayout(contents)
        inner.setContentsMargins(0, 0, 0, 0)
        # Each of these asks for exactly the height its contents need, and
        # that height grows with the system font. Without saying so they would
        # share out whatever the window has spare, which spreads the controls
        # apart and puts a lot of empty space between a label and the field it
        # belongs to.
        for group in (
            self._build_recording_list(),
            self._build_configuration(),
            self._build_vocabulary(),
            self._build_cost(),
            self._build_notes(),
            self._build_progress(),
        ):
            group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            inner.addWidget(group)
        # The report is the one part worth making bigger, so making the window
        # taller gives the room to it.
        inner.addWidget(self._build_report(), 3)
        inner.addStretch(1)

        # At a large Windows text size, or under heavy magnification, this
        # dialog is taller than the screen it has to fit on. Rather than
        # putting the bottom of it out of reach, it scrolls. Qt brings
        # whatever takes focus into view by itself, so tabbing through still
        # works exactly as it reads.
        self._scroll = QScrollArea(self)
        self._scroll.setWidget(contents)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # The scroll area is a container, not something to land on. Naming it
        # would put a stop on the way through that answers no keys of its own.
        self._scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        layout = QVBoxLayout(self)
        layout.addWidget(self._scroll, 1)
        # The buttons stay outside the scrolling part, so Start and Cancel are
        # always where the user left them.
        layout.addWidget(self._build_buttons())

    def _open_at_a_sensible_size(self) -> None:
        """Start out as tall as the dialog's contents, within reason."""
        self.resize(
            max(680, self.sizeHint().width()),
            min(self._preferred_height(), self._room_to_grow()),
        )

    def _grow_to_fit(self) -> None:
        """Make room for something that has just appeared, without moving it.

        The progress bar and the report both appear part way through, and the
        dialog has to find room for them. It grows downwards from wherever it
        is, so that everything already on screen stays exactly where the user
        last saw it, which matters under a magnifier. It never shrinks, so a
        dialog the user has made bigger stays that way, and it never grows
        past the screen, because past the screen is out of reach.
        """
        wanted = min(self._preferred_height(), self._room_to_grow())
        self.resize(self.width(), max(self.height(), wanted))

    def _preferred_height(self) -> int:
        """How tall the dialog would be if everything on it were shown at once.

        A scrolling area asks for a modest height whatever is inside it,
        because scrolling is exactly what it is for. That is no use for
        deciding how big to open: it would put a scroll bar on a dialog that
        had all the room it needed. So the contents are asked instead, and the
        difference between the two accounts for the buttons and the margins
        around them.
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
        # A little short of the working area, so the title bar and the edge of
        # the screen are not fighting for the same pixels.
        return int(screen.availableGeometry().height() * 0.9)

    def _recording_count_text(self) -> str:
        """The heading over the list, which says how many recordings there are.

        The count is in the heading rather than left to be worked out from the
        length of the list, so it is heard on the way in rather than counted
        on the way through.
        """
        count = len(self._recordings)
        if count == 1:
            return "1 recording to transcribe"
        return f"{count} recordings to transcribe"

    def _build_recording_list(self) -> QWidget:
        group = QGroupBox(self._recording_count_text(), self)
        self._recording_group = group

        self._recording_list = QListWidget(group)
        for recording in self._recordings:
            self._recording_list.addItem(_recording_line(recording))
        # Nothing here is chosen or acted on; the list is what the run will
        # work through. It still takes focus, so it can be read line by line
        # before the run starts.
        self._recording_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        _fit_rows(self._recording_list, len(self._recordings))
        self._explain(self._recording_list, RECORDINGS)

        inner = QVBoxLayout(group)
        inner.addWidget(self._recording_list)
        return group

    def _build_configuration(self) -> QWidget:
        group = QGroupBox("About these recordings", self)
        self._configuration_group = group
        form = QFormLayout(group)
        # Long rows wrap onto a second line rather than forcing the dialog
        # wider than the screen at a large font size.
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self._afrikaans_box = QCheckBox("&Afrikaans may be spoken in these recordings", group)
        self._explain(self._afrikaans_box, AFRIKAANS)
        form.addRow(self._afrikaans_box)

        # Alt+S belongs to the Start button, so the speaker count takes Alt+N.
        speakers_label = QLabel("Expected &number of speakers", group)
        self._speaker_count_box = QSpinBox(group)
        self._speaker_count_box.setRange(
            MINIMUM_EXPECTED_SPEAKER_COUNT, MAXIMUM_EXPECTED_SPEAKER_COUNT
        )
        self._explain(self._speaker_count_box, SPEAKER_COUNT)
        speakers_label.setBuddy(self._speaker_count_box)
        form.addRow(speakers_label, self._speaker_count_box)

        names_label = QLabel("&Known speaker names", group)
        self._speaker_names_edit = QLineEdit(group)
        self._speaker_names_edit.setPlaceholderText("Separate names with commas")
        self._explain(self._speaker_names_edit, KNOWN_SPEAKERS)
        names_label.setBuddy(self._speaker_names_edit)
        form.addRow(names_label, self._speaker_names_edit)

        context_label = QLabel("Recording c&ontext", group)
        self._context_edit = QPlainTextEdit(group)
        # Three lines is enough for the sentence or two this wants, and it
        # grows with the system font rather than being fixed in pixels.
        self._context_edit.setFixedHeight(self.fontMetrics().lineSpacing() * 4)
        self._explain(self._context_edit, CONTEXT)
        context_label.setBuddy(self._context_edit)
        form.addRow(context_label, self._context_edit)
        return group

    def _build_vocabulary(self) -> QWidget:
        """The list of vocabulary profiles, each of which can be switched on.

        Check boxes in a list rather than a multiple selection, because a
        selection is lost the moment somebody arrows past it and a check box
        is not. A screen reader also reads a check state as a state, which is
        what this is, rather than as a highlight.
        """
        group = QGroupBox("Vocabulary", self)
        self._vocabulary_group = group

        label = QLabel("&Vocabulary profiles to use", group)
        self._profile_list = QListWidget(group)
        profiles = list(self._vocabulary.profiles)
        for profile in profiles:
            item = QListWidgetItem(f"{profile.display_name} ({profile.level.display_name})")
            item.setData(Qt.ItemDataRole.UserRole, profile.id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self._profile_list.addItem(item)
        if not profiles:
            # Said in the list rather than left as an empty box, so that
            # somebody who finds nothing here is told why and where to go.
            empty = QListWidgetItem(
                "No vocabulary profiles have been set up yet. They are created in "
                "Settings, under Vocabulary."
            )
            empty.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self._profile_list.addItem(empty)
        _fit_rows(self._profile_list, max(1, len(profiles)))
        # Ticking a profile changes what ElevenLabs charges, so the estimate
        # below it is redrawn.
        self._profile_list.itemChanged.connect(lambda _item: self._show_cost())
        self._explain(self._profile_list, PROFILES)
        label.setBuddy(self._profile_list)

        inner = QVBoxLayout(group)
        inner.addWidget(label)
        inner.addWidget(self._profile_list)
        self._build_learned_names(group, inner)
        return group

    def _build_learned_names(self, group: QGroupBox, inner: QVBoxLayout) -> None:
        """The names this folder's reviews have taught, with a way to forget one.

        A slip in a review teaches a wrong name, and this list is the only
        place to see it and take it out. It is a plain list with a selection
        rather than check boxes, because the one thing to do here is act on
        one name at a time.
        """
        # Alt+N is the speaker count and Alt+R the review window button, so the
        # list takes Alt+L and Remove takes Alt+M.
        label = QLabel("Names &learned in this folder", group)
        self._learned_list = QListWidget(group)
        self._learned_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._explain(self._learned_list, LEARNED_NAMES)
        label.setBuddy(self._learned_list)
        # Delete removes the selected name, as it would in a list in Explorer.
        # An event filter rather than a shortcut, so the key does this only
        # while the list has the focus and does not reach the dialog's default
        # button.
        self._learned_list.installEventFilter(self)

        self._remove_name_button = QPushButton("Re&move name", group)
        self._remove_name_button.setAutoDefault(False)
        describe(
            self._remove_name_button,
            "Remove name",
            "Forgets the name selected in the list of names learned in this folder. "
            "A later correction in a review teaches it again.",
        )
        self._note_keys[self._remove_name_button] = LEARNED_NAMES

        # What the last removal came to. It is on screen as well as spoken, so
        # a sighted user and a screen reader user are told the same thing. A
        # label carrying a message is never given a name of its own, because
        # its name would hide what it says.
        self._learned_status = QLabel("", group)
        self._learned_status.setWordWrap(True)

        self._fill_learned_names()
        _fit_rows(self._learned_list, self._learned_list.count())
        self._learned_list.itemSelectionChanged.connect(self._update_remove_name_button)
        self._remove_name_button.clicked.connect(self.remove_selected_name)
        self._update_remove_name_button()

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self._remove_name_button, 0)

        inner.addWidget(label)
        inner.addWidget(self._learned_list)
        inner.addLayout(row)
        inner.addWidget(self._learned_status)

    def _build_cost(self) -> QWidget:
        group = QGroupBox("What this run will cost", self)
        self._cost_group = group

        self._cost_display = QPlainTextEdit(group)
        self._cost_display.setReadOnly(True)
        self._cost_display.setFixedHeight(self.fontMetrics().lineSpacing() * 8)
        self._explain(self._cost_display, COST)

        inner = QVBoxLayout(group)
        inner.addWidget(self._cost_display)
        return group

    def _build_notes(self) -> QWidget:
        """The panel that explains whichever setting the focus is on.

        Every setting here needs more explaining than fits on its label, and
        more than is bearable to hear read out on every focus. A tooltip
        cannot carry it either: it needs a mouse, it times out, and it cannot
        be scrolled or copied.

        So the explanation goes in a plain read-only text box that follows the
        focus. Tab onto a setting and its note appears; tab into the box to
        read the note at leisure, where it stays put rather than changing
        under you.
        """
        group = QGroupBox("About this setting", self)
        self._notes_group = group

        self._notes_text = QPlainTextEdit(group)
        self._notes_text.setReadOnly(True)
        self._notes_text.setFixedHeight(self.fontMetrics().lineSpacing() * _NOTE_LINES)
        describe(
            self._notes_text,
            "About this setting",
            "Explains the setting the focus is on. Use the arrow keys to read through "
            "it. It stays on the last setting while you read, and the Guide button "
            "shows all of them together.",
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

    def _build_progress(self) -> QWidget:
        self._progress_group = QGroupBox("Progress", self)
        self._progress_bar = QProgressBar(self._progress_group)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        describe(
            self._progress_bar,
            "Overall progress",
            "How far the run has come through all the recordings, as a percentage.",
        )

        # These labels carry messages, so they are left unnamed. A label has
        # no accessible value of its own: its text is its name, and naming it
        # would hide what it says.
        self._progress_label = QLabel("Starting...", self._progress_group)
        self._progress_label.setWordWrap(True)
        self._stage_label = QLabel("", self._progress_group)
        self._stage_label.setWordWrap(True)

        inner = QVBoxLayout(self._progress_group)
        inner.addWidget(self._progress_bar)
        inner.addWidget(self._progress_label)
        inner.addWidget(self._stage_label)
        # Nothing has started, so there is nothing to show yet.
        self._progress_group.setVisible(False)
        return self._progress_group

    def _build_report(self) -> QWidget:
        self._report_group = QGroupBox("What was done", self)
        self._report_text = QPlainTextEdit(self._report_group)
        self._report_text.setReadOnly(True)
        self._report_text.setMinimumHeight(self.fontMetrics().lineSpacing() * 5)
        describe(
            self._report_text,
            "What was done",
            "One paragraph for each recording, saying how many words it came to, how "
            "many need review, and anything that went wrong. Use the arrow keys to "
            "read through it.",
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
            "Start transcribing",
            "Checks that everything the services need is set up, shows what the run "
            "will cost, and then begins. Everything else is switched off until the "
            "run has finished or you cancel it.",
        )
        self._start_button.setDefault(True)

        # These two are what there is to do once a run has finished, so they
        # are here from the start rather than appearing at the end. They are
        # switched off until there is a transcript for them to act on, which a
        # screen reader reads as unavailable: that says "not yet" where a
        # hidden button would say nothing at all.
        self._folder_button = self._buttons.addButton(
            "Open transcript &folder", QDialogButtonBox.ButtonRole.ActionRole
        )
        self._folder_button.setAutoDefault(False)
        self._folder_button.setEnabled(False)
        describe(
            self._folder_button,
            "Open transcript folder",
            "Opens the folder holding the transcript, the exports and the untouched "
            "service responses. Each recording has its own folder beside it; this "
            "opens the one for the recording named in the report.",
        )

        self._review_button = self._buttons.addButton(
            "Open &review window", QDialogButtonBox.ButtonRole.ActionRole
        )
        self._review_button.setAutoDefault(False)
        self._review_button.setEnabled(False)
        describe(
            self._review_button,
            "Open review window",
            "Closes this dialog and opens the review window on this folder, starting "
            "at the first recording with words waiting to be settled.",
        )

        self._close_button = self._buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        self._close_button.setAutoDefault(False)
        self._describe_close_button()
        return self._buttons

    def _explain(self, widget: QWidget, key: str) -> QWidget:
        """Name a control from its note, and have the panel follow it.

        The short summary becomes the accessible description and the tooltip,
        so a screen reader reads it on focus and a mouse user sees it on
        hover. The full note is left to the panel, because hearing several
        paragraphs every time the focus moves would be unusable.
        """
        note = note_for(key)
        describe(widget, note.title, note.summary)
        self._note_keys[widget] = key
        return widget

    def _connect_signals(self) -> None:
        self._guide_button.clicked.connect(self.show_guide)
        self._start_button.clicked.connect(self.start)
        self._folder_button.clicked.connect(self.open_transcript_folder)
        self._review_button.clicked.connect(self.open_review_window)
        self._close_button.clicked.connect(self.reject)

        # F1 asks for help on what is in front of you, everywhere else in
        # Windows, so it does here too.
        self._guide_shortcut = QShortcut(QKeySequence(Qt.Key.Key_F1), self)
        self._guide_shortcut.activated.connect(self.show_guide)

        self._runner.recordingStarted.connect(self._on_recording_started)
        self._runner.recordingFinished.connect(self._on_recording_finished)
        self._runner.progressChanged.connect(self._on_progress_changed)
        self._runner.runFinished.connect(self._on_run_finished)

    def showEvent(self, event) -> None:
        """Start watching the focus while the dialog is actually on screen."""
        super().showEvent(event)
        self._watch_focus(True)

    def hideEvent(self, event) -> None:
        """Stop watching as soon as it is not.

        The watch is on the application, which outlives this dialog by a long
        way. Left connected, every opening of Transcribe would add another
        watcher looking at every focus change anywhere in the application.
        Closing is also not the same as being closed: Cancel and Escape both
        go through reject, which hides the dialog without raising a close
        event at all.
        """
        self._watch_focus(False)
        super().hideEvent(event)

    def _watch_focus(self, watching: bool) -> None:
        """Follow the focus, or stop following it.

        The application is asked about the focus rather than each control
        being watched, because the widget that actually takes focus is often
        a part of a control rather than the control itself.
        """
        application = QApplication.instance()
        if application is None or watching == self._watching_focus:
            return
        self._watching_focus = watching
        if watching:
            application.focusChanged.connect(self._on_focus_changed)
        else:
            application.focusChanged.disconnect(self._on_focus_changed)

    def _on_focus_changed(self, _old: QWidget | None, new: QWidget | None) -> None:
        """Show the note for whatever now has the focus, if it has one.

        The search walks up from the focused widget rather than looking it up
        directly, because the widget that actually takes focus is often a part
        of the control rather than the control itself: a spin box hands it to
        the text field inside it. Walking up finds the control the user thinks
        they are on.

        Focus that belongs to nothing in particular, including the note panel
        itself, leaves the note where it is. That is what lets the note be
        read: moving into it to read it must not change it.
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
        TranscriptionGuideDialog(self).exec()

    # -- Names learned in this folder -------------------------------------

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Remove the selected learned name when Delete is pressed in its list."""
        if (
            watched is self._learned_list
            and event.type() == QEvent.Type.KeyPress
            and event.key() == Qt.Key.Key_Delete
            and event.modifiers() == Qt.KeyboardModifier.NoModifier
        ):
            self.remove_selected_name()
            return True
        return super().eventFilter(watched, event)

    def _fill_learned_names(self) -> None:
        """Fill the list from the project file as it is on disk now.

        The file is read afresh each time rather than kept, because the review
        window may be open on the same folder and learning names as it saves.
        An empty list, and a file that could not be read, are each said in the
        list itself rather than left as an empty box, so somebody who finds
        nothing here is told why.
        """
        self._learned_list.clear()
        names: list[LearnedName] = []
        status = JsonReadStatus.MISSING
        if self._folder is not None:
            names, status = read_learned_names(self._folder)
        for name in names:
            item = QListWidgetItem(_learned_name_line(name))
            item.setData(Qt.ItemDataRole.UserRole, (name.text, name.language))
            self._learned_list.addItem(item)
        if names:
            return
        if status is JsonReadStatus.DAMAGED:
            text = (
                "The project file in this folder could not be read, so its learned "
                "names cannot be shown."
            )
        else:
            text = "No names have been learned in this folder yet."
        # Selectable, so it can be landed on and read like any other row, but
        # it carries no name, so Remove stays off while it is selected.
        empty = QListWidgetItem(text)
        empty.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        self._learned_list.addItem(empty)

    def _selected_learned_name(self) -> tuple[str, str] | None:
        """The text and language of the name selected in the list, if any."""
        for item in self._learned_list.selectedItems():
            chosen = item.data(Qt.ItemDataRole.UserRole)
            if chosen:
                return chosen[0], chosen[1]
        return None

    def _update_remove_name_button(self) -> None:
        """Switch Remove on only while a real name is selected.

        A screen reader reads a switched-off button as unavailable, which says
        "select a name first" where a button that did nothing would not.
        """
        self._remove_name_button.setEnabled(self._selected_learned_name() is not None)

    def remove_selected_name(self) -> bool:
        """Forget the selected learned name, and say so. Returns whether it went.

        Only that one name is taken out of the project file as it is on disk
        now, and the file is saved at once. A copy loaded when the dialog
        opened is never written back, because the review window may have
        learned other names since, and writing the old copy would lose them.

        The focus goes to the list, on the row that took the removed name's
        place, so a screen reader reads where the person now is, whether they
        pressed Delete in the list or the Remove button, which may now be off.
        """
        chosen = self._selected_learned_name()
        if chosen is None or self._folder is None:
            return False
        text, language = chosen
        row = self._learned_list.currentRow()

        removed = remove_learned_name(self._folder, text, language)
        if not removed:
            names, status = read_learned_names(self._folder)
            still_there = any(
                name.key == LearnedName(text=text, language=language).key for name in names
            )
            if status is not JsonReadStatus.READ or still_there:
                message = (
                    f"{text} could not be removed, because the project file in this "
                    "folder could not be read or saved. Nothing was changed."
                )
                self._learned_status.setText(message)
                announce(self._learned_list, message, urgent=True)
                return False

        self._fill_learned_names()
        self._learned_list.setCurrentRow(max(0, min(row, self._learned_list.count() - 1)))
        self._update_remove_name_button()
        self._learned_list.setFocus(Qt.FocusReason.OtherFocusReason)
        if removed:
            message = f"{text} removed. It will be learned again if you correct it in a review."
        else:
            message = f"{text} had already been removed from the project file."
        self._learned_status.setText(message)
        announce(self._learned_list, message)
        return removed

    # -- What the user chose ----------------------------------------------

    def _show_defaults(self) -> None:
        """Fill the configuration in from the settings, as a starting point."""
        processing = self._settings.processing
        self._afrikaans_box.setChecked(processing.default_afrikaans_enabled)
        self._speaker_count_box.setValue(processing.default_expected_speaker_count)

    def chosen_profile_ids(self) -> list[str]:
        """The vocabulary profiles the user switched on, in the order listed."""
        chosen: list[str] = []
        for row in range(self._profile_list.count()):
            item = self._profile_list.item(row)
            profile_id = item.data(Qt.ItemDataRole.UserRole)
            if profile_id and item.checkState() == Qt.CheckState.Checked:
                chosen.append(str(profile_id))
        return chosen

    def chosen_configuration(self) -> RecordingConfiguration:
        """What the user said about these recordings."""
        return RecordingConfiguration(
            afrikaans_enabled=self._afrikaans_box.isChecked(),
            expected_speaker_count=self._speaker_count_box.value(),
            known_speakers=_split_names(self._speaker_names_edit.text()),
            recording_context=self._context_edit.toPlainText().strip(),
            vocabulary_profile_ids=self.chosen_profile_ids(),
        )

    def chosen_options(self) -> PipelineOptions:
        """What the user chose, in the form the pipeline wants."""
        return PipelineOptions(
            configuration=self.chosen_configuration(),
            settings=self._settings,
            vocabulary=self._vocabulary,
            calibration_path=calibration_file_path(),
            folder_terms=self._folder_terms,
        )

    # -- What it will cost -------------------------------------------------

    def services_that_will_run(self) -> list[Provider]:
        """The services this run will send whole recordings to.

        Only the full-pass services are counted. AssemblyAI is asked about
        short windows around disputed words rather than about whole
        recordings, so charging it for the whole length would overstate the
        estimate considerably, and Deepgram is not part of a full pass at all.
        Both still appear in the sentence about what cannot be predicted.
        """
        switched_on = {
            Provider.ELEVENLABS: self._settings.elevenlabs.enabled,
            Provider.OPENAI: self._settings.openai_transcription.enabled,
            Provider.MICROSOFT: self._settings.microsoft.enabled,
            Provider.ASSEMBLYAI: self._settings.assemblyai.enabled,
            Provider.DEEPGRAM: self._settings.deepgram.enabled,
        }
        return [provider for provider in DEFAULT_FULL_PASS_PROVIDERS if switched_on.get(provider)]

    def known_duration_seconds(self) -> float:
        """How much audio the estimate can actually be worked out from."""
        return sum(
            recording.duration_seconds or 0.0
            for recording in self._recordings
            if recording.duration_seconds
        )

    def recordings_without_a_duration(self) -> list[AudioFile]:
        """The recordings whose length could not be read, and so are not costed."""
        return [
            recording for recording in self._recordings if not recording.duration_seconds
        ]

    def cost_estimate(self) -> CostEstimate:
        """What the services are expected to charge for this run.

        Adjudication is deliberately left at nothing rather than guessed at.
        How many disputes a recording produces depends on how much the
        services disagree, which cannot be known until they have answered, and
        a made-up number presented as a figure would be worse than saying
        plainly that it cannot be predicted.
        """
        return estimate_cost(
            self.known_duration_seconds(),
            self.services_that_will_run(),
            rates=self._settings.cost,
            vocabulary_terms_sent=self.vocabulary_terms_will_be_sent(),
        )

    def vocabulary_terms_will_be_sent(self) -> bool:
        """Whether the folder's learned names or the chosen profiles hold any term.

        ElevenLabs charges more for a request that carries terms, so the
        estimate has to know. A profile that is switched on but empty sends
        nothing and costs nothing extra.
        """
        if self._folder_terms:
            return True
        return bool(resolve_terms(self._vocabulary, self.chosen_profile_ids()))

    def cost_text(self) -> str:
        """The estimate written out: a line per service, then the whole story."""
        estimate = self.cost_estimate()
        lines = [f"Total audio length: {describe_duration(estimate.duration_seconds)}."]

        if not estimate.per_provider:
            lines.append(
                "No transcription service is switched on, so nothing would be sent and "
                "no transcript could be made."
            )
        for cost in estimate.per_provider:
            if cost.is_priced:
                lines.append(
                    f"{cost.provider.display_name}: about "
                    f"{describe_money(cost.amount or 0.0, estimate.currency)}."
                )
            else:
                lines.append(
                    f"{cost.provider.display_name}: unpriced, because no rate has been "
                    "entered for it in Settings. It is not in the total."
                )
        if any(cost.is_priced for cost in estimate.per_provider):
            lines.append(
                f"Estimated total: {describe_money(estimate.total, estimate.currency)}."
            )

        unmeasured = self.recordings_without_a_duration()
        if unmeasured:
            names = ", ".join(recording.name for recording in unmeasured)
            lines.append(
                f"The length of {_plural(len(unmeasured), 'recording')} could not be "
                f"read, so they are not in the figure above and the real cost will be "
                f"higher: {names}."
            )

        lines.append("")
        lines.append(estimate.summary)
        lines.append(
            "The rates are typed into Settings by hand, and services change their "
            "prices without telling us, so check them there before relying on this."
        )
        return "\n".join(lines)

    def _show_cost(self) -> None:
        self._cost_display.setPlainText(self.cost_text())
        self._cost_display.moveCursor(QTextCursor.MoveOperation.Start)

    def confirm_cost(self, message: str) -> bool:
        """Ask whether to spend the money, and mean the question.

        Cancel is the default button rather than Yes. This dialog is opened
        with the keyboard and Start is its default button, so an Enter meant
        for Start would otherwise carry straight through the confirmation and
        spend the money without anybody having read it.
        """
        box = QMessageBox(self)
        box.setWindowTitle("Transcribe")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText("Start transcribing these recordings?")
        box.setInformativeText(message)
        box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
        )
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        return box.exec() == QMessageBox.StandardButton.Yes

    def confirm_without_learned_names(self) -> bool:
        """Ask whether to go on when the folder's project file cannot be read.

        A message box is announced by screen readers when it opens and is
        answered from the keyboard. Cancel is the default, for the reason
        :meth:`confirm_cost` gives: an Enter meant for Start must not carry
        through and start a run the person has not agreed to.
        """
        path = (self._folder / PROJECT_FILE_NAME) if self._folder else PROJECT_FILE_NAME
        box = QMessageBox(self)
        box.setWindowTitle("Transcribe")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText("The names this folder learned in its reviews could not be read.")
        box.setInformativeText(
            f"The project file {path} is damaged, or another program has it open. "
            "Go on and transcribe without the learned names, or cancel and try "
            "again later?"
        )
        go_on = box.addButton("Go on without learned names", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        return box.clickedButton() is go_on

    def _read_folder_terms(self) -> tuple[tuple[VocabularyTerm, ...], JsonReadStatus]:
        """The folder's learned names as terms, and what reading the file found."""
        if self._folder is None:
            return (), JsonReadStatus.MISSING
        names, status = read_learned_names(self._folder)
        return tuple(terms_from_learned_names(names)), status

    def _say_not_started(self) -> None:
        self._set_progress_text("The run was not started.")
        self._progress_group.setVisible(True)
        self._grow_to_fit()
        announce(self._progress_bar, "The run was not started.", urgent=True)

    # -- Running ----------------------------------------------------------

    @property
    def folder(self) -> Path | None:
        """The folder the recordings sit in, or ``None`` when there are none."""
        return self._folder

    @property
    def is_running(self) -> bool:
        return self._runner.is_running

    @property
    def summary(self) -> RunSummary | None:
        """What the last run came to, or ``None`` if none has finished."""
        return self._summary

    @property
    def review_request(self) -> RecordingOutcome | None:
        """The recording the user asked to review, once they have asked.

        The review window is not opened from here. This dialog is modal and
        owns the thread that made the transcript, so opening a second window
        on top of it would put the review window behind a dialog that cannot
        be dismissed. The request is recorded instead, this dialog closes, and
        the main window opens the review.

        What the main window opens is the review of the whole folder. This
        recording is where the person is put down in it, not the extent of
        what they are shown.
        """
        return self._review_request

    def start(self) -> bool:
        """Begin the run, unless something stops it. Returns whether it began."""
        if self.is_running:
            return False
        if not self._recordings:
            self._refuse("There are no recordings to transcribe.")
            return False

        problems = self.preflight_problems()
        if problems:
            self._refuse_with_problems(problems)
            return False

        # The folder's learned names are read again, because a review may have
        # added some since the dialog opened. A file that is there but cannot
        # be read is asked about rather than quietly passed over.
        folder_terms, status = self._read_folder_terms()
        if status is JsonReadStatus.DAMAGED:
            if not self.confirm_without_learned_names():
                self._say_not_started()
                return False
            folder_terms = ()
        self._folder_terms = folder_terms

        # Worked out again here rather than trusted from when the dialog
        # opened, because Settings may have been changed in between.
        self._show_cost()
        if self._settings.cost.confirm_before_running and not self.confirm_cost(self.cost_text()):
            self._say_not_started()
            return False

        self._results = []
        self._summary = None
        self._review_request = None
        self._cancel_requested = False
        self._detached = False
        self._report_text.clear()
        self._report_group.setVisible(False)
        self._folder_button.setEnabled(False)
        self._review_button.setEnabled(False)
        self._progress_bar.setValue(0)
        self._progress_group.setVisible(True)
        self._grow_to_fit()
        self._set_progress_text(f"Starting on {_plural(len(self._recordings), 'recording')}.")
        self._set_stage_text("")
        # The focus is moved off the controls before they are switched off,
        # rather than after. A control that is disabled while it holds the
        # focus takes the focus with it, and the user is left nowhere.
        self._close_button.setFocus(Qt.FocusReason.OtherFocusReason)
        self._set_controls_enabled(False)

        paths = [recording.path for recording in self._recordings]
        if not self._runner.start(paths, self.chosen_options()):
            self._set_controls_enabled(True)
            self._describe_close_button()
            return False
        # Only now is the run really going, so only now does the Cancel button
        # mean "stop the run" rather than "close the dialog".
        self._describe_close_button()
        announce(
            self._progress_bar,
            f"Transcribing {_plural(len(self._recordings), 'recording')}.",
            urgent=True,
        )
        return True

    def preflight_problems(self) -> list[str]:
        """Everything that would stop this run, found before any money is spent.

        Three things are checked, and the order is the order in which they
        are cheap. The settings are asked whether every service that is on
        has its key and model. Then the vendor library behind each of those
        services is actually imported, because a package that was never
        installed is not found until the service is first called, which is
        after the recording has been prepared and the other services have
        been sent it. Then the first recording's transcript folder is made
        and written to, with a tiny file that is removed again, because a
        read-only folder or a path past the Windows limit would otherwise be
        discovered at the moment the finished transcript was being saved,
        an hour of paid answers too late.
        """
        problems = self._settings.missing_requirements()
        problems.extend(self._settings.missing_libraries())
        if self._recordings:
            store = TranscriptStore(
                self._recordings[0].path, self._settings.processing.transcript_folder_suffix
            )
            fault = store.probe_writable()
            if fault is not None:
                problems.append(fault)
        return problems

    def cancel(self) -> None:
        """Ask a running transcription to stop as soon as it can.

        Stopping is not instant. The flag is checked between stages, and a
        request already with a service can take many minutes to come back,
        during which the dialog would otherwise refuse to close at all. So
        the first Cancel asks the run to stop, and says so; a second Cancel
        closes the dialog and leaves the run to stop in the background, which
        :meth:`reject` explains.
        """
        if not self.is_running:
            return
        self._runner.cancel()
        self._cancel_requested = True
        self._describe_close_button()
        message = (
            "Stopping. The run stops at the end of the request it is waiting on, "
            "which can take a few minutes. Requests already sent are still charged "
            "for. Press Cancel again to close this window and let it stop in the "
            "background."
        )
        self._set_progress_text(message)
        announce(self._progress_bar, message, urgent=True)

    @property
    def is_stopping_in_background(self) -> bool:
        """Whether the dialog was closed on a run that had not yet stopped."""
        return self._detached and self.is_running

    def _set_controls_enabled(self, enabled: bool) -> None:
        """Switch everything except Cancel off while a run is going."""
        for widget in (
            self._recording_group,
            self._configuration_group,
            self._vocabulary_group,
            self._start_button,
        ):
            widget.setEnabled(enabled)

    def _describe_close_button(self) -> None:
        """Name the button for what it does now, which changes as the run does."""
        if self.is_running and self._cancel_requested:
            text, name = "&Cancel", "Cancel"
            description = (
                "The run has been asked to stop and is waiting on the request it "
                "already sent. Press this again to close the window now and let the "
                "run stop in the background. Recordings already transcribed are kept."
            )
        elif self.is_running:
            text, name = "&Cancel", "Cancel"
            description = (
                "Stops the run. Recordings already transcribed are kept, and the one "
                "in progress is abandoned. Requests already sent are still charged for."
            )
        elif self._summary is not None:
            text, name = "&Close", "Close"
            description = "Closes this dialog."
        else:
            text, name = "&Cancel", "Cancel"
            description = "Closes this dialog without transcribing anything."
        self._close_button.setText(text)
        describe(self._close_button, name, description)

    # -- What the run reports ---------------------------------------------

    def _on_recording_started(self, name: str, number: int, total: int) -> None:
        if self._detached:
            # Nobody is looking at a closed dialog, and an announcement
            # raised from a hidden window lands on top of whatever the person
            # is doing now, about a run they have already walked away from.
            return
        message = f"Transcribing {name}. Recording {number} of {total}."
        self._set_progress_text(message)
        # Each recording is announced as it starts. The progress bar has a
        # value a screen reader can read, but only if the user goes and asks
        # for it; this is the pace at which something actually changes.
        announce(self._progress_bar, message)

    def _on_progress_changed(self, percentage: int, stage: str) -> None:
        if self._detached:
            return
        self._progress_bar.setValue(percentage)
        # The bar moves far more often than the stage changes, and a screen
        # reader repeating the same sentence every second or two would drown
        # out everything else the user is doing.
        if stage == self._announced_stage:
            return
        self._set_stage_text(stage)
        # The sentence is announced as well as shown. A run takes minutes, and
        # a user who hears nothing for four of them has no way to tell a slow
        # service from a stuck application.
        announce(self._progress_bar, stage)

    def _on_recording_finished(self, outcome: RecordingOutcome) -> None:
        self._results.append(outcome)
        self._show_report()

    def _on_run_finished(self, summary: RunSummary) -> None:
        self._summary = summary
        self._results = list(summary.results)
        self._show_report()
        if not summary.cancelled:
            self._progress_bar.setValue(100)

        message = summarise(summary)
        self._set_progress_text(message)
        self._set_stage_text("")
        self._set_controls_enabled(True)
        self._describe_close_button()

        chosen = self.outcome_to_open()
        self._folder_button.setEnabled(chosen is not None)
        self._review_button.setEnabled(chosen is not None)
        if chosen is not None and summary.review_count:
            # The review button is the one thing worth doing next, so it takes
            # the focus rather than Close.
            self._review_button.setFocus(Qt.FocusReason.OtherFocusReason)
        else:
            self._close_button.setFocus(Qt.FocusReason.OtherFocusReason)
        if self._detached:
            # The dialog was closed on a run still stopping. Nobody is looking
            # at it, and a message box raised from a hidden window would land
            # on top of whatever the person is doing now, with the main
            # window's status bar already saying what became of the run.
            _log.info("A run closed in the background has finished: %s", message)
            self.detachedRunFinished.emit(summary)
            return
        self._come_to_the_front()
        self._show_completion_message(message)

    def _come_to_the_front(self) -> None:
        """Get the person's attention, wherever they went during the run.

        A run lasts long enough for anybody to go and do something else, and
        a message box that opens behind another program's window is a message
        box nobody sees for an hour. The taskbar entry is made to flash, and
        the dialog is brought forward so that the box that follows takes the
        focus and is read out. Windows does not always allow a window to take
        the foreground from another program, which is what the flash is for.
        """
        application = QApplication.instance()
        if application is not None:
            QApplication.alert(self)
        self.activateWindow()
        self.raise_()

    def _show_completion_message(self, message: str) -> None:
        """Say plainly that the run is over, and how it went.

        A message box is used because it takes the focus and every screen
        reader reads it without being asked. The same words stay on the dialog
        behind it, and the detail stays in the report, so nothing is lost by
        dismissing it.
        """
        box = QMessageBox(self)
        box.setWindowTitle("Transcribe")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText(message)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

    def _show_report(self) -> None:
        """Write out what has happened so far, one paragraph per recording.

        The whole report is rewritten each time rather than added to, so that
        it says the same thing whether it is read part way through a run or
        after the summary has arrived.
        """
        paragraphs = [outcome.message for outcome in self._results]
        chosen = self.outcome_to_open()
        # Only once the run has stopped, because until then the two buttons
        # are switched off and which recording they will act on can still
        # change as later ones finish.
        if self._summary is not None and chosen is not None:
            paragraphs.append(
                f"The transcript folder and review window buttons act on {chosen.name}. "
                "The review covers the whole folder and simply starts there."
            )
        self._report_text.setPlainText("\n\n".join(paragraphs) or "Nothing was transcribed.")
        self._report_group.setVisible(True)
        self._grow_to_fit()

    def _set_progress_text(self, message: str) -> None:
        self._progress_label.setText(message)

    def _set_stage_text(self, message: str) -> None:
        self._stage_label.setText(message)
        self._announced_stage = message

    def _refuse(self, message: str, spoken: str | None = None) -> None:
        """Report why the run cannot start, on screen and out loud.

        ``spoken`` replaces ``message`` in the announcement when the screen
        reader should hear more than the label shows.
        """
        self._progress_group.setVisible(True)
        self._grow_to_fit()
        self._set_progress_text(message)
        announce(self._progress_bar, spoken or message, urgent=True)

    def _refuse_with_problems(self, problems: list[str]) -> None:
        """Refuse to start, and say exactly what is missing.

        Refusing costs nothing; starting would cost real money. A run begun
        without an API key does not fail cleanly at the beginning: the
        services that are configured answer and are charged for, and the run
        then has nothing usable to show for it.

        The whole list goes in the report box, because several missing keys
        are several sentences, and a text box can be read back a line at a
        time where a spoken announcement cannot. What is announced is the
        count and the first problem in full, so a single missing key is heard
        by name without leaving Start. With more than one, the announcement
        also says how many there are and that the "What was done" box lists
        them all. Focus stays on Start rather than jumping to that box: moving
        it would make the screen reader read the box's name and description
        over the announcement, and Start is where the person returns once the
        problems are put right.
        """
        # "First" rather than "in Settings first": a missing key is put right
        # in Settings, but a library that will not load is put right by
        # running the launcher again, and a folder that cannot be written to
        # is put right in Explorer. Each sentence says which.
        headline = (
            f"The run cannot start. {_plural(len(problems), 'thing')} must be put right "
            "first."
        )
        lines = [headline, ""]
        lines.extend(problems)
        if any(problem.startswith("ElevenLabs") for problem in problems):
            lines.append("")
            lines.append(
                "ElevenLabs Scribe is the backbone of every transcript. Without it "
                "there are no word timings and no initial speaker labels, so there is "
                "nothing to play back and nobody to attribute the words to. It is the "
                "one service a run genuinely cannot be made without."
            )
        self._report_text.setPlainText("\n".join(lines))
        self._report_group.setVisible(True)
        if len(problems) == 1:
            spoken = f"{headline} {problems[0]}"
        else:
            spoken = (
                f"{headline} The first is: {problems[0]} All {len(problems)} are listed, "
                "one to a line, in the What was done box."
            )
        self._refuse(headline, spoken)

    # -- What to do with the result ---------------------------------------

    def outcome_to_open(self) -> RecordingOutcome | None:
        """Which finished recording the two result buttons act on.

        The first one with words waiting to be settled, because that is the
        one with work left to do on it. Failing that the first transcript,
        because a run where nothing needs reviewing is still worth looking at.
        The report names it, so the buttons are never acting on a recording
        the user has not been told about.
        """
        finished = [outcome for outcome in self._results if outcome.succeeded]
        for outcome in finished:
            if outcome.review_count:
                return outcome
        return finished[0] if finished else None

    def open_transcript_folder(self) -> None:
        """Open the transcript folder of the recording the report names."""
        chosen = self.outcome_to_open()
        if chosen is None:
            self._refuse("There is no transcript to open yet.")
            return
        folder = chosen.transcript_folder
        if folder is None:
            folder = TranscriptStore(
                chosen.path, self._settings.processing.transcript_folder_suffix
            ).folder
        if QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
            self._set_progress_text(f"Opened {folder}.")
            announce(self._progress_bar, f"Opened the transcript folder for {chosen.name}.")
        else:
            self._refuse(f"Windows could not open the transcript folder, {folder}.")

    def open_review_window(self) -> None:
        """Ask for the review window, and close so that it can be opened."""
        chosen = self.outcome_to_open()
        if chosen is None:
            self._refuse("There is no transcript to review yet.")
            return
        self._review_request = chosen
        self.accept()

    # -- Closing ----------------------------------------------------------

    def reject(self) -> None:
        """Cancel a run rather than closing on top of it, once.

        The dialog owns the thread doing the work, so closing while it runs
        leaves that thread reporting into a window that has gone. Escape and
        the Cancel button therefore stop the run first, and the dialog stays
        open to say how it went.

        But stopping can take minutes. The run only looks at the flag between
        stages, and a request already with a service cannot be recalled, so a
        dialog that refused to close until then would hold the person in
        front of it for as long as the slowest service took. The second press
        closes the dialog and leaves the run to stop by itself. That is safe
        because the runner already tolerates a receiver that has gone: each
        result it tries to send is dropped with a line in the log rather
        than raised, and the recordings already transcribed are on the disk
        where they were saved. The main window says that the run is still
        stopping, so nothing is closed in silence.
        """
        if self.is_running and not self._cancel_requested:
            self.cancel()
            return
        if self.is_running:
            self._detach()
        super().reject()

    def closeEvent(self, event) -> None:
        """Answer the close box the same way as Cancel, and never join a live run.

        Closing the window is the same decision as pressing Cancel, so it is
        answered the same way: the first attempt stops the run, the second
        closes on top of it. Once nothing is running the thread is stopped
        and waited for before the dialog goes, because it reports through
        signals on an object that is about to be destroyed; a run that has
        been asked to stop and is still waiting on a service is not waited
        for here, because that wait is the whole reason the second press
        exists.
        """
        if self.is_running and not self._cancel_requested:
            self.cancel()
            event.ignore()
            return
        if self.is_running:
            self._detach()
        else:
            self._runner.stop()
        super().closeEvent(event)

    def _detach(self) -> None:
        """Let go of a run that is still stopping, so the dialog can close.

        The runner keeps going until its request comes back and then sends a
        summary nobody is waiting for. The application's own shutdown still
        stops it, through the runner's connection to aboutToQuit, and that
        wait is bounded to a few seconds there too.

        Closing the window reaches here twice, once from the close event and
        once more from :meth:`reject` underneath it, so a second call does
        nothing: connecting the clean-up twice would delete the dialog twice.
        """
        if self._detached:
            return
        self._detached = True
        self._runner.cancel()
        # The dialog gets rid of itself once the summary has arrived, rather
        # than being deleted by whoever opened it. The runner is its child,
        # and deleting the runner while its thread is still reporting would
        # take away the object that withdraws the request keeping the machine
        # awake. Connected after the dialog's own slot, so that it runs last.
        self._runner.runFinished.connect(lambda _summary: self.deleteLater())
        _log.info(
            "The Transcribe dialog was closed while the run was still stopping; the "
            "run will stop in the background."
        )


def _recording_line(recording: AudioFile) -> str:
    """One row of the list: the file name, and how long it is if that is known.

    The length is on the row because it is what the cost is worked out from.
    A user looking at an estimate they think is too high should be able to see
    which recording is responsible without leaving the dialog.
    """
    if recording.duration_seconds:
        return f"{recording.name}, {describe_duration(recording.duration_seconds)}"
    return f"{recording.name}, length unknown"


def _learned_name_line(name: LearnedName) -> str:
    """One row of the learned names: the name, its language, how it was misheard.

    For example "Bosch (English), also heard as Bosh". It is written as one
    sentence so that a screen reader reads the whole row as it would read it
    aloud, rather than as columns to be worked out.
    """
    language = language_name(name.language)
    if language == NO_LANGUAGE:
        line = f"{name.text} (language not known)"
    else:
        line = f"{name.text} ({language})"
    forms = [form for form in name.wrong_forms if form.strip()]
    if not forms:
        return line
    if len(forms) == 1:
        heard = forms[0]
    else:
        heard = f"{', '.join(forms[:-1])} and {forms[-1]}"
    return f"{line}, also heard as {heard}"


def _fit_rows(widget: QListWidget, count: int) -> None:
    """Make a list as tall as its contents, up to a few rows, then let it scroll.

    The height is worked out from the height of a real row, so it follows the
    system font rather than assuming a size somebody's text scaling will make
    wrong.
    """
    row_height = widget.sizeHintForRow(0)
    if row_height <= 0:
        return
    rows = min(max(1, count), _VISIBLE_ROWS)
    widget.setFixedHeight(rows * row_height + 2 * widget.frameWidth() + 4)


def _split_names(text: str) -> list[str]:
    """Turn a comma-separated list of names into a list, ignoring the gaps."""
    return [name.strip() for name in text.split(",") if name.strip()]


def _plural(number: int, noun: str) -> str:
    """Say "1 recording" or "3 recordings", so counts read as English."""
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"
