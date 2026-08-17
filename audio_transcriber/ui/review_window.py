"""The window for working through everything a folder could not settle.

A blended transcript is not a list of words with one confidence number
against each of them. Every word carries three separate answers, to three
separate questions, from three possibly different services: what was said,
when it was said, and who said it. This window is where a person settles
the ones that could not be settled without them, and it is built around
that separation rather than hiding it. The detail panel shows the text, the
timing and the speaker as three things, and the corrections for them are
separate actions, because correcting a spelling is not a statement about
the clock and never should be.

**The unit of work is a word across a folder, not a word in a file.** A
surname the services are unsure of is unsure in the same way in all eleven
recordings of the same meeting, and deciding it eleven times is eleven
chances to decide it differently. So the left of the window holds two
lists. ``Word Groups`` is the words, gathered across every recording, and it
is where a spelling is decided once. ``Occurrences`` is where that word
actually falls, one row per moment in one recording, and it is where the
speaker and the timing are decided, one occurrence at a time, because those
genuinely do differ between one mention and the next.

That split is the reason a group-wide replacement is safe to offer at all.
It changes text in files the person has not opened, and it is allowed to do
that only because it changes nothing else about them: not the speaker, not
the timing, not the language. Every message the window announces says so.

**Nothing here has a Save button.** A correction is written through the
moment it is made: the transcript through the ``save_correction`` callback,
and the project's own record of the review through
:meth:`~audio_transcriber.transcription.project.ProjectStore.save`. A save
that fails is announced urgently and never passes quietly, because the
person believes their work is kept and would otherwise find out after
closing the window.

**A replacement is always applied to what a word originally said.** The
occurrence remembers the form the services produced and never has it
overwritten, so editing a replacement twice gives the same result as
setting it correctly the first time, and so the rules the project learns
are keyed on the spellings a service actually produces rather than on the
spelling this window last wrote.

Playing the audio is the heart of the screen, so it happens by itself: the
highlight moving to an occurrence starts it, without taking the focus out
of the list. Two things stop that becoming noise. It waits a moment and is
cancelled if the highlight moves again, so holding the Down arrow through
eight occurrences does not queue eight clips. And because this window
changes audio file constantly, and Qt loads a file in the background, a
seek issued straight after ``load`` would silently do nothing at all: the
seek waits for the media to be ready. That is the one thing in here most
likely to work on a fast machine and fail on a slow one.

**Transcripts are read one at a time, when they are wanted, and let go
of.** The window is given a list of recording names and a function that
reads one, rather than a dictionary of all of them, and the difference is
not a stylistic one. A transcript of an hour of speech with four services
behind it is about 27 MB on disk and about 31 MB of live Python objects. A
folder of fifty such recordings was measured at 37 seconds to read and
about 1.6 GB held for as long as the window stayed open. That is what
pressing Ctrl+R would have cost, and nobody will sit through it.

Nothing on the two lists needs a transcript at all: the project state
already holds every word, file, time, confidence and decision they show.
That is true of the second block as well as the first, and it did not use
to be. The words the pipeline flagged are a property of a transcript, so
the first version of this window found them by reading one, which meant
that they appeared only for the recordings something had happened to read
and vanished from every later opening of the same folder. They are now
saved in the project as
:class:`~audio_transcriber.transcription.project.FlaggedItem`, written
whenever a recording is read for the analysis and read back when the window
opens, for exactly the reason the occurrences are.

A transcript is wanted for exactly two things, and both are narrow. One is
the occurrence the person is looking at, whose candidates, timing, speaker
and corrections all come from it, and which is kept because they will work
through several occurrences of the same recording and re-reading 27 MB on
every arrow key is its own kind of slow. The other is a run of the
analysis, which walks the folder one recording at a time and holds none of
it.

So there is no dictionary of transcripts here, and building one at the top
of the constructor would restore the whole cost while looking like a
simplification. A word this window changes reaches its transcript through
:meth:`ReviewWindow._transcript`, and a change that spans several files is
made a recording at a time so that only ever one of them is in memory.

The window knows nothing about providers, reconciliation or how a
transcript is stored. It takes the recording names, a way to read one, the
audio paths, a project store, something that can play audio, and a callback
to hand corrected transcripts back to. That is what lets the whole screen
be tested without a network, an API key or an audio file.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import (
    QAbstractTableModel,
    QEvent,
    QModelIndex,
    QObject,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QAction, QKeySequence, QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from audio_transcriber.audio.player import AudioPlayer
from audio_transcriber.formatting import spoken_duration
from audio_transcriber.transcription.confidence import strength_display
from audio_transcriber.transcription.grouping import (
    affected_summary,
    note_rule_applied,
    reprocess,
)
from audio_transcriber.transcription.model import (
    AlignmentStatus,
    AudioSpan,
    Confidence,
    FinalToken,
    Provider,
    ProviderToken,
    ReviewReason,
    ReviewStatus,
    RiskCategory,
    TimingStatus,
    TokenReference,
    Transcript,
)
from audio_transcriber.transcription.project import (
    MAXIMUM_AUTO_PLAY_DELAY_SECONDS,
    Occurrence,
    ProjectStore,
    ReplacementRule,
    WordGroup,
)
from audio_transcriber.ui.accessibility import announce, describe
from audio_transcriber.ui.review_lists import (
    GroupRow,
    OccurrenceModel,
    OccurrenceView,
    WordGroupModel,
    WordGroupView,
    build_group_rows,
    flagged_in,
    group_count_text,
    group_key,
    language_name,
    loose_key,
    low_confidence_count,
    restored_token,
    spoken_group_summary,
    spoken_occurrence_summary,
    time_display,
)
from audio_transcriber.ui.review_queue import (
    ReviewQueueModel,
    chosen_text,
    reason_text,
)

#: How much audio to add either side of a word when it is played. A word on
#: its own is not judgeable: the sentence around it is what tells a person
#: whether "fifteen" or "fifty" is the one that makes sense.
CONTEXT_SECONDS = 3.0

#: The same again, for the second play button, when three seconds turned
#: out not to be enough to place what was being talked about.
WIDE_CONTEXT_SECONDS = 12.0

#: Said when the wait is set to nothing at all, which is a real choice and not
#: a mistake: somebody working by eye, with no screen reader running, has
#: nothing for the clip to talk over and wants the audio the moment they
#: arrive.
NO_AUTO_PLAY_DELAY = "The audio starts as soon as you arrive."

NOT_ATTRIBUTED = "Not attributed to a service"
NOT_REPORTED = "Not reported"
NO_SPEAKER = "No speaker attributed"
NO_RISK = "None recorded"
NO_LLM_DECISION = "The language model was not consulted."
NO_LANGUAGE_EVIDENCE = "No language evidence was recorded."
NO_TIMING = "No timing at all"
NO_ITEM_SELECTED = "No item selected"
NO_AUDIO = "This word has no audio behind it, so there is nothing to play."
NO_CANONICAL_AUDIO = "There is no audio file for this recording, so nothing can be played."

#: Said when the word an occurrence points at is not in the transcript any
#: more, which happens when the recording was transcribed again and the word
#: could not be found. The occurrence is still shown, because a decision that
#: could not be kept is something the person has to be able to see; what it
#: cannot do is be corrected, since there is nothing left to correct.
WORD_NOT_IN_TRANSCRIPT = (
    "This word is no longer in the transcript, so it cannot be corrected. "
    "Process the low confidence words again to look for it."
)

#: Said when the transcript itself could not be read just now. Kept apart
#: from the message above, because the two are different news: one says the
#: word has gone, which is a fact about the recording, and this one says we
#: could not look, which says nothing about the word at all. Telling a person
#: their decision has been lost when the file was merely busy would be a
#: worse mistake than saying nothing.
TRANSCRIPT_NOT_READABLE = (
    "The transcript of {name} could not be read just now, so this word cannot be "
    "corrected. Nothing you have decided about it has been lost."
)

NO_WORD_SELECTED = "No word is selected."
NOTHING_TO_PROCESS = (
    "There are no transcripts in this folder, so there is nothing to process."
)

IN_VOCABULARY = "In the vocabulary"
NOT_IN_VOCABULARY = "Not in the vocabulary"
CURRENT_CHOICE = "The current choice"
NOT_CHOSEN = "Not chosen"
THE_APPLICATIONS_CHOICE = "The application's choice"

#: What a service returned, as opposed to what it says. A comma is a
#: written mark that no service can really disagree about, but laughter is
#: something that happened in the room, and a person deciding whether a
#: candidate belongs in the transcript needs to be able to tell the two
#: apart rather than reading "laughter" as an ordinary word.
SPOKEN_WORD = "A spoken word"
AUDIO_EVENT = "A sound rather than a word"
PUNCTUATION = "Punctuation"
UNKNOWN_KIND = "Not known"


# -- Putting the model's states into words -------------------------------

_TIMING_STATUS_TEXT: dict[TimingStatus, str] = {
    TimingStatus.EXACT_PROVIDER_TIME: "A service timed this exact word and nothing changed it",
    TimingStatus.MAPPED_FORMAT_EQUIVALENT: (
        "Only the formatting changed, so the original span still fits"
    ),
    TimingStatus.MAPPED_SUBSTITUTION: (
        "A different word now sits in the span of the word it replaced"
    ),
    TimingStatus.SHARED_PHRASE_SPAN: (
        "Several words share one span, because the join between them is unknown"
    ),
    TimingStatus.FORCED_ALIGNED: "The span was measured again against the audio",
    TimingStatus.APPROXIMATE_SPAN: "The right region, but its edges are not exact",
    TimingStatus.UNALIGNED: "No audio evidence maps to this word at all",
    TimingStatus.UNCERTAIN: "Alignment was attempted and could not be justified",
}

_ALIGNMENT_STATUS_TEXT: dict[AlignmentStatus, str] = {
    AlignmentStatus.EXACT: "The same word, character for character",
    AlignmentStatus.FORMAT_EQUIVALENT: "The same word, once spelling habits are set aside",
    AlignmentStatus.SUBSTITUTION: "A genuinely different word in the same place",
    AlignmentStatus.MERGE: "Several words of the backbone became this one word",
    AlignmentStatus.SPLIT: "One word of the backbone became several words",
    AlignmentStatus.INSERTION: "A service heard a word the backbone does not have",
    AlignmentStatus.DELETION: "The backbone has a word a service did not hear",
    AlignmentStatus.UNALIGNED: "No correspondence could be established",
}

#: The reasons that are about the clock rather than about the words.
#: Confirming a span answers these and leaves everything else standing.
_TIMING_REVIEW_REASONS = frozenset(
    {
        ReviewReason.WEAK_ALIGNMENT,
        ReviewReason.FORCED_ALIGNMENT_FAILED,
        ReviewReason.UNALIGNED_WORD,
    }
)


def _counted(count: int, noun: str) -> str:
    """"1 group" or "4 groups", because "1 groups" reads as a bug in the code.

    A sentence that is obviously machine-assembled invites the reader to
    distrust the numbers in it, and these numbers are the ones a person acts
    on before changing files they have not opened.
    """
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _seconds_phrase(seconds: float) -> str:
    """A short length in seconds, to one decimal place.

    Words last a fraction of a second, and the spoken duration helper
    rounds to whole seconds because that is the right grain for a
    recording. Both are used here: the ends of a span are read as positions
    in the recording, and the length between them is read as a measurement.
    """
    return f"{seconds:.1f} seconds"


def provider_name(provider: Provider | None) -> str:
    return provider.display_name if provider is not None else NOT_ATTRIBUTED


def timing_text(token: FinalToken) -> str:
    """Where in the recording this word sits, in words rather than numbers.

    The start and the length are given rather than the start and the end.
    A position is read out to the nearest second, because that is the
    grain at which a person navigates a recording, and two ends rounded
    that way would say a half-second word ran from thirty seconds to
    thirty seconds, or a six-and-a-half-second region ran from
    twenty-seven to thirty-four. The length is the part that has to be
    exact, so it is the part given exactly.

    A word with no boundaries of its own is described by the wider span it
    was found in, because claiming a start it does not have would be a lie
    the rest of the application would then believe.
    """
    if token.start is not None and token.end is not None:
        length = _seconds_phrase(token.end - token.start)
        return f"From {spoken_duration(token.start)}, {length} long"
    span = token.source_audio_span
    if span is None:
        return NO_TIMING
    return (
        f"Somewhere in the {_seconds_phrase(span.duration)} "
        f"from {spoken_duration(span.start)}"
    )


def speaker_text(transcript: Transcript, token: FinalToken) -> str:
    if token.speaker is None:
        return NO_SPEAKER
    speaker = transcript.speaker_for(token.speaker)
    return speaker.display_name if speaker is not None else f"Speaker {token.speaker}"


def language_evidence_text(token: FinalToken) -> str:
    """Every language that was in the running, and how likely each was.

    Kept as several numbers rather than one answer, because a code-switch
    boundary is genuinely uncertain and the person deciding deserves to see
    how close it was.
    """
    scores = token.language_evidence.scores
    if not scores:
        return NO_LANGUAGE_EVIDENCE
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return ", ".join(
        f"{language.display_name} {score * 100:.0f} percent" for language, score in ordered
    )


def risk_text(categories: list[RiskCategory]) -> str:
    """The risk categories spelled out, because these must never be guessed."""
    if not categories:
        return NO_RISK
    return ", ".join(category.display_name for category in categories)


def provider_confidence_text(token: ProviderToken | None) -> str:
    """What one service said about its own certainty, in whatever form it gives.

    Services report on scales that do not mean the same thing, so the
    figure is shown as the service gave it and labelled with what it is,
    rather than being converted into a common percentage that would invent
    a comparison nobody can justify.
    """
    if token is None:
        return NOT_REPORTED
    if token.confidence is not None:
        return f"{token.confidence * 100:.0f} percent confident"
    if token.log_probability is not None:
        return f"Log probability {token.log_probability:.2f}"
    return NOT_REPORTED


def kind_text(token: ProviderToken | None) -> str:
    """Whether a service returned a word, a sound or a written mark.

    Audio events are kept apart from punctuation because they are skipped
    by alignment for opposite reasons. Showing a laugh as though it were
    the word "laughter" would have a person correcting something that was
    never a transcription mistake in the first place.
    """
    if token is None:
        return UNKNOWN_KIND
    if token.is_audio_event:
        return AUDIO_EVENT
    if token.is_punctuation:
        return PUNCTUATION
    return SPOKEN_WORD


def _provider_token(
    transcript: Transcript,
    provider: Provider,
    references: tuple[TokenReference, ...],
) -> ProviderToken | None:
    """The service's own word behind a candidate, where the trail leads to one."""
    result = transcript.provider_results.get(provider)
    if result is None:
        return None
    for reference in references:
        if reference.provider is provider:
            return result.token_at(reference.index)
    return None


# -- Corrections the transcript itself does not cover ---------------------


def with_confirmation(transcript: Transcript, token_id: str) -> Transcript:
    """Return a copy with one word confirmed as correct, unchanged.

    This is the common case and the fastest action on the screen: a person
    read the word, listened to it, and found nothing wrong. Their approval
    is evidence as strong as a correction, so all three confidences become
    high and the word leaves the queue.

    Unlike a correction, this does not set ``human_corrected``. Nothing
    changed, so there is nothing here for the vocabulary or the provider
    statistics to learn from. Raising the text confidence does matter,
    though: an unresolved word is exported as ``[UNCERTAIN: ...]``, and a
    word a person has approved must not still be written out that way.
    """
    tokens = list(transcript.tokens)
    for position, token in enumerate(tokens):
        if token.id != token_id:
            continue
        # The copy shares the original's lists, so every list this touches
        # is rebound to a new one rather than changed in place. Changing
        # one in place would reach back into the transcript being replaced.
        confirmed = replace(token)
        confirmed.text_confidence = Confidence.HIGH
        # The saved strength has to follow the category up, for the same
        # reason the category is raised at all. A word left sitting at the
        # strength the services argued it to would keep being picked up by
        # anything that sorts or filters on that number, and the person who
        # approved it would be shown it again as though they never had.
        confirmed.confidence_strength = 1.0
        confirmed.timing_confidence = Confidence.HIGH
        confirmed.speaker_confidence = Confidence.HIGH
        confirmed.review_status = ReviewStatus.CONFIRMED
        confirmed.review_reasons = []
        tokens[position] = confirmed
        break
    return replace(transcript, tokens=tokens)


def with_timing_decision(transcript: Transcript, token_id: str, accepted: bool) -> Transcript:
    """Return a copy with the span of one word confirmed or rejected.

    Confirming answers the questions that are about the clock and leaves
    every other reason standing, so a word that was also flagged because
    the services disagreed about the spelling stays in the queue until that
    is settled too. Correcting the timing says nothing about the text or
    the speaker, which is the same separation the whole design rests on.

    Rejecting keeps the numbers rather than clearing them. They are the
    only clue left to where the word is, and clearing them would take away
    the ability to play it, which is precisely what the person needs next.
    What changes is that the span is marked uncertain, so nothing
    downstream treats it as measured, and the word stays in the queue for
    forced alignment or a second look.
    """
    tokens = list(transcript.tokens)
    for position, token in enumerate(tokens):
        if token.id != token_id:
            continue
        decided = replace(token)
        decided.human_corrected = True
        if accepted:
            decided.timing_confidence = Confidence.HIGH
            remaining = [
                reason for reason in token.review_reasons
                if reason not in _TIMING_REVIEW_REASONS
            ]
            decided.review_reasons = remaining
            decided.review_status = (
                ReviewStatus.PENDING if remaining else ReviewStatus.CORRECTED
            )
        else:
            decided.timing_status = TimingStatus.UNCERTAIN
            decided.timing_confidence = Confidence.UNRESOLVED
            reasons = list(token.review_reasons)
            if ReviewReason.WEAK_ALIGNMENT not in reasons:
                reasons.append(ReviewReason.WEAK_ALIGNMENT)
            decided.review_reasons = reasons
            decided.review_status = ReviewStatus.PENDING
        tokens[position] = decided
        break
    return replace(transcript, tokens=tokens)


# -- What each service offered -------------------------------------------

CANDIDATE_COLUMN_SERVICE = 0
CANDIDATE_COLUMN_TEXT = 1
CANDIDATE_COLUMN_CONFIDENCE = 2
CANDIDATE_COLUMN_VOCABULARY = 3
CANDIDATE_COLUMN_KIND = 4
CANDIDATE_COLUMN_CHOICE = 5
CANDIDATE_COLUMN_COUNT = 6

CANDIDATE_COLUMN_TITLES = (
    "Service",
    "Candidate",
    "Service confidence",
    "Known term",
    "Kind",
    "Chosen",
)


@dataclass(frozen=True)
class CandidateRow:
    """One service's answer for one word, ready to be read across a row.

    Every field is already a sentence or a phrase. Nothing here is a tick,
    a colour or a blank cell whose meaning depends on which column it is
    in, because all three are invisible to somebody hearing the row read
    out.
    """

    service: str
    text: str
    confidence: str
    vocabulary: str
    kind: str
    choice: str


def candidate_rows(transcript: Transcript, token: FinalToken | None) -> list[CandidateRow]:
    """Every service's candidate for a word, beside the one that was chosen.

    A candidate backed by three services becomes three rows, one for each,
    because what each of them reported about its own certainty is different
    and is half the reason a person can decide at all.

    The word the application settled on is added at the end when no
    candidate carries it, so the current choice is always on the table even
    when it came from somewhere other than the candidate list.
    """
    if token is None:
        return []
    rows: list[CandidateRow] = []
    chosen_seen = False
    for candidate in token.candidates:
        chosen = candidate.text == token.text
        chosen_seen = chosen_seen or chosen
        vocabulary = IN_VOCABULARY if candidate.in_vocabulary else NOT_IN_VOCABULARY
        choice = CURRENT_CHOICE if chosen else NOT_CHOSEN
        if not candidate.providers:
            rows.append(
                CandidateRow(
                    NOT_ATTRIBUTED,
                    candidate.text,
                    NOT_REPORTED,
                    vocabulary,
                    UNKNOWN_KIND,
                    choice,
                )
            )
            continue
        for provider in candidate.providers:
            reported = _provider_token(transcript, provider, candidate.source_tokens)
            rows.append(
                CandidateRow(
                    provider.display_name,
                    candidate.text,
                    provider_confidence_text(reported),
                    vocabulary,
                    kind_text(reported),
                    choice,
                )
            )
    if not chosen_seen:
        rows.append(
            CandidateRow(
                THE_APPLICATIONS_CHOICE,
                chosen_text(token),
                NOT_REPORTED,
                _vocabulary_of_chosen(token),
                UNKNOWN_KIND,
                CURRENT_CHOICE,
            )
        )
    return rows


def _vocabulary_of_chosen(token: FinalToken) -> str:
    """Whether the chosen word is a known term, where a candidate says so."""
    for candidate in token.candidates:
        if candidate.text == token.text and candidate.in_vocabulary:
            return IN_VOCABULARY
    return NOT_IN_VOCABULARY


class CandidateTableModel(QAbstractTableModel):
    """The candidates of one word, as a standard table.

    A table rather than a drawn panel, so that Windows accessibility sees
    real rows, columns and headers, and a screen reader can walk across a
    row hearing which service said what.
    """

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._rows: list[CandidateRow] = []

    def set_rows(self, rows: list[CandidateRow]) -> None:
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def row_at(self, row: int) -> CandidateRow | None:
        if 0 <= row < len(self._rows):
            return self._rows[row]
        return None

    def rows(self) -> list[CandidateRow]:
        return list(self._rows)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return CANDIDATE_COLUMN_COUNT

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ):
        if orientation != Qt.Orientation.Horizontal:
            return None
        if not 0 <= section < CANDIDATE_COLUMN_COUNT:
            return None
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.AccessibleTextRole):
            return CANDIDATE_COLUMN_TITLES[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self.row_at(index.row())
        if row is None:
            return None
        if role in (
            Qt.ItemDataRole.DisplayRole,
            Qt.ItemDataRole.AccessibleTextRole,
            Qt.ItemDataRole.ToolTipRole,
        ):
            return (
                row.service,
                row.text,
                row.confidence,
                row.vocabulary,
                row.kind,
                row.choice,
            )[index.column()]
        if role == Qt.ItemDataRole.UserRole:
            return row
        return None


# -- What each control is for --------------------------------------------

REASON_FILTERS = "reason_filters"
CONFIDENCE_FILTERS = "confidence_filters"
GROUPS = "groups"
OCCURRENCES = "occurrences"
WORD_DETAILS = "word_details"
CANDIDATES = "candidates"
CANDIDATE_TEXT = "candidate_text"
TIMING = "timing"
SPEAKER = "speaker"
LANGUAGE = "language"
RISK = "risk"
DECISION = "decision"
PLAYBACK = "playback"
AUTO_PLAY = "auto_play"
AUTO_PLAY_DELAY = "auto_play_delay"
REPLACEMENT = "replacement"
CORRECT_SPEAKER = "correct_speaker"
TIMING_DECISION = "timing_decision"
CONFIRM = "confirm"
ISOLATE = "isolate"
PROCESSING = "processing"
THRESHOLDS = "thresholds"
VISIBILITY = "visibility"
NAVIGATE = "navigate"


@dataclass(frozen=True)
class ControlNote:
    """What a control is called, and what there is to say about it."""

    title: str
    summary: str
    """One or two sentences, for the accessible description and the tooltip."""

    note: str
    """The fuller explanation, for the panel that follows the focus."""


def _reflowed(text: str) -> str:
    """Put each paragraph on one line so the panel can wrap it to fit.

    The notes are written wrapped in the source, because that is where they
    are read and reviewed. A screen reader reads a text box a line at a
    time, so a paragraph broken across six lines is read as six lines with
    a pause in every gap.
    """
    paragraphs = text.strip().split("\n\n")
    return "\n\n".join(
        " ".join(line.strip() for line in paragraph.splitlines()).strip()
        for paragraph in paragraphs
    )


_NOTES: dict[str, ControlNote] = {
    REASON_FILTERS: ControlNote(
        "Reasons for review",
        "Clear a reason to hide the words flagged for it.",
        _reflowed(
            """
            Every word in the queue was flagged for one or more reasons. Clearing a
            reason hides the words that carry it, which is how you work through one
            kind of problem at a time: all the disputed numbers first, say, and the
            uncertain speakers afterwards.

            The count under the filters always says how many items are showing out of
            how many there are, and it is read out whenever you change a filter, so an
            empty list is never mistaken for a finished one.
            """
        ),
    ),
    CONFIDENCE_FILTERS: ControlNote(
        "Confidence categories",
        "Clear a category to hide the words the application rates that way.",
        _reflowed(
            """
            A word is only as trustworthy as its weakest part, so the category here is
            the weakest of the three: the text, the timing and the speaker. Text you can
            rely on, sitting at a time you cannot, is not a settled word.

            Categories are used rather than percentages because the services score
            themselves on scales that do not mean the same thing, and averaging them
            would invent a precision that is not there.
            """
        ),
    ),
    GROUPS: ControlNote(
        "Word Groups",
        "The words needing a person, gathered across every recording in the folder.",
        _reflowed(
            """
            One row per word rather than per mention. A surname the services were unsure
            of in eleven recordings is one row here, and deciding it once decides it in
            all eleven. Each row gives the word, how many occurrences it has and in how
            many files, how confident the application is, why it is in the list, and
            whether it has been reviewed. All of that is words and numbers in columns,
            never a colour or a mark.

            The weakest words come first. The items the transcription itself flagged
            follow them as a second block, so that working down the list stays
            predictable, and the column saying why a row is here tells you which block
            you are in without having to count.

            Ctrl+F3 and Ctrl+Shift+F3 move to the next and previous word from anywhere
            in the window.
            """
        ),
    ),
    OCCURRENCES: ControlNote(
        "Occurrences",
        "Where the selected word actually falls: one row per moment in one recording.",
        _reflowed(
            """
            Each row gives the file, the position in it, the confidence, the language,
            the replacement that will be applied and whether this one has been settled.
            Moving the highlight reads the row out and plays the audio, without taking
            the focus out of the list.

            This is where the speaker and the timing are corrected, because those really
            do differ between one mention of a word and the next. The spelling is
            decided once for the whole word, in the list beside this one.

            F3 and Shift+F3 move to the next and previous occurrence from anywhere in
            the window.
            """
        ),
    ),
    WORD_DETAILS: ControlNote(
        "The word itself",
        "What the services produced, and where and when they produced it.",
        _reflowed(
            """
            The detected word is what the services actually returned, before anything
            was applied to it. It is kept unchanged for the life of the project, because
            a replacement is always worked out from what a word originally said rather
            than from what it currently says. That is what makes changing your mind
            about a replacement give the same result as having typed it correctly the
            first time.

            The confidence is the application's own combined estimate for this word,
            worked out from how far the services agreed and how well the audio lined up.
            It is not any one service's figure and is not comparable with one. A word
            nobody measured says so, rather than being shown as nought.
            """
        ),
    ),
    CANDIDATES: ControlNote(
        "Service candidates",
        "What each service heard, and what it said about its own certainty.",
        _reflowed(
            """
            One row per service, so a word three services agreed on appears three times
            with three separate confidences. That repetition is the point: agreement is
            evidence, and you can only weigh it if you can see who is agreeing.

            The last two columns say whether the candidate matches a known vocabulary
            term and which one the application chose. Both are words rather than ticks
            or colours.
            """
        ),
    ),
    CANDIDATE_TEXT: ControlNote(
        "Candidate text",
        "The highlighted candidate on its own, where it can be read closely.",
        _reflowed(
            """
            The candidates that matter most are names, numbers and spellings, and those
            have to be read character by character rather than glanced at. This field
            holds the highlighted candidate on its own, takes the focus, and shows a
            real caret, so a screen reader can spell it out and a magnifier can follow
            it.

            The button beside it copies the candidate into the correction box, which is
            the quickest way to accept what one service heard.
            """
        ),
    ),
    TIMING: ControlNote(
        "When it was said",
        "The span, how it was arrived at, and which service timed it.",
        _reflowed(
            """
            The timing is a separate answer from the text, with its own source and its
            own confidence. A word can be spelled the way one service heard it and
            timed the way another did, and that is an ordinary word rather than a
            broken one.

            The status says how the span was arrived at. Some states, such as a shared
            phrase span, mean the boundaries are honestly unknown, and playing the word
            will play the wider region it sits in rather than pretending otherwise.
            """
        ),
    ),
    SPEAKER: ControlNote(
        "Who said it",
        "The speaker, which service decided, and how confident it was.",
        _reflowed(
            """
            The speaker is the third separate answer. Correcting it changes nothing
            about the text or the timing, and correcting either of those changes
            nothing about the speaker.

            Speaker attribution is usually the work of one service acting as the
            authority, so a low confidence here often means overlapping speech or a
            very short utterance rather than a poor recording.
            """
        ),
    ),
    LANGUAGE: ControlNote(
        "Language",
        "The language decided on, and how likely each one was.",
        _reflowed(
            """
            The evidence is kept as several numbers rather than one answer, because the
            boundary where a speaker switches language is genuinely uncertain, and
            because which services are allowed an opinion on a span depends on how sure
            the language is rather than on a decision already taken.
            """
        ),
    ),
    RISK: ControlNote(
        "Risk categories",
        "Kinds of content that must never be settled by what sounds plausible.",
        _reflowed(
            """
            Amounts, dates, telephone numbers, account numbers and the like are listed
            here in full. Getting a common word wrong makes a transcript slightly worse.
            Getting one of these wrong makes it dangerous, and the mistake is invisible
            because the wrong value reads perfectly well.

            Nothing here is ever chosen on plausibility, by the scoring rules or by the
            language model. Where the evidence stays split, the uncertainty is preserved
            and brought to you instead.
            """
        ),
    ),
    DECISION: ControlNote(
        "What the language model decided",
        "The adjudicating model's answer, where it was asked at all.",
        _reflowed(
            """
            The language model is asked only about words the deterministic rules could
            not settle, and it is allowed to decline. Where it declined, or where it was
            never asked, that is what this says.

            Its answer is evidence like any other, not an instruction. It is shown here
            so that you can disagree with it.
            """
        ),
    ),
    PLAYBACK: ControlNote(
        "Playing the word",
        "Plays the word with several seconds either side. F5, or Shift+F5 for more.",
        _reflowed(
            """
            A word without its sentence cannot be judged, so playing never plays the
            word alone. Three seconds go either side of it, and the second button gives
            twelve, for when you need to hear what was being talked about rather than
            just the word.

            Where the word's own boundaries are not known, the wider region it was found
            in is played instead. That is exactly the case where you most need to hear
            it, and the field above says what will actually be played.
            """
        ),
    ),
    AUTO_PLAY: ControlNote(
        "Playing by itself",
        "Whether landing on an occurrence plays it without being asked. Remembered.",
        _reflowed(
            """
            Moving the highlight onto an occurrence plays it after the wait set below.
            The focus stays where it is, so you are never taken out of the list you are
            working in, and moving again before the wait is up cancels the first clip
            rather than queueing it, so holding the Down arrow does not fill the room
            with eight overlapping words.

            The recording changes as you move between occurrences, and a file takes a
            moment to open, so the first clip in a new file starts a little after the
            others. Switch this off if you would rather ask for the audio yourself with
            F5; the choice is remembered with the rest of this folder's review.
            """
        ),
    ),
    AUTO_PLAY_DELAY: ControlNote(
        "The wait before the audio starts",
        "How long to wait after landing on an occurrence before it plays. Remembered.",
        _reflowed(
            """
            Landing on an occurrence reads the row out and then plays it. The reading
            takes as long as your screen reader takes, which is a speed only you know,
            and if the audio starts before the reading has finished you hear two voices
            at once and learn nothing from either. This is how long the window waits.

            Set it a little longer than it takes your reader to get through a row. Two
            seconds suits an ordinary speaking rate; a fast rate wants less and a slow
            one more. Nought means the audio starts the instant you arrive, which is
            what you want if no screen reader is running.

            The wait is also what stops eight clips queueing when you hold the Down
            arrow, because each move starts it again from the beginning. Making it
            longer therefore makes moving through a long list quieter as well.
            """
        ),
    ),
    REPLACEMENT: ControlNote(
        "The replacement",
        "What this word should say. Applies to every occurrence, or to this one alone.",
        _reflowed(
            """
            Type what was actually said. Applying it to the word changes every
            occurrence of it, in every recording, and the message says exactly how many
            occurrences and how many files that is before you have to look anywhere.
            Applying it to this occurrence only changes the one you are on, and that
            occurrence then keeps its own answer even if you change the word's later.

            Each occurrence is rewritten from what it originally said, never from what
            it currently says, so correcting a replacement you got wrong leaves no trace
            of the first attempt.

            Neither kind of change touches when a word was said or who said it. Those
            are separate answers with separate sources, and they are left exactly as
            they were. The speaker is corrected one occurrence at a time, below.
            """
        ),
    ),
    ISOLATE: ControlNote(
        "Keeping an occurrence on its own",
        "Takes this occurrence out of its word, to be decided by itself.",
        _reflowed(
            """
            Grouping is a suggestion. It gathers words that are probably the same
            intended word, and it is allowed to reach, because a group you disagree with
            costs one keystroke to break up and a group that was too timid costs you the
            same decision over and over.

            This is that keystroke. The occurrence leaves the group, appears in the
            first list on its own, and keeps whatever you decide about it. Running the
            analysis again never gathers it back up, because taking it out was a
            statement about that word and a new threshold is not evidence against it.
            """
        ),
    ),
    PROCESSING: ControlNote(
        "Processing and regrouping",
        "Finds the weak words in every recording and gathers them into words.",
        _reflowed(
            """
            Processing looks at every word of every transcript in the folder, not only
            the words already flagged for review, and keeps the ones whose confidence
            falls below the minimum. A word with no measured confidence is left out
            entirely rather than guessed at. Regrouping runs the same analysis again,
            which is what you do after moving the minimum or the tolerance.

            Neither one can undo your work. Words you have reviewed, replacements you
            have applied, occurrences you have kept on their own and groups you made
            yourself all survive; what is rebuilt around them is only the part the
            analysis suggested in the first place.

            Where a replacement you accepted earlier answers a word found now, it is
            applied and the occurrence says that a project rule did it rather than you.
            """
        ),
    ),
    THRESHOLDS: ControlNote(
        "The two thresholds",
        "How weak a word must be to appear, and how alike two words must be to group.",
        _reflowed(
            """
            The minimum confidence decides which words are put in front of you at all.
            Raise it and more words appear, including many that are perfectly correct;
            lower it and fewer appear, and some genuine mistakes stay in the transcript.

            The grouping tolerance decides how far apart two confidences may be while
            still being treated as the same word. It is a boundary between neighbours
            rather than a rule about the whole group, and it never splits two spellings
            that match exactly, because the same surname at 42 and at 61 percent is
            still the same surname.

            Neither takes effect until you regroup the words, so that changing your mind
            about a number does not rearrange the list underneath you while you are
            still typing it.
            """
        ),
    ),
    VISIBILITY: ControlNote(
        "What the first list shows",
        "Whether words you have settled, and words the pipeline flagged, are shown.",
        _reflowed(
            """
            A word you have settled leaves the list, because the list is the work that
            is left. Showing reviewed words brings them back, with a column saying how
            each was settled: replaced, correct as detected, or applied automatically
            from a rule you accepted earlier. That last distinction matters, because
            only one of the three is a decision you remember making.

            The other uncertainties are the words the transcription itself flagged, for
            reasons such as the services disagreeing or a risky value being involved.
            They are a separate block rather than mixed in, and they can be switched off
            entirely while you work through the low-confidence words.
            """
        ),
    ),
    CORRECT_SPEAKER: ControlNote(
        "Correcting the speaker",
        "Attributes the word to somebody else. The text is left alone.",
        _reflowed(
            """
            The list holds the speakers already found in this recording. You can also
            type a label that is not in the list, for a speaker the services never
            separated out.

            As with the text, this changes one answer and one only. The words and the
            timing stay as they were.
            """
        ),
    ),
    TIMING_DECISION: ControlNote(
        "Confirming or rejecting the timing",
        "Says whether the span is right, without touching the text or the speaker.",
        _reflowed(
            """
            Confirming answers the questions that are about the clock and leaves the
            rest standing, so a word that is also disputed on spelling stays in the
            queue until that is settled too.

            Rejecting keeps the numbers but marks them uncertain, so nothing downstream
            treats them as measured. They are kept because they are the only clue left
            to where the word is, and losing them would lose the ability to play it.
            """
        ),
    ),
    CONFIRM: ControlNote(
        "Saying a word is right as it stands",
        "Settles it without changing it. F4 for this occurrence, or the whole word.",
        _reflowed(
            """
            This is the common case and the fastest thing on the screen: you listened,
            you read the candidates, and the application had it right. One keystroke,
            F4, and the occurrence is settled.

            Correcting the word as detected does the same for every occurrence of it at
            once, and says how many that is before doing it. Where a replacement had
            already been applied, the words go back to what the services produced and
            the rules that came from that decision are dropped, so that a file
            transcribed next week is not still answered by a decision you have reversed.

            Your approval is treated as evidence as strong as a correction, so the word
            becomes fully confident. Nothing is recorded as changed, because nothing
            changed, so there is nothing here for the vocabulary to learn.
            """
        ),
    ),
    NAVIGATE: ControlNote(
        "Moving through the work",
        "F3 and Shift+F3 for occurrences, Ctrl+F3 and Ctrl+Shift+F3 for words.",
        _reflowed(
            """
            The occurrence keys are the plain ones, because the occurrence is the inner
            loop: you arrive at a word, listen to its occurrences, and decide. The whole
            row is read out when the highlight moves, which is enough to work straight
            down a list without going to look at the detail panel for every one.

            Moving to a word rather than an occurrence takes the Ctrl key, and reads out
            the word, how many occurrences it has, in how many files, and whether it has
            been reviewed.

            F6 moves between the panels of this window, and says which one you have
            landed on.
            """
        ),
    ),
}


def note_for(key: str) -> ControlNote:
    return _NOTES[key]




class ReviewWindow(QMainWindow):
    """Works through everything a folder of recordings could not settle."""

    transcriptChanged = Signal(str, object)
    """A correction was made, carrying the recording name and the whole
    corrected transcript. The name travels with it because one window now
    holds every recording in the folder, and a transcript on its own no
    longer says which file it belongs to."""

    def __init__(
        self,
        folder: Path,
        recording_names: list[str],
        load_transcript: Callable[[str], Transcript | None],
        recording_paths: dict[str, Path],
        project_store: ProjectStore,
        player: AudioPlayer,
        save_correction: Callable[[str, Transcript], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._folder = Path(folder)
        self.setWindowTitle(f"Review: {self._folder.name}")
        self._recording_names = list(recording_names)
        self._load_transcript = load_transcript
        self._recording_paths = dict(recording_paths)
        # One transcript at a time, and the one being worked on. The module
        # docstring says what a dictionary of all of them would cost.
        self._cached_name: str | None = None
        self._cached_transcript: Transcript | None = None
        self._project_store = project_store
        # Loading never fails and never returns nothing, so there is no error
        # path here. A folder nobody has reviewed and a folder whose file was
        # damaged both arrive as an empty project, and the window opens on it.
        self._state = project_store.load()
        self._player = player
        self._save_correction = save_correction

        self._note_keys: dict[QWidget, str] = {}
        self._showing_note: str | None = None
        self._watching_focus = False
        # Where playback should stop. Qt has no notion of playing a region,
        # so the position is watched and playback paused when it arrives.
        self._stop_at_ms: int | None = None
        # Set while the window is changing the filter check boxes itself, so
        # that putting them all back on is one change and one announcement
        # rather than seventeen.
        self._setting_filters = False
        # The same, for the three toggles that appear both as a check box and
        # as a menu entry. Each sets the other, and without this they would
        # answer each other round the loop.
        self._setting_toggles = False
        # Set while the window is rebuilding the lists. A selection it moves
        # itself must still fill the panels in, because that is what keeps the
        # screen honest, and must announce nothing and play nothing, because
        # the person did not move.
        self._quiet = False
        # Set while the occurrence selection is moving because the word moved.
        # The row is still shown and still played; what it does not do is
        # speak, because the word it belongs to is being read out already.
        self._announce_occurrence = True
        # Set when the window has had to take the focus off a control it was
        # about to switch off. Whoever is about to say what happened adds it
        # to their own sentence, so the two never talk over each other.
        self._focus_caught = False

        # Where the person is. Kept as identifiers rather than row numbers,
        # because every change rebuilds both lists and a row number would then
        # quietly point at whichever word had moved into that place.
        self._selected_group_key: str | None = None
        self._selected_occurrence_id: str | None = None
        self._hidden_reviewed = 0
        # The occurrence the highlight has just come from, so that landing on
        # the next one can leave out whatever has not changed between the two.
        # Cleared whenever the word changes, because arriving at a word is when
        # the whole sentence is wanted.
        self._previous_occurrence: Occurrence | None = None

        # What each of the three numbers was when the person last finished with
        # it. Qt raises editingFinished for a box that was merely visited as
        # well as for one that was changed, and a spin box that has been tabbed
        # through must not announce a threshold nobody moved.
        settings = self._state.settings
        self._settled_minimum = round(settings.minimum_confidence * 100)
        self._settled_tolerance = round(settings.grouping_tolerance * 100)
        self._settled_delay = settings.auto_play_delay_seconds
        # Set once a close has been refused because the project could not be
        # saved, so that asking a second time closes anyway. A window somebody
        # cannot get out of would be a worse fault than the one being reported.
        self._close_refused = False

        # Playback. The window owns its player, so it may load whatever file
        # the selected occurrence needs. Loading is asynchronous, so what to
        # play is remembered until the media says it is ready.
        self._loaded_path: Path | None = None
        self._media_ready = False
        self._pending_play: tuple[int, int, str] | None = None
        self._play_timer = QTimer(self)
        self._play_timer.setSingleShot(True)

        # The words the pipeline itself flagged, which are the second block of
        # the first list. The queue model is not shown anywhere; it is used
        # for what it already knows, which is which words need a person and
        # which of those the reason and confidence filters are letting
        # through. Keeping that in one place is what stops the filters coming
        # to mean something slightly different here from what they mean there.
        self._queue_model = ReviewQueueModel(self)
        self._uncertain: list[tuple[str, FinalToken]] = []
        self._settled: list[tuple[str, FinalToken]] = []
        # Which recordings have been read *in this window*, and so are being
        # described by their real words rather than by what the project
        # remembers of them. Nothing about the lists depends on this any
        # more; it is kept because a correction can be written into a
        # recording the window has never read, and the two cases part company
        # in _hand_on.
        self._scanned: set[str] = set()
        # The recordings one run of the analysis managed to read, so that it
        # can say how many it could not. Emptied at the start of every run.
        self._read_this_run: set[str] = set()
        # The flagged words of the whole folder, taken from the project rather
        # than found by reading the transcripts. This is the line that makes
        # the second block complete on a folder nobody has opened a file of,
        # and the reasoning behind it is at
        # audio_transcriber.transcription.project.FlaggedItem: reading the
        # folder instead is the obvious-looking simplification and it costs 37
        # seconds on fifty hour-long recordings, every single time.
        self._restore_flagged()

        self._group_model = WordGroupModel(self)
        self._occurrence_model = OccurrenceModel(self)

        self._build_ui()
        self._build_menus()
        self._connect_signals()
        self.refresh(
            word_key=None
            if self._state.last_group_id is None
            else group_key(self._state.last_group_id),
            occurrence_id=self._state.last_occurrence_id,
        )
        self._size_to_fit_the_screen()

    def _size_to_fit_the_screen(self) -> None:
        """Open at a comfortable size, but never wider than the screen it is on.

        The window wants about 1440 by 860 logical pixels, which is what its
        two lists and its detail panel need side by side. Asking for that flat
        is what it used to do, and on a 1366-wide laptop, or on any screen at
        150 percent Windows scaling, the right-hand edge of the window opened
        past the right-hand edge of the display. Everything in the corrections
        panel was then off screen on first run, with no scroll bar leading to
        it, and a person who cannot see where the window went cannot drag it
        back either.

        So the wanted size is clamped to what the screen actually offers, less
        a small margin for the taskbar and the window frame. ``availableGeometry``
        is already in the same logical pixels the rest of Qt uses, so the
        scaling is accounted for without being mentioned.
        """
        wanted_width, wanted_height = 1440, 860
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            wanted_width = min(wanted_width, available.width() - 40)
            wanted_height = min(wanted_height, available.height() - 40)
        self.resize(max(wanted_width, self.minimumWidth()), max(wanted_height, 400))

    # -- Building the window ---------------------------------------------

    def _build_ui(self) -> None:
        self._splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._splitter.addWidget(self._build_left_side())
        self._splitter.addWidget(self._build_right_side())
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 2)
        describe(self._splitter, "Window divider")
        self.setCentralWidget(self._splitter)

        # This label carries messages, so it is left unnamed. A label has no
        # accessible value of its own: its text is its name, and naming it
        # would hide whatever it says behind the name.
        self._status_label = QLabel("Ready", self)
        status_bar = QStatusBar(self)
        status_bar.addWidget(self._status_label, 1)
        status_bar.setSizeGripEnabled(True)
        self.setStatusBar(status_bar)

    def _build_left_side(self) -> QWidget:
        """The two lists side by side, with what shapes them underneath.

        Side by side rather than one above the other because the two are read
        together: the word on the left, and where it occurs on the right. The
        controls go underneath both, since they decide what the first list
        holds, and they scroll, because at a large Windows text size they are
        taller than any window.
        """
        side = QWidget(self)
        layout = QVBoxLayout(side)
        layout.setContentsMargins(8, 8, 4, 8)

        lists = QSplitter(Qt.Orientation.Horizontal, side)
        lists.addWidget(self._build_group_list(side))
        lists.addWidget(self._build_occurrence_list(side))
        lists.setChildrenCollapsible(False)
        lists.setStretchFactor(0, 1)
        lists.setStretchFactor(1, 1)
        describe(lists, "List divider")

        # This one carries the count, so it is left unnamed for the same
        # reason as the status label.
        self._count_label = QLabel("", side)
        self._count_label.setWordWrap(True)

        layout.addWidget(lists, 3)
        layout.addWidget(self._count_label)
        layout.addWidget(self._build_settings(side), 2)
        return side

    def _build_group_list(self, parent: QWidget) -> QWidget:
        holder = QWidget(parent)
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        # "Word Groups", capitalised exactly as the accessible name below is.
        # The two used to differ by a letter -- "Word groups" on screen and
        # "Word Groups" to a reader -- which is a small thing until somebody
        # is being talked through the window by a colleague looking at their
        # own screen, and the two of them are naming the same list differently.
        #
        # The Alt letter is G rather than W. W is the Word menu's, and Qt does
        # not refuse a second claimant: it cycles between them, so the same
        # keystroke did different things on successive presses.
        self._groups_label = QLabel("Word &Groups", holder)
        self._groups = WordGroupView(holder)
        self._groups.setModel(self._group_model)
        self._explain(self._groups, GROUPS)
        # The accessible name is the one the specification names. It is set
        # deliberately rather than left to follow the label, because the label
        # carries an ampersand that has no business being read out.
        self._groups.setAccessibleName("Word Groups")
        self._groups_label.setBuddy(self._groups)
        layout.addWidget(self._groups_label)
        layout.addWidget(self._groups, 1)
        return holder

    def _build_occurrence_list(self, parent: QWidget) -> QWidget:
        holder = QWidget(parent)
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        self._occurrences_label = QLabel("&Occurrences", holder)
        self._occurrences = OccurrenceView(holder)
        self._occurrences.setModel(self._occurrence_model)
        self._explain(self._occurrences, OCCURRENCES)
        self._occurrences.setAccessibleName("Occurrences")
        self._occurrences_label.setBuddy(self._occurrences)
        layout.addWidget(self._occurrences_label)
        layout.addWidget(self._occurrences, 1)
        return holder

    def _build_settings(self, parent: QWidget) -> QWidget:
        """The analysis, what the first list shows, and the two thresholds.

        Everything in here changes what is in front of the person rather than
        changing a word, which is why it is gathered in one panel and kept
        well away from the corrections.
        """
        self._settings_group = QGroupBox("Review settings", parent)
        contents = QWidget(self._settings_group)
        inner = QVBoxLayout(contents)
        inner.setContentsMargins(0, 0, 0, 0)

        self._process_button = QPushButton("Process low confidence words", contents)
        describe(
            self._process_button,
            "Process low confidence words",
            "Looks at every word of every transcript in this folder and gathers the "
            "weak ones into words you can decide once.",
        )
        self._note_keys[self._process_button] = PROCESSING

        self._regroup_button = QPushButton("Regroup words", contents)
        describe(
            self._regroup_button,
            "Regroup words",
            "Runs the analysis again with the current minimum confidence and grouping "
            "tolerance. Nothing you have decided is lost.",
        )
        self._note_keys[self._regroup_button] = PROCESSING
        actions = QHBoxLayout()
        actions.addWidget(self._process_button)
        actions.addWidget(self._regroup_button)
        actions.addStretch(1)
        inner.addLayout(actions)

        settings = self._state.settings
        self._show_reviewed_box = QCheckBox("Show reviewed words", contents)
        self._show_reviewed_box.setChecked(settings.show_reviewed)
        describe(
            self._show_reviewed_box,
            "Show reviewed words",
            "Brings the words you have already settled back into the list, each saying "
            "how it was settled.",
        )
        self._note_keys[self._show_reviewed_box] = VISIBILITY

        self._show_uncertainties_box = QCheckBox("Show other uncertainties", contents)
        self._show_uncertainties_box.setChecked(settings.show_other_uncertainties)
        describe(
            self._show_uncertainties_box,
            "Show other uncertainties",
            "Whether the words the transcription flagged for review are listed after "
            "the low confidence words.",
        )
        self._note_keys[self._show_uncertainties_box] = VISIBILITY

        self._auto_play_box = QCheckBox("Play audio auto&matically", contents)
        self._auto_play_box.setChecked(settings.play_automatically)
        describe(
            self._auto_play_box,
            "Play audio automatically",
            "Whether landing on an occurrence plays it without being asked. The focus "
            "stays where it is either way.",
        )
        self._note_keys[self._auto_play_box] = AUTO_PLAY
        for box in (
            self._show_reviewed_box,
            self._show_uncertainties_box,
            self._auto_play_box,
        ):
            inner.addWidget(box)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self._minimum_spin = self._add_percentage(
            form,
            contents,
            "Minimum confidence",
            "Below this, a word is put in front of you. Takes effect when you regroup.",
            round(settings.minimum_confidence * 100),
        )
        self._tolerance_spin = self._add_percentage(
            form,
            contents,
            "Grouping tolerance",
            "How far two confidences may differ and still be treated as the same word. "
            "Takes effect when you regroup.",
            round(settings.grouping_tolerance * 100),
        )
        # The third number, and the only one of the three that is about the
        # person rather than about the words. It sits here with the other two
        # because it is a saved setting of this folder's review and this is
        # where those live, and it is a number a person will come back and
        # adjust once they have heard their own reader work through a few rows.
        self._delay_spin = self._add_number(
            form,
            contents,
            "Wait before playing",
            "How long to wait after landing on an occurrence before the audio starts, "
            "so that the clip does not talk over your screen reader.",
            settings.auto_play_delay_seconds,
            suffix=" seconds",
            maximum=MAXIMUM_AUTO_PLAY_DELAY_SECONDS,
            note_key=AUTO_PLAY_DELAY,
        )
        inner.addLayout(form)

        inner.addWidget(self._build_filters(contents))
        inner.addStretch(1)

        # At a large Windows text size these are taller than the window, so
        # they scroll. Qt brings whatever takes focus into view, so tabbing
        # through still works exactly as it reads.
        scroll = QScrollArea(self._settings_group)
        scroll.setWidget(contents)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        # Both bars are offered as needed, which the horizontal one was not.
        # ``setWidgetResizable`` squeezes the contents to the width of the
        # viewport, and with no horizontal bar to escape by, anything that
        # cannot be squeezed that far is simply cut off with no way to reach
        # it. At a large Windows text size that is the row of buttons at the
        # top of this panel; under a magnifier, where the viewport is a
        # fraction of its nominal width, it is most of the panel.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # A scroll area is a container, not somewhere to land. Giving it the
        # focus would put a stop on the way through that answers no keys.
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        outer = QVBoxLayout(self._settings_group)
        outer.addWidget(scroll)
        return self._settings_group

    def _add_percentage(
        self,
        form: QFormLayout,
        parent: QWidget,
        label_text: str,
        summary: str,
        value: int,
    ) -> QSpinBox:
        """A threshold as whole percentage points, in a spin box."""
        return self._add_number(
            form,
            parent,
            label_text,
            summary,
            value,
            suffix=" percent",
            maximum=100,
            note_key=THRESHOLDS,
        )

    def _add_number(
        self,
        form: QFormLayout,
        parent: QWidget,
        label_text: str,
        summary: str,
        value: int,
        suffix: str,
        maximum: int,
        note_key: str,
    ) -> QSpinBox:
        """A whole number with its unit, in a spin box.

        A spin box rather than a slider. A slider has no value a screen reader
        can read out exactly and no way to set a particular number from the
        keyboard; a spin box says "fifty five percent", answers the arrow keys
        and can be typed into. The unit is part of the value rather than a
        separate label, so the number is never read out on its own.
        """
        label = QLabel(label_text, parent)
        spin = QSpinBox(parent)
        spin.setRange(0, maximum)
        spin.setSuffix(suffix)
        spin.setValue(value)
        describe(spin, label_text, summary)
        # A spin box hands the focus to a line edit inside it, and that field
        # is what a screen reader lands on, so it is named too rather than
        # being read as a blank. The same is true of the editable combo box
        # below.
        if spin.lineEdit() is not None:
            describe(spin.lineEdit(), label_text)
        self._note_keys[spin] = note_key
        label.setBuddy(spin)
        form.addRow(label, spin)
        return spin

    def _build_filters(self, parent: QWidget) -> QWidget:
        """The reason and confidence filters, as real check boxes.

        Check boxes rather than a list of strings in a combo box, because a
        combo box would let one reason be picked at a time and could not show
        which of them are on without being opened.

        They narrow the second block of the first list and nothing else. That
        is not a reduction from what they used to do: the words they filter
        are exactly the words that used to be the whole queue. A low
        confidence word carries no flagged reason to filter on, and the
        minimum confidence above is what decides which of those appear.
        """
        self._filters_group = QGroupBox("Filters on other uncertainties", parent)
        self._reason_boxes: dict[ReviewReason, QCheckBox] = {}
        self._confidence_boxes: dict[Confidence, QCheckBox] = {}

        inner = QVBoxLayout(self._filters_group)

        reasons = QGroupBox("Reasons for review", self._filters_group)
        reasons_layout = QVBoxLayout(reasons)
        for reason in ReviewReason:
            # No Alt letter: there are thirteen of these and the letters
            # would collide with each other and with the buttons. The arrow
            # keys and Tab reach them, which is how a long list of check
            # boxes is worked anywhere in Windows.
            box = QCheckBox(reason.display_name, reasons)
            box.setChecked(True)
            # Named for the reason itself rather than for the group, so a
            # screen reader says which filter is being switched.
            describe(box, reason.display_name, note_for(REASON_FILTERS).summary)
            self._note_keys[box] = REASON_FILTERS
            box.toggled.connect(
                lambda shown, r=reason: self._on_reason_toggled(r, shown)
            )
            self._reason_boxes[reason] = box
            reasons_layout.addWidget(box)

        categories = QGroupBox("Confidence categories", self._filters_group)
        categories_layout = QVBoxLayout(categories)
        for confidence in Confidence:
            box = QCheckBox(confidence.display_name, categories)
            box.setChecked(True)
            describe(box, confidence.display_name, note_for(CONFIDENCE_FILTERS).summary)
            self._note_keys[box] = CONFIDENCE_FILTERS
            box.toggled.connect(
                lambda shown, c=confidence: self._on_confidence_toggled(c, shown)
            )
            self._confidence_boxes[confidence] = box
            categories_layout.addWidget(box)

        self._show_everything_button = QPushButton("Show &everything", self._filters_group)
        describe(
            self._show_everything_button,
            "Show everything",
            "Puts every filter back on, so the whole of the second block is on show "
            "again.",
        )
        self._note_keys[self._show_everything_button] = REASON_FILTERS

        inner.addWidget(reasons)
        inner.addWidget(categories)
        inner.addWidget(self._show_everything_button)
        return self._filters_group

    def _build_right_side(self) -> QWidget:
        contents = QWidget(self)
        inner = QVBoxLayout(contents)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.addWidget(self._build_details(), 3)
        for group in (self._build_playback(), self._build_corrections(), self._build_notes()):
            # Minimum rather than Fixed: each of these takes the height it asks
            # for and no more, because the stretch below soaks up what is left,
            # but it is now allowed to grow when what is inside it grows. Fixed
            # pinned them to the height they happened to want when they were
            # built, which is why the text panels inside them could not follow
            # a change of font.
            group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
            inner.addWidget(group)
        inner.addStretch(1)

        scroll = QScrollArea(self)
        scroll.setWidget(contents)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        return scroll

    def _build_details(self) -> QWidget:
        """The three answers, shown as three things rather than one summary.

        The text, the timing and the speaker each get their own group, with
        their own source and their own confidence, because that is what they
        are. Merging them into one line would hide the very thing the person
        is being asked to decide about.

        The word itself comes first, which it did not have to before. One
        window now holds a whole folder, so which word, in which file, at what
        second, is no longer obvious from the fact that this window is open.
        """
        self._details_group = QGroupBox("Item details", self)
        layout = QVBoxLayout(self._details_group)

        word = QGroupBox("The word itself", self._details_group)
        word_form = QFormLayout(word)
        word_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self._detected_edit = self._add_field(
            word_form, word, "Detected word", "Detected word", WORD_DETAILS
        )
        self._strength_edit = self._add_field(
            word_form, word, "Confidence", "Confidence", WORD_DETAILS
        )
        self._file_edit = self._add_field(
            word_form, word, "Source file", "Source file", WORD_DETAILS
        )
        self._position_edit = self._add_field(
            word_form, word, "Position", "Position", WORD_DETAILS
        )
        self._word_language_edit = self._add_field(
            word_form, word, "Language of this word", "Language of this word", WORD_DETAILS
        )
        layout.addWidget(word)

        said = QGroupBox("What was said", self._details_group)
        said_layout = QVBoxLayout(said)
        self._candidates_model = CandidateTableModel(self)
        self._candidates = QTableView(said)
        self._candidates.setModel(self._candidates_model)
        self._candidates.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._candidates.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._candidates.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._candidates.setAlternatingRowColors(True)
        self._candidates.setWordWrap(False)
        self._candidates.setTabKeyNavigation(False)
        self._candidates.verticalHeader().setVisible(False)
        candidate_header = self._candidates.horizontalHeader()
        candidate_header.setSectionsClickable(False)
        candidate_header.setHighlightSections(False)
        candidate_header.setSectionResizeMode(
            CANDIDATE_COLUMN_TEXT, QHeaderView.ResizeMode.Stretch
        )
        self._explain(self._candidates, CANDIDATES)
        said_layout.addWidget(self._candidates)

        candidate_form = QFormLayout()
        candidate_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self._candidate_text = self._add_field(
            candidate_form, said, "Candi&date text", "Candidate text", CANDIDATE_TEXT
        )
        said_layout.addLayout(candidate_form)

        self._use_candidate_button = QPushButton("&Use this candidate", said)
        describe(
            self._use_candidate_button,
            "Use this candidate",
            "Copies the highlighted candidate into the replacement box, ready to apply.",
        )
        self._note_keys[self._use_candidate_button] = CANDIDATE_TEXT
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self._use_candidate_button, 0)
        said_layout.addLayout(row)
        layout.addWidget(said, 1)

        when = QGroupBox("When it was said", self._details_group)
        when_form = QFormLayout(when)
        when_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self._timing_edit = self._add_field(when_form, when, "Timing", "Timing", TIMING)
        self._timing_status_edit = self._add_field(
            when_form, when, "Timing status", "Timing status", TIMING
        )
        self._timing_source_edit = self._add_field(
            when_form, when, "Timing source", "Timing source", TIMING
        )
        self._timing_confidence_edit = self._add_field(
            when_form, when, "Timing confidence", "Timing confidence", TIMING
        )
        self._alignment_edit = self._add_field(
            when_form, when, "How it matched up", "How it matched up", TIMING
        )
        layout.addWidget(when)

        who = QGroupBox("Who said it", self._details_group)
        who_form = QFormLayout(who)
        who_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self._speaker_edit = self._add_field(who_form, who, "Speaker", "Speaker", SPEAKER)
        self._speaker_source_edit = self._add_field(
            who_form, who, "Speaker source", "Speaker source", SPEAKER
        )
        self._speaker_confidence_edit = self._add_field(
            who_form, who, "Speaker confidence", "Speaker confidence", SPEAKER
        )
        layout.addWidget(who)

        rest = QGroupBox("Language, risk and decisions", self._details_group)
        rest_form = QFormLayout(rest)
        rest_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self._language_edit = self._add_field(rest_form, rest, "Language", "Language", LANGUAGE)
        self._language_evidence_edit = self._add_field(
            rest_form, rest, "Language evidence", "Language evidence", LANGUAGE
        )
        self._risk_edit = self._add_field(
            rest_form, rest, "Risk categories", "Risk categories", RISK
        )
        # A word can be both weak and flagged. It appears once, among the low
        # confidence words, and this is where the reasons it was also flagged
        # for are shown, so that neither road to the same word hides what the
        # other one knew about it.
        self._reasons_edit = self._add_field(
            rest_form, rest, "Why it needs review", "Why it needs review", RISK
        )

        decision_label = QLabel("What the language model decided", rest)
        self._decision_text = QPlainTextEdit(rest)
        self._decision_text.setReadOnly(True)
        self._decision_text.setMinimumHeight(self.fontMetrics().lineSpacing() * 4)
        self._decision_text.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        self._explain(self._decision_text, DECISION)
        decision_label.setBuddy(self._decision_text)
        rest_form.addRow(decision_label, self._decision_text)
        layout.addWidget(rest)
        return self._details_group

    def _add_field(
        self,
        form: QFormLayout,
        parent: QWidget,
        label_text: str,
        accessible_name: str,
        note_key: str,
    ) -> QLineEdit:
        """Add a value that can be read closely.

        A read-only line edit rather than a label. It takes focus, so the
        value can be reached with the Tab key, read a character at a time
        and copied, and it shows a real text caret that ZoomText can track.
        A label offers none of that.
        """
        label = QLabel(label_text, parent)
        edit = QLineEdit(parent)
        edit.setReadOnly(True)
        describe(edit, accessible_name, note_for(note_key).summary)
        self._note_keys[edit] = note_key
        label.setBuddy(edit)
        form.addRow(label, edit)
        return edit

    def _build_playback(self) -> QWidget:
        self._playback_group = QGroupBox("Playback", self)
        layout = QVBoxLayout(self._playback_group)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self._span_edit = self._add_field(
            form, self._playback_group, "What will be played", "What will be played", PLAYBACK
        )
        layout.addLayout(form)

        # Y rather than G, which the Word Groups label now has. Nothing else in
        # the window answers Y.
        self._play_button = QPushButton("Pla&y the word again", self._playback_group)
        describe(
            self._play_button,
            "Play the word again",
            f"Plays the word with {CONTEXT_SECONDS:.0f} seconds either side of it. F5.",
        )
        self._note_keys[self._play_button] = PLAYBACK

        self._play_wide_button = QPushButton("Play with more conte&xt", self._playback_group)
        describe(
            self._play_wide_button,
            "Play with more context",
            f"Plays the word with {WIDE_CONTEXT_SECONDS:.0f} seconds either side of it, "
            "for when the sentence alone is not enough. Shift+F5.",
        )
        self._note_keys[self._play_wide_button] = PLAYBACK

        row = QHBoxLayout()
        row.addWidget(self._play_button)
        row.addWidget(self._play_wide_button)
        row.addStretch(1)
        layout.addLayout(row)
        return self._playback_group

    def _build_corrections(self) -> QWidget:
        """The corrections, grouped by what they act on rather than by kind.

        The replacement acts on a word and reaches every file it occurs in.
        The speaker and the timing act on one occurrence. Keeping the two
        apart on the screen is the same separation the whole design rests on,
        and the field between them says, in counted numbers, exactly how far a
        change to the word would reach.
        """
        self._corrections_group = QGroupBox("Corrections", self)
        layout = QVBoxLayout(self._corrections_group)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        text_label = QLabel("Rep&lacement", self._corrections_group)
        self._replacement_edit = QLineEdit(self._corrections_group)
        describe(
            self._replacement_edit,
            "Replacement",
            "What this word should say. Press Enter, or use Apply to this word, to "
            "change every occurrence of it. The timing and the speaker are left alone. "
            "F2 comes here.",
        )
        self._note_keys[self._replacement_edit] = REPLACEMENT
        text_label.setBuddy(self._replacement_edit)
        form.addRow(text_label, self._replacement_edit)

        self._affected_edit = self._add_field(
            form,
            self._corrections_group,
            "What a change to this word would affect",
            "What a change to this word would affect",
            REPLACEMENT,
        )

        speaker_label = QLabel("Spea&ker", self._corrections_group)
        self._speaker_box = QComboBox(self._corrections_group)
        self._speaker_box.setEditable(True)
        describe(
            self._speaker_box,
            "Speaker",
            "Who said this one occurrence. Choose a speaker found in the recording, or "
            "type a label of your own. The text and the timing are left alone.",
        )
        # An editable combo box hands the focus to a line edit inside it,
        # and that field is what a screen reader lands on, so it is named
        # too rather than being read as a blank.
        if self._speaker_box.lineEdit() is not None:
            describe(self._speaker_box.lineEdit(), "Speaker")
        self._note_keys[self._speaker_box] = CORRECT_SPEAKER
        speaker_label.setBuddy(self._speaker_box)
        form.addRow(speaker_label, self._speaker_box)
        layout.addLayout(form)

        self._apply_word_button = QPushButton("&Apply to this word", self._corrections_group)
        describe(
            self._apply_word_button,
            "Apply to this word",
            "Replaces every occurrence of this word, in every recording. The speaker "
            "and the timing of each of them are left exactly as they were.",
        )
        self._note_keys[self._apply_word_button] = REPLACEMENT

        self._apply_occurrence_button = QPushButton(
            "Apply to this occurrence only", self._corrections_group
        )
        describe(
            self._apply_occurrence_button,
            "Apply to this occurrence only",
            "Replaces this one occurrence and leaves the rest of the word alone. This "
            "occurrence then keeps its own answer if the word's changes later.",
        )
        self._note_keys[self._apply_occurrence_button] = REPLACEMENT

        self._correct_as_detected_button = QPushButton(
            "&Correct as detected", self._corrections_group
        )
        describe(
            self._correct_as_detected_button,
            "Correct as detected",
            "Says every occurrence of this word is right as the services produced it, "
            "and settles them all.",
        )
        self._note_keys[self._correct_as_detected_button] = CONFIRM

        # T rather than I, which belongs to the Item menu.
        self._isolate_button = QPushButton("Isolate &this occurrence", self._corrections_group)
        describe(
            self._isolate_button,
            "Isolate this occurrence",
            "Takes this occurrence out of its word so it can be decided on its own. "
            "Regrouping never gathers it back up.",
        )
        self._note_keys[self._isolate_button] = ISOLATE

        # H rather than P, which belongs to the Project menu, and this is the
        # collision that was actually dangerous. Alt+P reaching for a menu
        # instead pressed this button, which attributes the word under the
        # highlight to whatever is in the speaker box and writes the transcript
        # immediately, because there is no Save button on this screen to give
        # anybody a moment to notice.
        self._apply_speaker_button = QPushButton("Apply t&he speaker", self._corrections_group)
        describe(
            self._apply_speaker_button,
            "Apply the speaker",
            "Attributes this one occurrence to the chosen speaker. The text and the "
            "timing are left exactly as they were.",
        )
        self._note_keys[self._apply_speaker_button] = CORRECT_SPEAKER

        self._confirm_timing_button = QPushButton("Con&firm the timing", self._corrections_group)
        describe(
            self._confirm_timing_button,
            "Confirm the timing",
            "Says the span is right. Reasons that are about the words are left "
            "standing, so the item stays in the list if anything else is unsettled.",
        )
        self._note_keys[self._confirm_timing_button] = TIMING_DECISION

        self._reject_timing_button = QPushButton("&Reject the timing", self._corrections_group)
        describe(
            self._reject_timing_button,
            "Reject the timing",
            "Says the span is wrong. The numbers are kept so the word can still be "
            "played, but they are marked uncertain and the item stays in the list.",
        )
        self._note_keys[self._reject_timing_button] = TIMING_DECISION

        self._confirm_button = QPushButton(
            "Confirm this occurrence as correct", self._corrections_group
        )
        describe(
            self._confirm_button,
            "Confirm this occurrence as correct",
            "Leaves this occurrence exactly as it is and settles it. F4.",
        )
        self._note_keys[self._confirm_button] = CONFIRM

        self._next_button = QPushButton("&Next occurrence", self._corrections_group)
        describe(
            self._next_button, "Next occurrence", "Moves to the next occurrence. F3."
        )
        self._note_keys[self._next_button] = NAVIGATE

        self._previous_button = QPushButton("Previous occurrence", self._corrections_group)
        describe(
            self._previous_button,
            "Previous occurrence",
            "Moves to the previous occurrence. Shift+F3.",
        )
        self._note_keys[self._previous_button] = NAVIGATE

        for buttons in (
            (self._apply_word_button, self._apply_occurrence_button),
            (self._correct_as_detected_button, self._isolate_button),
            (self._apply_speaker_button,),
            (self._confirm_timing_button, self._reject_timing_button),
            (self._confirm_button,),
            (self._previous_button, self._next_button),
        ):
            row = QHBoxLayout()
            for button in buttons:
                row.addWidget(button)
            row.addStretch(1)
            layout.addLayout(row)
        return self._corrections_group

    def _build_notes(self) -> QWidget:
        """The panel that explains whichever control has the focus.

        This screen has more jargon on it than any other in the
        application: timing states, alignment states, confidence categories
        and risk categories all mean something particular. None of it fits
        on a label, and hearing a paragraph read out every time the focus
        moves would be unusable, so the short summary goes on the control
        and the full explanation goes here, following the focus.

        Moving into this panel to read it leaves the note where it is.
        Otherwise the act of reading a note would change it.
        """
        self._notes_group = QGroupBox("About this control", self)
        self._notes_text = QPlainTextEdit(self._notes_group)
        self._notes_text.setReadOnly(True)
        self._notes_text.setMinimumHeight(self.fontMetrics().lineSpacing() * 7)
        self._notes_text.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        describe(
            self._notes_text,
            "About this control",
            "Explains the control the focus is on. Use the arrow keys to read through "
            "it. It stays on the last control while you read.",
        )
        layout = QVBoxLayout(self._notes_group)
        layout.addWidget(self._notes_text)
        return self._notes_group

    def _explain(self, widget: QWidget, key: str) -> QWidget:
        """Name a control from its note, and have the panel follow it."""
        note = note_for(key)
        describe(widget, note.title, note.summary)
        self._note_keys[widget] = key
        return widget

    def _build_menus(self) -> None:
        """Put every action on a menu as well as on a control.

        The controls are spread down two sides of a wide window, a long way
        from either list. A menu bar is where a screen reader user looks for
        what a window can do, and it is where the shortcut keys are written
        down rather than having to be known. The two thresholds appear here as
        well, as entries that move the focus to them, so that nothing this
        window can do is reachable only by finding a control by sight.
        """
        menu_bar = self.menuBar()

        project_menu = menu_bar.addMenu("&Project")
        self._process_action = self._add_action(
            project_menu,
            "&Process Low Confidence Words",
            QKeySequence("Ctrl+L"),
            self.process_low_confidence_words,
        )
        self._regroup_action = self._add_action(
            project_menu, "Re&group Words", QKeySequence("Ctrl+G"), self.regroup_words
        )
        project_menu.addSeparator()
        self._minimum_action = self._add_action(
            project_menu, "Set the &Minimum Confidence", None, self.focus_minimum_confidence
        )
        self._tolerance_action = self._add_action(
            project_menu, "Set the Grouping &Tolerance", None, self.focus_grouping_tolerance
        )
        self._delay_action = self._add_action(
            project_menu, "Set the &Wait before Playing", None, self.focus_auto_play_delay
        )

        word_menu = menu_bar.addMenu("&Word")
        self._next_group_action = self._add_action(
            word_menu, "&Next Word", QKeySequence("Ctrl+F3"), self.go_to_next_group
        )
        self._previous_group_action = self._add_action(
            word_menu, "&Previous Word", QKeySequence("Ctrl+Shift+F3"), self.go_to_previous_group
        )
        word_menu.addSeparator()
        self._correct_text_action = self._add_action(
            word_menu,
            "Edit the &Replacement",
            QKeySequence(Qt.Key.Key_F2),
            self.focus_replacement,
        )
        self._apply_word_action = self._add_action(
            word_menu, "&Apply to This Word", None, self.apply_replacement_to_word
        )
        self._correct_as_detected_action = self._add_action(
            word_menu, "&Correct as Detected", None, self.correct_word_as_detected
        )

        item_menu = menu_bar.addMenu("&Item")
        self._next_action = self._add_action(
            item_menu, "&Next Occurrence", QKeySequence(Qt.Key.Key_F3), self.go_to_next_item
        )
        self._previous_action = self._add_action(
            item_menu, "&Previous Occurrence", QKeySequence("Shift+F3"), self.go_to_previous_item
        )
        item_menu.addSeparator()
        self._apply_occurrence_action = self._add_action(
            item_menu,
            "Apply to This &Occurrence Only",
            None,
            self.apply_replacement_to_occurrence,
        )
        self._apply_speaker_action = self._add_action(
            item_menu, "Apply the &Speaker", None, self.apply_speaker_correction
        )
        self._isolate_action = self._add_action(
            item_menu, "&Isolate This Occurrence", QKeySequence("Ctrl+I"), self.isolate_occurrence
        )
        item_menu.addSeparator()
        self._confirm_timing_action = self._add_action(
            item_menu, "Confirm the T&iming", None, lambda: self.decide_timing(True)
        )
        self._reject_timing_action = self._add_action(
            item_menu, "&Reject the Timing", None, lambda: self.decide_timing(False)
        )
        self._confirm_action = self._add_action(
            item_menu,
            "&Confirm This Occurrence as Correct",
            QKeySequence(Qt.Key.Key_F4),
            self.confirm_item,
        )

        playback_menu = menu_bar.addMenu("Play&back")
        self._play_action = self._add_action(
            playback_menu,
            "Play the &Word Again",
            QKeySequence(Qt.Key.Key_F5),
            lambda: self.play_span(False),
        )
        self._play_wide_action = self._add_action(
            playback_menu,
            "Play with &More Context",
            QKeySequence("Shift+F5"),
            lambda: self.play_span(True),
        )
        playback_menu.addSeparator()
        self._auto_play_action = self._add_toggle(
            playback_menu,
            "Play &Automatically",
            self._state.settings.play_automatically,
            self.set_play_automatically,
        )

        view_menu = menu_bar.addMenu("&View")
        self._next_panel_action = self._add_action(
            view_menu, "&Next Panel", QKeySequence(Qt.Key.Key_F6), self.focus_next_panel
        )
        # Shift+F6 for the other direction, which is the Windows convention and
        # was simply missing. A person who overshoots a panel had no way back
        # except going round all seven again.
        self._previous_panel_action = self._add_action(
            view_menu,
            "&Previous Panel",
            QKeySequence("Shift+F6"),
            self.focus_previous_panel,
        )
        view_menu.addSeparator()
        self._widen_lists_action = self._add_action(
            view_menu,
            "Widen the &Lists",
            QKeySequence("Ctrl+Shift+Left"),
            lambda: self.adjust_divider(-1),
        )
        self._widen_details_action = self._add_action(
            view_menu,
            "Widen the &Details",
            QKeySequence("Ctrl+Shift+Right"),
            lambda: self.adjust_divider(1),
        )
        view_menu.addSeparator()
        self._show_reviewed_action = self._add_toggle(
            view_menu,
            "Show &Reviewed Words",
            self._state.settings.show_reviewed,
            self.set_show_reviewed,
        )
        self._show_uncertainties_action = self._add_toggle(
            view_menu,
            "Show Other &Uncertainties",
            self._state.settings.show_other_uncertainties,
            self.set_show_other_uncertainties,
        )
        self._add_action(view_menu, "Show &Everything", None, self.show_everything)

    def _add_action(self, menu, text: str, shortcut, slot) -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def _add_toggle(self, menu, text: str, checked: bool, slot) -> QAction:
        """A menu entry that is on or off, and says which.

        A checkable action rather than an entry whose wording flips between
        "Show" and "Hide". Qt exposes the checked state to a screen reader, so
        the entry is read as on or off without a listener having to work it
        out from the verb.
        """
        action = QAction(text, self)
        action.setCheckable(True)
        action.setChecked(checked)
        action.toggled.connect(slot)
        menu.addAction(action)
        return action

    def _connect_signals(self) -> None:
        # Connected after everything is built, so that setting the initial
        # value of a check box or a spin box during the build does not look
        # like the person changing it.
        self._groups.selectionModel().currentRowChanged.connect(self._on_group_row_changed)
        self._occurrences.selectionModel().currentRowChanged.connect(
            self._on_occurrence_row_changed
        )
        self._candidates.selectionModel().currentRowChanged.connect(self._on_candidate_changed)

        self._process_button.clicked.connect(self.process_low_confidence_words)
        self._regroup_button.clicked.connect(self.regroup_words)
        self._show_reviewed_box.toggled.connect(self.set_show_reviewed)
        self._show_uncertainties_box.toggled.connect(self.set_show_other_uncertainties)
        self._auto_play_box.toggled.connect(self.set_play_automatically)
        # Two signals each, and which does what is the whole point. The value
        # is kept up to date on every intermediate number, silently, so that
        # nothing else in the window can ever read a stale threshold. The
        # saving and the saying wait for the person to finish, because
        # valueChanged fires on every keystroke and every repeat of a held
        # arrow key: typing "55" used to give two full sentences and holding
        # the arrow from nought to sixty gave sixty, which is minutes of
        # speech that cannot be interrupted or typed through, and sixty writes
        # of the project file to go with it.
        self._minimum_spin.valueChanged.connect(self._on_minimum_changed)
        self._minimum_spin.editingFinished.connect(self._on_minimum_settled)
        self._tolerance_spin.valueChanged.connect(self._on_tolerance_changed)
        self._tolerance_spin.editingFinished.connect(self._on_tolerance_settled)
        self._delay_spin.valueChanged.connect(self._on_delay_changed)
        self._delay_spin.editingFinished.connect(self._on_delay_settled)
        self._show_everything_button.clicked.connect(self.show_everything)

        self._use_candidate_button.clicked.connect(self.use_selected_candidate)
        self._play_button.clicked.connect(lambda: self.play_span(False))
        self._play_wide_button.clicked.connect(lambda: self.play_span(True))
        self._apply_word_button.clicked.connect(self.apply_replacement_to_word)
        self._replacement_edit.returnPressed.connect(self.apply_replacement_to_word)
        self._apply_occurrence_button.clicked.connect(self.apply_replacement_to_occurrence)
        self._correct_as_detected_button.clicked.connect(self.correct_word_as_detected)
        self._isolate_button.clicked.connect(self.isolate_occurrence)
        self._apply_speaker_button.clicked.connect(self.apply_speaker_correction)
        self._confirm_timing_button.clicked.connect(lambda: self.decide_timing(True))
        self._reject_timing_button.clicked.connect(lambda: self.decide_timing(False))
        self._confirm_button.clicked.connect(self.confirm_item)
        self._next_button.clicked.connect(self.go_to_next_item)
        self._previous_button.clicked.connect(self.go_to_previous_item)

        self._play_timer.timeout.connect(self._play_after_waiting)
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.errorOccurred.connect(self._on_player_error)

    # -- The focus, and the note that follows it -------------------------

    def showEvent(self, event) -> None:
        """Start watching the focus while the window is actually on screen."""
        super().showEvent(event)
        self._watch_focus(True)

    def changeEvent(self, event) -> None:
        """Follow a change of text size rather than staying at the first one.

        The two text panels are several lines tall, and how tall a line is
        depends on the font. Both were measured once, while the window was
        being built, and pinned there. Somebody who then turned Windows text
        size up -- which is the first thing a person with low vision does, and
        which they may well do with this window already open -- got larger
        letters in panels that had not grown to hold them, so the notes they
        were reading lost their last two lines behind the bottom edge.
        """
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange:
            self._size_the_text_panels()

    def _size_the_text_panels(self) -> None:
        """Give the two read-only text panels a floor measured in lines of text."""
        spacing = self.fontMetrics().lineSpacing()
        self._decision_text.setMinimumHeight(spacing * 4)
        self._notes_text.setMinimumHeight(spacing * 7)

    def hideEvent(self, event) -> None:
        """Stop watching as soon as it is not.

        The watch is on the application, which outlives this window. Left
        connected, every review window ever opened would go on looking at
        every focus change anywhere in the application for the rest of its
        life.
        """
        self._watch_focus(False)
        super().hideEvent(event)

    def closeEvent(self, event) -> None:
        """Stop the audio, let the file go, and put the project away.

        Where the person had got to is written down here rather than on every
        arrow press, because it is the one piece of state that changes
        constantly and matters only once. Everything they actually decided has
        already been saved at the moment they decided it.

        The player is stopped and cleared, which used to be done only when the
        main window was the one closing this one. Pressing this window's own
        close button part-way through a twelve-second clip left the sound
        playing with no window on screen to stop it, and left the player
        holding an open handle on the recording, so the person could not then
        move, rename or delete the file they had just been listening to and
        nothing anywhere would tell them why. Clearing the source rather than
        merely stopping is what actually releases the handle.

        A project save that fails here refuses the close, once. Only the resume
        markers are at stake -- every decision was written through when it was
        made -- so this is the mildest failure in the window, but announcing it
        while the window is on its way off the screen is announcing it to
        nobody. Refusing keeps the window there for the message to be read
        from. Asking a second time closes anyway, because a window somebody
        cannot get out of would be a worse fault than the one being reported.
        """
        self._play_timer.stop()
        self._pending_play = None
        self._player.stop()
        self._player.load(None)
        self._loaded_path = None
        self._remember_place()
        if not self._save_project(quiet=True) and not self._close_refused:
            self._close_refused = True
            self._set_status(
                f"{self._save_failure_text()} Every decision you made is already in the "
                "transcripts; what cannot be kept is the mark saying where you had got "
                "to. Close the window again to close it anyway.",
                alert=True,
                urgent=True,
            )
            event.ignore()
            return
        super().closeEvent(event)

    def _watch_focus(self, watching: bool) -> None:
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

        The search walks up from the focused widget, because the widget
        that takes the focus is often a part of a control rather than the
        control itself: an editable combo box hands it to the field inside
        it. Walking up finds the control the user thinks they are on.
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

    def _panels(self) -> list[tuple[str, QWidget, object]]:
        """The panels F6 walks, in the order they are read on the screen.

        Each names the control the focus should land on. Calling setFocus on
        the panel itself is no good: a plain container keeps the focus and
        answers no keys, while a group box passes it to a child that is not
        always the first one.
        """
        return [
            ("Word Groups", self._groups, self._focus_groups),
            ("Occurrences", self._occurrences, self._focus_occurrences),
            ("Review settings", self._settings_group, self._focus_settings),
            ("Item details", self._details_group, self._focus_details),
            ("Playback", self._playback_group, self._focus_playback),
            ("Corrections", self._corrections_group, self._focus_corrections),
            ("About this control", self._notes_group, self._focus_notes),
        ]

    def focus_next_panel(self) -> None:
        """Move the focus to the panel after this one."""
        self._step_panel(1)

    def focus_previous_panel(self) -> None:
        """Move the focus to the panel before this one, which F6 alone cannot.

        Shift+F6 is what Windows does everywhere else and it was missing here,
        so overshooting a panel meant going round all seven of them again.
        """
        self._step_panel(-1)

    def _step_panel(self, delta: int) -> None:
        """Move the focus one panel, and put the panel's name on the screen.

        The name goes to the status label and is not announced. Moving the
        focus makes the screen reader read out the control that has been
        landed on, and its group box along with it, so an announcement raised
        at the same moment arrives on top of the very sentence that answers
        the question it was trying to answer. The label is there for anyone
        who wants to look, and for a magnifier user whose view of the window is
        a few square inches wide and who genuinely cannot see where they are.
        """
        panels = self._panels()
        focused = self.focusWidget()
        current_index = -1
        for index, (_name, panel, _focus) in enumerate(panels):
            if focused is not None and (focused is panel or panel.isAncestorOf(focused)):
                current_index = index
                break
        if current_index < 0 and delta < 0:
            # Nowhere in particular, and asked to go back: the last panel is
            # what going back from the top means.
            current_index = 0
        name, _panel, focus = panels[(current_index + delta) % len(panels)]
        focus()
        self._set_status(name)

    def adjust_divider(self, direction: int) -> None:
        """Give the lists or the detail panel more of the window, from the keyboard.

        A splitter is dragged with the mouse, and that is all it is: Qt gives
        its handles no keyboard behaviour of their own. Widening one side is
        exactly the adjustment somebody working under a magnifier wants most,
        because at four times magnification a panel a third of the window wide
        holds about two words, and it was the one thing in this window that
        could not be done without a mouse at all.

        The step is a tenth of the window, which is big enough to be worth a
        keystroke and small enough to be steered by.
        """
        sizes = self._splitter.sizes()
        if len(sizes) != 2 or sum(sizes) <= 0:
            return
        step = max(1, sum(sizes) // 10)
        step = min(step, sizes[1] - 1) if direction > 0 else -min(step, sizes[0] - 1)
        if step == 0:
            return
        self._splitter.setSizes([sizes[0] + step, sizes[1] - step])
        share = round(self._splitter.sizes()[0] * 100 / sum(self._splitter.sizes()))
        self._set_status(f"The lists now take {share} percent of the window.", alert=True)

    def focus_word_list(self) -> None:
        """Put the focus on the first list, Word Groups.

        For whoever opens this window and wants the person put somewhere
        chosen rather than wherever Qt happens to leave the focus when a window
        is shown. Word Groups is the right somewhere: it is where the work is
        picked, the row it lands on has already been selected, so the reading
        names the word the person has been brought to, and the occurrences of
        that word are one Tab, one F6 or one F3 away.
        """
        self._focus_groups()

    def focus_occurrence_list(self) -> None:
        """Put the focus on the second list, Occurrences."""
        self._focus_occurrences()

    def _focus_groups(self) -> None:
        self._groups.setFocus(Qt.FocusReason.TabFocusReason)

    def _focus_occurrences(self) -> None:
        self._occurrences.setFocus(Qt.FocusReason.TabFocusReason)

    def _focus_settings(self) -> None:
        self._process_button.setFocus(Qt.FocusReason.TabFocusReason)

    def _focus_details(self) -> None:
        self._detected_edit.setFocus(Qt.FocusReason.TabFocusReason)

    def _focus_playback(self) -> None:
        self._focus_first_enabled([self._play_button, self._play_wide_button, self._span_edit])

    def _focus_corrections(self) -> None:
        self._focus_first_enabled(
            [self._replacement_edit, self._confirm_button, self._next_button]
        )

    def _focus_notes(self) -> None:
        self._notes_text.setFocus(Qt.FocusReason.TabFocusReason)

    def _focus_first_enabled(self, widgets: list[QWidget]) -> None:
        """Land on the first of these that can actually take the focus.

        The controls that act on an item are switched off when there is no
        item, and a disabled control cannot be focused, so F6 would
        otherwise leave the focus wherever it was and say it had moved.
        """
        for widget in widgets:
            if widget.isEnabled():
                widget.setFocus(Qt.FocusReason.TabFocusReason)
                return
        self._focus_groups()

    def focus_replacement(self) -> None:
        """Put the focus in the replacement box, with the word ready to replace."""
        if not self._replacement_edit.isEnabled():
            return
        self._replacement_edit.setFocus(Qt.FocusReason.OtherFocusReason)
        self._replacement_edit.selectAll()

    def focus_minimum_confidence(self) -> None:
        self._minimum_spin.setFocus(Qt.FocusReason.OtherFocusReason)
        self._minimum_spin.selectAll()

    def focus_grouping_tolerance(self) -> None:
        self._tolerance_spin.setFocus(Qt.FocusReason.OtherFocusReason)
        self._tolerance_spin.selectAll()

    def focus_auto_play_delay(self) -> None:
        self._delay_spin.setFocus(Qt.FocusReason.OtherFocusReason)
        self._delay_spin.selectAll()

    # -- What the window is showing ---------------------------------------

    @property
    def recording_names(self) -> list[str]:
        """Every recording in the folder that has a transcript worth reading."""
        return list(self._recording_names)

    @property
    def state(self):
        """The project as it stands. Every decision made here is already in it."""
        return self._state

    def _transcript(self, recording_name: str) -> Transcript | None:
        """The transcript of one recording, read now or remembered from before.

        Exactly one transcript is held, and it is the one the person is
        working in, because they will settle several occurrences of the same
        recording before moving on and re-reading tens of megabytes on every
        arrow key would be its own kind of slow. Moving to another recording
        lets the previous one go.

        A ``None`` from the loader is remembered as well as a transcript. It
        means the file cannot be read just now, which is a state the window
        has to be able to show rather than retry silently on every keystroke;
        moving away and back is what asks again.
        """
        if recording_name == self._cached_name:
            return self._cached_transcript
        self._cached_name = recording_name
        self._cached_transcript = self._load_transcript(recording_name)
        return self._cached_transcript

    def transcript_for(self, recording_name: str) -> Transcript | None:
        """One recording's transcript, corrections included. Reads it if need be."""
        return self._transcript(recording_name)

    @property
    def current_transcript(self) -> Transcript | None:
        """The transcript the selected occurrence belongs to."""
        occurrence = self.current_occurrence()
        if occurrence is None:
            return None
        return self._transcript(occurrence.recording_name)

    def current_row(self) -> GroupRow | None:
        return self._group_model.row_at(self._groups.selected_row())

    def current_occurrence(self) -> Occurrence | None:
        return self._occurrence_model.occurrence_at(self._occurrences.selected_row())

    def current_token(self) -> FinalToken | None:
        """The word of the transcript the selected occurrence points at.

        ``None`` where the occurrence has gone stale, which is to say where
        the recording was transcribed again and the word could not be found.
        Every correction checks this before doing anything, because a
        correction with nothing to correct must say so rather than appearing
        to work.
        """
        occurrence = self.current_occurrence()
        if occurrence is None:
            return None
        transcript = self._transcript(occurrence.recording_name)
        if transcript is None:
            return None
        return transcript.token_by_id(occurrence.token_id)

    def _restore_flagged(self) -> None:
        """Fill the second block from the project, reading nothing.

        The window opens on what the project remembers of every recording it
        has ever analysed, so the block is whole from the first moment rather
        than filling in as files happen to be read. What each item can say
        about itself is a summary, and :func:`restored_token` is explicit about
        which fields are real and which are left at their defaults until the
        person selects the row and the transcript is loaded for the detail
        panel.

        A recording read later in this session replaces its own items here,
        which is how a summary gives way to the real words without anything
        else in the window having to know the difference.

        Nothing is checked against a transcript on the way in, because checking
        would mean reading the folder and reading the folder is the whole cost
        being avoided. An item pointing at a word that has since gone is caught
        instead at the moment the person lands on it: the transcript is loaded
        for the detail panel in any case, the word is not found in it, and the
        panel says so and refuses every correction rather than appearing to
        work. That is a check paid for by a read that was going to happen.
        """
        self._uncertain = []
        self._settled = []
        for item in self._state.flagged:
            token = restored_token(item)
            target = self._settled if item.settled else self._uncertain
            target.append((item.recording_name, token))
        self._sort_uncertainties()

    def _note_uncertainties(self, recording_name: str, transcript: Transcript) -> None:
        """Remember which words of one recording are flagged, and which are settled.

        What belongs in the block is decided by :func:`flagged_in` and nowhere
        else, and this then keeps the real word behind each of the ones it
        named. Deciding it again here would be two rules that have to agree
        about which words a person still has to look at, and the day they stop
        agreeing is the day the window and the file it saves describe different
        folders.

        What is kept is the tokens themselves rather than the transcript they
        came from, which is a few hundred of the eight thousand words of an
        hour of speech. That is what lets the second block of the first list
        span a whole folder without the folder being held in memory.

        The settled ones are read off the transcript rather than remembered
        from the session, so that a word confirmed last week still says so
        today: a confirmation is written into the transcript as a review
        status, and a correction as the text it replaced.

        The same words go into the project on the way past, and that is what
        makes the block survive the window closing. They are written in full
        for this recording, replacing whatever was there, rather than being
        merged into it. Merging would be the natural-looking thing and it is
        the wrong thing: a word that has gone from a regenerated transcript
        would then stay in the project for ever as a row pointing at nothing,
        and this is the only moment anything is in a position to notice that it
        has gone. Replacing wholesale is safe because a flagged word is a fact
        about the transcript rather than a decision anybody made, so there is
        nothing here worth the careful re-matching an occurrence gets.
        """
        self._scanned.add(recording_name)
        self._uncertain = [
            pair for pair in self._uncertain if pair[0] != recording_name
        ]
        self._settled = [pair for pair in self._settled if pair[0] != recording_name]
        items = flagged_in(recording_name, transcript)
        words = {token.id: token for token in transcript.tokens}
        for item in items:
            target = self._settled if item.settled else self._uncertain
            target.append((recording_name, words[item.token_id]))
        self._sort_uncertainties()
        self._state.flagged = [
            item for item in self._state.flagged if item.recording_name != recording_name
        ] + items

    def _sort_uncertainties(self) -> None:
        """Put the block in the order somebody works down it, and tell the filters.

        Kept in the order of the recordings and then of the clock, so that the
        block reads as a walk through the folder rather than as a heap in the
        order the files happened to be read or corrected in.
        """
        self._uncertain.sort(key=lambda pair: (pair[0], pair[1].start or 0.0))
        self._settled.sort(key=lambda pair: (pair[0], pair[1].start or 0.0))
        self._queue_model.set_tokens([token for _name, token in self._uncertain])

    def _visible_uncertainties(self) -> list[tuple[str, FinalToken]]:
        """The flagged words the filters are letting through, plus settled ones.

        The settled words are added without being filtered, and deliberately.
        A word somebody confirmed carries no review reasons any more, so every
        reason filter would hide it, and asking to see reviewed words would
        then show nothing at all.
        """
        allowed = {id(token) for token in self._queue_model.visible_tokens()}
        return [pair for pair in self._uncertainty_block() if id(pair[1]) in allowed] + (
            [pair for pair in self._settled if not self._is_a_word(pair)]
            if self._state.settings.show_reviewed
            else []
        )

    def _uncertainty_block(self) -> list[tuple[str, FinalToken]]:
        """Every flagged word that is not already in the list as a weak word.

        A word can be both quietly weak and flagged for a stated reason. It
        appears once, among the low confidence words, with the reasons it was
        flagged for shown in its details. Two rows for one word would have a
        person decide it twice and give them every chance to disagree with
        themselves.
        """
        return [pair for pair in self._uncertain if not self._is_a_word(pair)]

    def _is_a_word(self, pair: tuple[str, FinalToken]) -> bool:
        """Whether a project occurrence already speaks for this token."""
        recording_name, token = pair
        return any(
            occurrence.recording_name == recording_name and occurrence.token_id == token.id
            for occurrence in self._state.occurrences
        )

    def refresh(self, word_key: str | None = None, occurrence_id: str | None = None) -> None:
        """Rebuild both lists from the project, and stay where the person was.

        Everything that changes anything comes through here, so that the two
        lists, the counts and the detail panel can never be describing
        different states of the same folder. The position is restored by
        identifier: the lists are rebuilt from scratch, and a row number would
        by now be pointing at whatever word had moved into that place.
        """
        wanted_group = word_key or self._selected_group_key
        wanted_occurrence = occurrence_id or self._selected_occurrence_id
        previous_row = self._groups.selected_row()

        uncertainties = self._visible_uncertainties()
        settings = self._state.settings
        rows = build_group_rows(
            self._state,
            uncertainties,
            settings.show_reviewed,
            settings.show_other_uncertainties,
        )
        # How much is being kept back is worked out by building the list again
        # as though nothing were hidden and taking the difference. It is a few
        # microseconds and it cannot fall out of step with the rule that does
        # the hiding, which counting the reviewed items separately would.
        if settings.show_reviewed:
            self._hidden_reviewed = 0
        else:
            self._hidden_reviewed = len(
                build_group_rows(
                    self._state, uncertainties, True, settings.show_other_uncertainties
                )
            ) - len(rows)

        self._quiet = True
        try:
            self._group_model.set_rows(rows)
            target = self._group_model.row_for_key(wanted_group)
            if target < 0 and self._group_model.rowCount():
                # The word has left the list, so the row it occupied now holds
                # the next one along, which is where the work carries on.
                target = min(max(previous_row, 0), self._group_model.rowCount() - 1)
            if target >= 0:
                self._groups.select_row(target)
                self._show_group(self._group_model.row_at(target), wanted_occurrence)
            else:
                self._selected_group_key = None
                self._show_group(None, None)
        finally:
            self._quiet = False
        self._update_counts()

    def _update_counts(self) -> None:
        """Say how much of the folder is on show, on the screen and to a reader."""
        rows = self._group_model.rows()
        # What could show if no filter were hiding anything, which is what the
        # sentence measures the visible ones against. A word that is in the
        # list already as a weak word is not counted here either, or the count
        # would say something was missing that is in fact on the screen.
        total = len(self._uncertainty_block()) + (
            len([pair for pair in self._settled if not self._is_a_word(pair)])
            if self._state.settings.show_reviewed
            else 0
        )
        text = group_count_text(
            low_confidence_count(rows),
            len(rows) - low_confidence_count(rows),
            total,
            self._state.settings.show_other_uncertainties,
            self._hidden_reviewed,
        )
        if self._recording_names and not self._state.processed_at and not self._scanned:
            # A folder nobody has ever analysed is the one case where an empty
            # second block genuinely means "nobody has looked" rather than
            # "there is nothing here", and it must not be read as the second.
            #
            # It used to say this whenever any recording had not been read in
            # this window, which was true then and is not any more. The flagged
            # words now come out of the project, which holds them for every
            # recording that has ever been analysed, and anything that has not
            # been analysed is read before the window is given it. So an empty
            # block on an analysed folder is a folder with nothing flagged in
            # it, and saying otherwise would send somebody looking for work
            # that does not exist.
            text = (
                f"{text} The words the transcription flagged are found when the "
                "folder is processed."
            )
        self._count_label.setText(text)
        # The count also goes into the table's description, which a screen
        # reader reads after the name when the table takes focus. Somebody
        # tabbing into the list then hears how much of it they are seeing
        # without having to go and find the label.
        note = note_for(GROUPS)
        self._groups.setAccessibleDescription(f"{text} {note.summary}")
        self._groups.setToolTip(f"{text} {note.summary}")

    def _remember_place(self) -> None:
        """Record where the person had got to, for the next time this folder opens.

        Only a real project occurrence is remembered. An item the pipeline
        flagged has no project record behind it, and its identifier would be
        cleared as damage the next time the file was loaded.
        """
        row = self.current_row()
        occurrence = self.current_occurrence()
        self._state.last_group_id = row.group_id if row is not None else None
        self._state.last_occurrence_id = (
            occurrence.id
            if occurrence is not None and self._state.occurrence(occurrence.id) is not None
            else None
        )

    def select_recording(self, recording_name: str, announce_arrival: bool = True) -> bool:
        """Move to the first occurrence belonging to one recording.

        This is what the main window calls after transcribing a file, so that
        "open the review" lands on the work that has just appeared rather than
        wherever the person happened to be last week. Finding nothing is an
        ordinary answer rather than a failure: a recording whose every word
        was confident has nothing in this window, and the person is left where
        the resume markers put them.

        ``announce_arrival`` is for the one caller that has something better to
        say. Opening a review after a transcription has several facts to give
        the person that they cannot see -- how many recordings are in the
        review, how many words a rule answered on their behalf, which files
        could not be read -- and landing here used to read the row out first
        and have that longer sentence cut across it. Passing False lands
        silently and leaves the whole announcement to the caller.

        The focus note is carried in the sentence when there is one, and is
        taken and dropped when there is not. The only caller that lands
        silently does so on a window it has just opened, where no control has
        ever held the focus and so there is never a note to lose.
        """
        for row in self._group_model.rows():
            for occurrence in row.occurrences:
                if occurrence.recording_name != recording_name:
                    continue
                position = self._group_model.row_for_key(row.key)
                self._groups.select_row(position)
                self._show_group(row, occurrence.id)
                note = self._take_focus_note()
                if announce_arrival:
                    announce(self._groups, f"{spoken_group_summary(row)}{note}")
                return True
        return False

    # -- Moving through the lists -----------------------------------------

    def _show_group(self, row: GroupRow | None, occurrence_id: str | None = None) -> None:
        """Fill the second list from a word, and land on one of its occurrences."""
        self._selected_group_key = row.key if row is not None else None
        # A new word means the next occurrence read out is the first of its
        # kind, so it gets the whole sentence rather than only what has changed
        # since the last one. This also covers every rebuild of the lists,
        # since they all come through here.
        self._previous_occurrence = None
        occurrences = row.occurrences if row is not None else ()
        replacement = row.replacement if row is not None else None
        self._occurrence_model.set_occurrences(occurrences, replacement)
        self._affected_edit.setText(self._affected_text(row))
        self._affected_edit.setCursorPosition(0)
        if not occurrences:
            self._selected_occurrence_id = None
            self._show_occurrence(None)
            return
        # The remembered occurrence first, and the top of the list where it is
        # not in this word, which is where somebody arriving at a word for the
        # first time wants to be.
        target = max(self._occurrence_model.row_for_id(occurrence_id), 0)
        # The occurrence is moving because the word moved, so it is not
        # announced: the word is about to be read out, and reading the first
        # of its occurrences as well would say two things at once about one
        # keypress. The audio still starts, because arriving at a word and
        # hearing it is the point.
        self._announce_occurrence = False
        try:
            self._occurrences.select_row(target)
        finally:
            self._announce_occurrence = True
        # A model reset sometimes lands on the row it was already on, which
        # raises no signal, so the panel is brought up to date here whether
        # the signal came or not.
        self._show_occurrence(self._occurrence_model.occurrence_at(target))

    def _affected_text(self, row: GroupRow | None) -> str:
        """What changing this word would reach, counted rather than estimated.

        A real group is described by
        :func:`~audio_transcriber.transcription.grouping.affected_summary`, so
        that the words on the screen and the words read out after the change
        cannot drift apart. A word that has no group behind it is not passed
        through that function at all, because it has no group identifier to
        pass, and it is described in the same shape here.
        """
        if row is None:
            return NO_WORD_SELECTED
        if row.group_id is not None:
            return affected_summary(self._state, row.group_id)
        return "This replacement applies to 1 occurrence in 1 file."

    def _on_group_row_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
        row = self._group_model.row_at(current.row()) if current.isValid() else None
        self._show_group(row)
        if row is not None and not self._quiet:
            announce(self._groups, spoken_group_summary(row))
            # Started again here, after the sentence has been handed over,
            # rather than being left where filling the second list happened to
            # start it. That was before the row had been described at all, so
            # the whole of the wait was spent while the reader had not yet been
            # given anything to say, and the clip and the sentence then went
            # off together.
            self._start_automatic_playback()

    def _on_occurrence_row_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
        occurrence = (
            self._occurrence_model.occurrence_at(current.row()) if current.isValid() else None
        )
        self._show_occurrence(occurrence)
        if occurrence is None or self._quiet:
            return
        if self._announce_occurrence:
            announce(
                self._occurrences,
                spoken_occurrence_summary(
                    occurrence,
                    self._occurrence_model.group_replacement,
                    self._previous_occurrence,
                ),
            )
        self._previous_occurrence = occurrence
        self._start_automatic_playback()

    def go_to_next_item(self) -> None:
        self._step_occurrence(1)

    def go_to_previous_item(self) -> None:
        self._step_occurrence(-1)

    def go_to_next_group(self) -> None:
        self._step_group(1)

    def go_to_previous_group(self) -> None:
        self._step_group(-1)

    def _step_occurrence(self, delta: int) -> None:
        """Move up or down the occurrences, saying so when there is nowhere to go.

        Stopping at the ends rather than wrapping round matters here: a list
        is worked through from one end to the other, and silently starting
        again at the top would have somebody reviewing the same words twice
        without noticing. The ends of a word are not a dead end, though, so
        they say which way out there is.
        """
        count = self._occurrence_model.rowCount()
        if count == 0:
            # About occurrences, because that is what was asked about. This
            # used to read out the count of words showing in the *other* list,
            # so somebody who pressed F3 with nothing selected asked how many
            # occurrences there were and was told how many words there were,
            # in a sentence that mentioned neither the key they pressed nor the
            # list they were in.
            row = self.current_row()
            if row is None:
                self._set_status(
                    "No word is selected, so there are no occurrences to move through. "
                    f"{self._count_label.text()}",
                    alert=True,
                )
            else:
                self._set_status(f"{row.word} has no occurrences left.", alert=True)
            return
        row = self._occurrences.selected_row()
        target = 0 if row < 0 else row + delta
        if target < 0:
            self._set_status(
                "This is the first occurrence of this word. Ctrl+Shift+F3 goes to the "
                "previous word.",
                alert=True,
            )
            return
        if target >= count:
            self._set_status(
                "This is the last occurrence of this word. Ctrl+F3 goes to the next word.",
                alert=True,
            )
            return
        self._occurrences.select_row(target)

    def _step_group(self, delta: int) -> None:
        count = self._group_model.rowCount()
        if count == 0:
            self._set_status(self._count_label.text(), alert=True)
            return
        row = self._groups.selected_row()
        target = 0 if row < 0 else row + delta
        if target < 0:
            self._set_status("This is the first word in the list.", alert=True)
            return
        if target >= count:
            self._set_status("This is the last word in the list.", alert=True)
            return
        self._groups.select_row(target)

    # -- The filters and the toggles ---------------------------------------

    def _on_reason_toggled(self, reason: ReviewReason, shown: bool) -> None:
        self._queue_model.set_reason_shown(reason, shown)
        self._filters_changed()

    def _on_confidence_toggled(self, confidence: Confidence, shown: bool) -> None:
        self._queue_model.set_confidence_shown(confidence, shown)
        self._filters_changed()

    def show_everything(self) -> None:
        """Put every filter back on, in one change and one announcement."""
        self._setting_filters = True
        try:
            for box in list(self._reason_boxes.values()) + list(self._confidence_boxes.values()):
                box.setChecked(True)
            self._queue_model.show_everything()
        finally:
            self._setting_filters = False
        self._filters_changed()

    def _filters_changed(self) -> None:
        """Say how much is showing, because an empty list says nothing.

        A filter that has quietly hidden everything looks exactly like a
        broken list. The count is on screen as well, so a sighted user and a
        screen reader user are told the same thing.
        """
        if self._setting_filters:
            return
        self.refresh()
        self._set_status(f"{self._count_label.text()}{self._take_focus_note()}", alert=True)

    def set_show_reviewed(self, shown: bool) -> None:
        """Show or hide the words already settled, and remember which.

        The focus note is carried in this sentence rather than left standing,
        and that is not a nicety. Bringing the reviewed words back, or sending
        them away, can empty the list the person is standing in, and emptying
        it takes the focus off the controls that acted on it. Without the note
        here the move would be silent, and the flag saying it had happened
        would stay set and turn up at the end of some later, unrelated
        sentence about something else entirely.
        """
        if self._setting_toggles or shown == self._state.settings.show_reviewed:
            return
        self._state.settings.show_reviewed = shown
        self._match_toggle(self._show_reviewed_box, self._show_reviewed_action, shown)
        self.refresh()
        self._save_and_say(
            f"Reviewed words are now {'shown' if shown else 'hidden'}. "
            f"{self._count_label.text()}{self._take_focus_note()}"
        )

    def set_show_other_uncertainties(self, shown: bool) -> None:
        """Show or hide the second block, and remember which."""
        if self._setting_toggles or shown == self._state.settings.show_other_uncertainties:
            return
        self._state.settings.show_other_uncertainties = shown
        self._match_toggle(self._show_uncertainties_box, self._show_uncertainties_action, shown)
        self.refresh()
        self._save_and_say(
            f"Other uncertainties are now {'shown' if shown else 'hidden'}. "
            f"{self._count_label.text()}{self._take_focus_note()}"
        )

    def set_play_automatically(self, playing: bool) -> None:
        """Turn automatic playback on or off, and remember which."""
        if self._setting_toggles or playing == self._state.settings.play_automatically:
            return
        self._state.settings.play_automatically = playing
        self._match_toggle(self._auto_play_box, self._auto_play_action, playing)
        if not playing:
            self._play_timer.stop()
            self._pending_play = None
        self._save_and_say(
            "Occurrences now play as you reach them."
            if playing
            else "Occurrences no longer play by themselves. F5 plays the one you are on."
        )

    def _match_toggle(self, box: QCheckBox, action: QAction, value: bool) -> None:
        """Keep a check box and its menu entry saying the same thing.

        Either can be the one the person used, and each sets the other, so
        both are moved with the guard up. Without it the two would answer each
        other round the loop and the setting would be saved twice.
        """
        self._setting_toggles = True
        try:
            box.setChecked(value)
            action.setChecked(value)
        finally:
            self._setting_toggles = False

    # The three numbers each answer two signals, and the split between them is
    # the point rather than an accident of how Qt names things.
    #
    # ``valueChanged`` fires on every intermediate value a spin box passes
    # through: on each of the two keystrokes of "55", and on every repeat of a
    # held arrow key. The value itself is taken here, so that a regroup asked
    # for by a keyboard shortcut -- which moves no focus, and so ends no
    # editing -- still uses the number the person can see. Nothing is said and
    # nothing is written, because sixty announcements of sixty different
    # thresholds is several minutes of speech that cannot be interrupted or
    # typed through, and sixty writes of the project file underneath it.
    #
    # ``editingFinished`` fires once, when the person leaves the box or presses
    # Enter, which is when they have actually decided. That is where the saving
    # and the saying belong. It is also raised for a box that was only visited,
    # so each of these checks whether the number really moved before saying
    # anything at all.

    def _on_minimum_changed(self, percentage: int) -> None:
        self._state.settings.minimum_confidence = percentage / 100

    def _on_minimum_settled(self) -> None:
        percentage = self._minimum_spin.value()
        if percentage == self._settled_minimum:
            return
        self._settled_minimum = percentage
        self._state.settings.minimum_confidence = percentage / 100
        self._save_and_say(
            f"The minimum confidence is now {percentage} percent. Regroup the words, "
            "with Ctrl+G, to apply it."
        )

    def _on_tolerance_changed(self, percentage: int) -> None:
        self._state.settings.grouping_tolerance = percentage / 100

    def _on_tolerance_settled(self) -> None:
        percentage = self._tolerance_spin.value()
        if percentage == self._settled_tolerance:
            return
        self._settled_tolerance = percentage
        self._state.settings.grouping_tolerance = percentage / 100
        self._save_and_say(
            f"The grouping tolerance is now {percentage} percent. Regroup the words, "
            "with Ctrl+G, to apply it."
        )

    def _on_delay_changed(self, seconds: int) -> None:
        self._state.settings.auto_play_delay_seconds = seconds

    def _on_delay_settled(self) -> None:
        """Take the new wait, and say what it will now feel like.

        Unlike the two thresholds this one takes effect at once, because it
        costs nothing to apply and because the way to choose it is to try it:
        move the highlight, hear whether the reader finishes first, adjust.
        Waiting for a regroup would make that impossible.
        """
        seconds = self._delay_spin.value()
        if seconds == self._settled_delay:
            return
        self._settled_delay = seconds
        self._state.settings.auto_play_delay_seconds = seconds
        self._save_and_say(
            NO_AUTO_PLAY_DELAY
            if seconds == 0
            else f"The audio now starts {_counted(seconds, 'second')} after you arrive "
            "on an occurrence."
        )

    # -- The analysis -------------------------------------------------------

    def process_low_confidence_words(self) -> bool:
        """Find the weak words in every recording and gather them into words."""
        return self._run_analysis("Processed")

    def regroup_words(self) -> bool:
        """Run the same analysis again, which is what a changed threshold needs.

        Processing and regrouping are the same work, and saying otherwise here
        would be a lie in the code to match a difference that only exists in
        the person's head. What differs is when each is reached for: the first
        time a folder is looked at, and after moving a threshold. So they are
        one function with two names and two sentences.
        """
        return self._run_analysis("Regrouped")

    def _run_analysis(self, verb: str) -> bool:
        """Rebuild the project's suggestions, keeping every decision in it.

        Two things happen here that are easy to miss. The analysis never edits
        a transcript, so where a rule the person accepted earlier answers a
        word found now, this window is what actually writes that correction
        into the file and saves it. And a word the analysis answered that way
        is still a job done rather than a job to do, so it is reported
        separately from the words that need somebody.
        """
        if not self._recording_names:
            self._set_status(NOTHING_TO_PROCESS, alert=True, urgent=True)
            return False
        self._remember_place()
        # Said *before* the work starts, and the order is the whole of the
        # point. This runs on the graphical thread, one transcript at a time,
        # and fifty hour-long recordings take about 37 seconds of it. During
        # that time the window answers no keys and repaints nothing, and
        # Windows eventually paints "Not Responding" into the title bar, which
        # a screen reader reads out. A person who was told nothing beforehand
        # cannot tell that from a crash, and their only move is to kill the
        # application in the middle of a run that writes transcripts.
        #
        # So they are told what is about to happen and how much of it there is,
        # and processEvents is called so that the sentence actually reaches the
        # screen and the screen reader before the thread stops answering. A
        # wait cursor goes on for anybody watching with their eyes. None of
        # this makes it faster; all of it makes the wait explicable.
        self._set_status(
            f"Looking at every word of {_counted(len(self._recording_names), 'recording')}. "
            "The window will not answer until it has finished.",
            alert=True,
        )
        QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            return self._analyse(verb)
        finally:
            QApplication.restoreOverrideCursor()

    def _analyse(self, verb: str) -> bool:
        """The analysis itself, with the window already told to expect a wait."""
        # Emptied for this run rather than read off the set of everything the
        # window has ever scanned, because the sentence below is about this
        # run. A recording read successfully an hour ago and unreadable now
        # has to be counted as unreadable now, or somebody is told their whole
        # folder was looked at when part of it was not.
        self._read_this_run = set()
        previous = self._state
        self._state = reprocess(previous, self._recording_names, self._scanning_loader)
        # The analysis builds a fresh project out of the old one and knows
        # nothing about the flagged words, which are the window's business and
        # not its own. The loader has been writing them into the old project as
        # it read each recording, so bringing that list across is what carries
        # them over: it holds fresh words for every recording this run read and
        # the previous words for every recording it could not.
        self._state.flagged = previous.flagged
        applied = self._write_rule_answers()
        self.refresh(
            word_key=None
            if self._state.last_group_id is None
            else group_key(self._state.last_group_id),
            occurrence_id=self._state.last_occurrence_id,
        )
        recordings = len(self._recording_names)
        # The words still wanting a person, which is not the same as every
        # occurrence in the project. The project holds the settled ones too:
        # the words somebody reviewed last week, and the ones a rule answered
        # during this very run. Counting all of them promised "4 words to look
        # at" above a list that had nothing in it, which reads as a broken
        # window rather than as a finished folder.
        words = len(
            [occurrence for occurrence in self._state.occurrences if not occurrence.reviewed]
        )
        groups = len(self._state.groups)
        unreadable = len(
            [name for name in self._recording_names if name not in self._read_this_run]
        )
        message = (
            f"{verb} {_counted(recordings, 'recording')}. "
            f"{_counted(words, 'word')} to look at, in {_counted(groups, 'group')}. "
            f"{self._count_label.text()}"
        )
        if unreadable:
            message = (
                f"{message} {_counted(unreadable, 'recording')} could not be read, so "
                "nothing in them was looked at."
            )
        if applied:
            # A further N, not "N of them": they are not among the words above
            # any more, because they are already settled. The verb agrees with
            # the number as well, which it did not: the count was interpolated
            # into a sentence whose "were" stayed put whatever it said.
            message = (
                f"{message} A further {_counted(applied, 'word')} "
                f"{'was' if applied == 1 else 'were'} answered automatically by "
                "replacements you accepted earlier, and needs nothing from you."
            )
        self._save_and_say(f"{message}{self._take_focus_note()}")
        return True

    def _scanning_loader(self, recording_name: str) -> Transcript | None:
        """Read a transcript for the analysis, and note its flagged words on the way.

        The analysis walks the whole folder, one recording at a time, and a
        transcript is the only place the flagged words can be found. Riding
        along with the pass the analysis is making anyway is what keeps the
        folder read once rather than twice, and it is the only moment in the
        window's life when reading every recording is justified at all. What
        this leaves behind goes into the project, so that no later opening has
        to do it again.

        Nothing is held here. The transcript goes back to the analysis, which
        lets it go when it has finished with it, and what stays behind is a
        few hundred tokens rather than tens of megabytes.
        """
        transcript = self._transcript(recording_name)
        if transcript is not None:
            self._read_this_run.add(recording_name)
            self._note_uncertainties(recording_name, transcript)
        return transcript

    def _write_rule_answers(self) -> int:
        """Write into the transcripts the corrections the analysis only recorded.

        The analysis records that a project rule answers a word and stops
        there, deliberately, because deciding and rewriting are different
        kinds of act and only one of them belongs in a function that a person
        may run fifty times while exploring a threshold. Doing the writing
        here means an automatic correction takes exactly the same road into a
        transcript as one somebody typed, including being saved at once and
        being announced if the save fails.

        Each rule's tally is raised through
        :func:`~audio_transcriber.transcription.grouping.note_rule_applied`
        rather than by adding one here, and that is the point of the function
        existing. Two places in the application write a rule's correction into
        a transcript -- this one, and ``apply_rules`` when a new file is
        transcribed -- and only the second was keeping the count. Whether a
        rule's ``occurrence_count`` was right therefore depended on which of
        the two roads a particular correction happened to take, which is not a
        difference anybody looking at the number could have known about.
        """
        applied = 0
        for recording_name, occurrences in self._by_recording(
            [
                occurrence
                for occurrence in self._state.occurrences
                if occurrence.auto_applied and occurrence.replacement
            ]
        ):
            transcript = self._transcript(recording_name)
            if transcript is None:
                continue
            changed = False
            for occurrence in occurrences:
                token = transcript.token_by_id(occurrence.token_id)
                if token is None or token.text == occurrence.replacement:
                    continue
                transcript = transcript.with_correction(
                    token.id, text=occurrence.replacement
                )
                changed = True
                applied += 1
                if occurrence.applied_rule_id is not None:
                    note_rule_applied(self._state, occurrence.applied_rule_id)
            if changed:
                self._hand_on(recording_name, transcript)
        return applied

    @staticmethod
    def _by_recording(
        occurrences: list[Occurrence],
    ) -> list[tuple[str, list[Occurrence]]]:
        """Gather occurrences by the file they are in, keeping the project order.

        Everything that changes a word across several files goes through this
        first, so that one transcript is read, corrected, handed on and let go
        of before the next is touched. Walking the occurrences in their own
        order instead would hold every transcript the word appears in at once,
        which for a surname said in fifty recordings is the whole folder.
        """
        gathered: dict[str, list[Occurrence]] = {}
        for occurrence in occurrences:
            gathered.setdefault(occurrence.recording_name, []).append(occurrence)
        return list(gathered.items())

    # -- The item on show --------------------------------------------------

    def _show_occurrence(self, occurrence: Occurrence | None) -> None:
        """Fill in every part of the detail panel for one occurrence, or clear it."""
        self._selected_occurrence_id = occurrence.id if occurrence is not None else None
        transcript = (
            self._transcript(occurrence.recording_name) if occurrence is not None else None
        )
        token = (
            transcript.token_by_id(occurrence.token_id)
            if occurrence is not None and transcript is not None
            else None
        )

        self._candidates_model.set_rows(
            candidate_rows(transcript, token) if transcript is not None else []
        )
        if self._candidates_model.rowCount():
            self._candidates.setCurrentIndex(
                self._candidates_model.index(0, CANDIDATE_COLUMN_SERVICE)
            )
        else:
            self._set_value(self._candidate_text, "")

        if occurrence is None:
            for edit in (
                self._detected_edit,
                self._strength_edit,
                self._file_edit,
                self._position_edit,
                self._word_language_edit,
                self._timing_edit,
                self._timing_status_edit,
                self._timing_source_edit,
                self._timing_confidence_edit,
                self._alignment_edit,
                self._speaker_edit,
                self._speaker_source_edit,
                self._speaker_confidence_edit,
                self._language_edit,
                self._language_evidence_edit,
                self._risk_edit,
                self._reasons_edit,
                self._span_edit,
            ):
                self._set_value(edit, NO_ITEM_SELECTED)
            self._decision_text.setPlainText(NO_LLM_DECISION)
            self._replacement_edit.setText("")
            self._fill_speaker_box(None)
            self._set_item_controls_enabled(False)
            return

        self._set_value(self._detected_edit, occurrence.detected_text)
        self._set_value(
            self._strength_edit, strength_display(occurrence.confidence_strength)
        )
        self._set_value(self._file_edit, occurrence.recording_name)
        self._set_value(self._position_edit, time_display(occurrence.start))
        self._set_value(self._word_language_edit, language_name(occurrence.language))

        if token is None:
            # Two different things bring us here and they are not the same
            # news. The transcript could not be read at all just now, which is
            # a passing fault and says nothing about the word; or it was read
            # and the word is not in it, which means the recording was
            # transcribed again and the word could not be found. Everything
            # about the occurrence itself is still shown either way, because
            # it is what the person decided about.
            trouble = (
                TRANSCRIPT_NOT_READABLE.format(name=occurrence.recording_name)
                if transcript is None
                else WORD_NOT_IN_TRANSCRIPT
            )
            for edit in (
                self._timing_edit,
                self._timing_status_edit,
                self._timing_source_edit,
                self._timing_confidence_edit,
                self._alignment_edit,
                self._speaker_edit,
                self._speaker_source_edit,
                self._speaker_confidence_edit,
                self._language_edit,
                self._language_evidence_edit,
                self._risk_edit,
                self._reasons_edit,
                self._span_edit,
            ):
                self._set_value(edit, trouble)
            self._decision_text.setPlainText(NO_LLM_DECISION)
            self._replacement_edit.setText(self._replacement_for(occurrence))
            self._fill_speaker_box(None)
            self._set_item_controls_enabled(False)
            return

        self._set_value(self._timing_edit, timing_text(token))
        self._set_value(self._timing_status_edit, _TIMING_STATUS_TEXT[token.timing_status])
        self._set_value(self._timing_source_edit, provider_name(token.timing_source))
        self._set_value(self._timing_confidence_edit, token.timing_confidence.display_name)
        self._set_value(self._alignment_edit, _ALIGNMENT_STATUS_TEXT[token.alignment_status])

        self._set_value(self._speaker_edit, speaker_text(transcript, token))
        self._set_value(self._speaker_source_edit, provider_name(token.speaker_source))
        self._set_value(self._speaker_confidence_edit, token.speaker_confidence.display_name)

        self._set_value(self._language_edit, token.language.display_name)
        self._set_value(self._language_evidence_edit, language_evidence_text(token))
        self._set_value(self._risk_edit, risk_text(token.risk_categories))
        self._set_value(self._reasons_edit, reason_text(token))
        self._decision_text.setPlainText(token.llm_decision or NO_LLM_DECISION)

        self._set_value(self._span_edit, self._span_description())
        self._replacement_edit.setText(self._replacement_for(occurrence))
        self._replacement_edit.setCursorPosition(0)
        self._fill_speaker_box(transcript)
        self._speaker_box.setCurrentText(self._speaker_box_text(transcript, token))
        self._set_item_controls_enabled(True)

    def _replacement_for(self, occurrence: Occurrence) -> str:
        """What the replacement box should hold for an occurrence.

        Its own replacement where it has one, its word's where the word has
        one, and otherwise what the services detected, so that F2 selects
        something worth editing rather than an empty box. The detected form is
        offered rather than the current text, because a replacement is always
        worked out from what a word originally said.
        """
        return (
            occurrence.replacement
            or self._occurrence_model.group_replacement
            or occurrence.detected_text
        )

    def _fill_speaker_box(self, transcript: Transcript | None) -> None:
        """Offer the speakers of this recording, keeping what was typed.

        The item text is what a person calls the speaker and the item data is
        the label the services used, because a correction has to record the
        label rather than the friendly name. The list is rebuilt on every move
        now, rather than once when the window opened, because moving between
        occurrences moves between recordings and each has speakers of its own.
        """
        typed = self._speaker_box.currentText()
        self._speaker_box.clear()
        if transcript is not None:
            for speaker in transcript.speakers:
                self._speaker_box.addItem(speaker.display_name, speaker.id)
        self._speaker_box.setCurrentText(typed)

    def _speaker_box_text(self, transcript: Transcript, token: FinalToken) -> str:
        """What to show in the speaker box for a word.

        A speaker the recording knows about is shown by the name a person
        gave them; anything else is shown by its raw label. Both come back
        as the right label when the correction is applied, which is what
        stops the friendly name from being saved as though it were one.
        """
        if not token.speaker:
            return ""
        speaker = transcript.speaker_for(token.speaker)
        return speaker.display_name if speaker is not None else token.speaker

    def _span_description(self) -> str:
        """What pressing Play would actually play, said in full."""
        span = self.span_to_play()
        if self._audio_path() is None:
            return NO_CANONICAL_AUDIO
        if span is None:
            return NO_AUDIO
        return f"From {spoken_duration(span.start)}, {_seconds_phrase(span.duration)} in all"

    def _set_item_controls_enabled(self, enabled: bool) -> None:
        """Switch the controls that act on an item on or off.

        The focus is moved off them first, deliberately. A control that is
        disabled while it holds the focus takes the focus away with it, and
        the user is left nowhere with nothing said about it.
        """
        playable = enabled and self.span_to_play() is not None
        row = self.current_row()
        # Isolating means taking a word out of a group, so it needs a group
        # with something to take it out of. Offering it on a word that has one
        # occurrence would be offering to move a word from where it is to
        # where it already is.
        can_isolate = bool(
            enabled and row is not None and row.group_id is not None and len(row.occurrences) > 1
        )
        # The Occurrences table is guarded here too, and it was the one control
        # on the screen that was not. It is switched off exactly when it has no
        # rows, which is not the same test as ``enabled``: a word whose
        # transcript could not be read switches every correction off while the
        # table itself is still full of rows worth moving through.
        #
        # Being in the list mattered because of what happens at the end of a
        # word. Confirming its last occurrence empties the table, and somebody
        # whose focus was in it was then standing in a table with no rows: the
        # arrow keys did nothing, the reader said nothing because there was
        # nothing to read, and the only way out was to guess that Tab would
        # work. Being in this list means the focus is caught and moved to the
        # word list, and the sentence about what just happened says so.
        has_occurrences = self._occurrence_model.rowCount() > 0
        widgets: list[tuple[QWidget, bool]] = [
            (self._occurrences, has_occurrences),
            (self._play_button, playable),
            (self._play_wide_button, playable),
            (self._use_candidate_button, enabled),
            (self._replacement_edit, enabled),
            (self._speaker_box, enabled),
            (self._apply_word_button, enabled),
            (self._apply_occurrence_button, enabled),
            (self._correct_as_detected_button, enabled),
            (self._isolate_button, can_isolate),
            (self._apply_speaker_button, enabled),
            (self._confirm_timing_button, enabled),
            (self._reject_timing_button, enabled),
            (self._confirm_button, enabled),
        ]
        # The menu commands are switched off alongside the buttons they
        # duplicate. A greyed-out button beside a live menu entry that then
        # does nothing is worse than either on its own.
        actions: list[tuple[QAction, bool]] = [
            (self._play_action, playable),
            (self._play_wide_action, playable),
            (self._apply_word_action, enabled),
            (self._apply_occurrence_action, enabled),
            (self._correct_as_detected_action, enabled),
            (self._isolate_action, can_isolate),
            (self._apply_speaker_action, enabled),
            (self._confirm_timing_action, enabled),
            (self._reject_timing_action, enabled),
            (self._confirm_action, enabled),
            (self._correct_text_action, enabled),
        ]
        losing_focus = any(
            not wanted and self._holds_focus(widget) for widget, wanted in widgets
        )
        if losing_focus:
            self._focus_groups()
            # Not announced here. Something more useful is always about to
            # be said, and two announcements in a row means the second one
            # arrives while the user is still listening to the first.
            self._focus_caught = True
        for widget, wanted in widgets:
            widget.setEnabled(wanted)
        for action, wanted in actions:
            action.setEnabled(wanted)

    def _holds_focus(self, widget: QWidget) -> bool:
        # The window is asked which of its widgets has the focus rather than
        # the widget itself, because hasFocus is false whenever the window
        # is not the active one on the desktop.
        focused = self.focusWidget()
        return focused is not None and (focused is widget or widget.isAncestorOf(focused))

    @staticmethod
    def _set_value(edit: QLineEdit, text: str) -> None:
        """Put a value in a field without dragging a reader away from their place.

        The caret is put back where it was when the field has the focus,
        because replacing the text moves it to the end on its own, and
        somebody reading the field a word at a time, or following the caret
        with ZoomText, would lose their place.
        """
        if edit.text() == text:
            return
        if edit.window().focusWidget() is edit:
            caret = edit.cursorPosition()
            edit.setText(text)
            edit.setCursorPosition(min(caret, len(text)))
        else:
            edit.setText(text)
            edit.setCursorPosition(0)

    def _on_candidate_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
        row = self._candidates_model.row_at(current.row()) if current.isValid() else None
        self._set_value(self._candidate_text, row.text if row is not None else "")

    def use_selected_candidate(self) -> None:
        """Copy the highlighted candidate into the replacement box."""
        text = self._candidate_text.text()
        if not text:
            self._set_status("No candidate is highlighted.", alert=True, urgent=True)
            return
        self._replacement_edit.setText(text)
        self._replacement_edit.setCursorPosition(0)
        self._set_status(
            f"{text} put in the replacement box. Apply it to change the word.", alert=True
        )

    # -- Playback ----------------------------------------------------------

    def span_to_play(self, wide: bool = False) -> AudioSpan | None:
        """The region that playing would cover, padding included.

        The word's own span is used where it has one, and the wider
        containing span where it does not, which is exactly when a person
        most needs to hear it. The padding is clamped to the recording, so
        a word in the first seconds does not ask for audio from before the
        beginning.
        """
        token = self.current_token()
        if token is None:
            return None
        span = token.audible_span
        if span is None:
            return None
        padding = WIDE_CONTEXT_SECONDS if wide else CONTEXT_SECONDS
        transcript = self.current_transcript
        audio = transcript.canonical_audio if transcript is not None else None
        return span.padded(padding, padding, limit=audio.duration if audio else None)

    def _audio_path(self) -> Path | None:
        occurrence = self.current_occurrence()
        if occurrence is None:
            return None
        return self._audio_path_for(occurrence.recording_name)

    def _audio_path_for(self, recording_name: str) -> Path | None:
        """The file this recording's timestamps actually refer to.

        The canonical audio comes first, because every time in the transcript
        is measured against it, and a converted copy may have been trimmed or
        resampled in a way that would shift every number if the original were
        played instead. The original is the fallback rather than the first
        choice, for when the copy is a working file that has since been
        cleared away.
        """
        transcript = self._transcript(recording_name)
        audio = transcript.canonical_audio if transcript is not None else None
        if audio is not None and audio.path:
            return Path(audio.path)
        return self._recording_paths.get(recording_name)

    def _start_automatic_playback(self) -> None:
        """Line up the audio for the occurrence just landed on.

        The wait is what stops eight clips queueing when somebody holds the
        Down arrow: each move restarts a single-shot timer, so only the
        occurrence they stop on is ever played. Nothing here touches the
        focus, because the person is working in a list and being moved out of
        it by the sound arriving would be worse than no sound at all.

        How long the wait is belongs to the person and is saved with the rest
        of this folder's review. It used to be 400 milliseconds, fixed, which
        solved only the crude half of the problem. Holding the Down arrow no
        longer queued eight clips, but landing on a single occurrence still
        started the audio 400 milliseconds after handing the screen reader a
        sentence that takes it three to eight seconds to read. The person heard
        the file name, then two voices at once, and never learned the
        confidence, the language or the replacement -- the three facts they
        were about to make a decision on. Nobody but the person knows how fast
        their own reader speaks, so nobody but the person can set this.

        A word with no audio behind it is passed over in silence rather than
        being complained about, since a complaint on every arrow press through
        a run of untimed words would be unusable.
        """
        if not self._state.settings.play_automatically:
            return
        if self.span_to_play() is None:
            return
        self._play_timer.start(self._state.settings.auto_play_delay_seconds * 1000)

    def _play_after_waiting(self) -> None:
        self.play_span(False, automatic=True)

    def play_span(self, wide: bool = False, automatic: bool = False) -> bool:
        """Play the audio around the selected word. Returns whether it started.

        Returning True does not mean a sound has come out of the speakers
        yet. The recording changes as the person moves between occurrences,
        and Qt opens a file in the background, so a seek issued immediately
        after ``load`` is dropped on the floor without a word: the player has
        no media to seek in yet. What to play is therefore remembered and
        carried out when the media reports its length, which is Qt's way of
        saying it is ready. The current window never met this because it
        opened one file and never changed it.
        """
        path = self._audio_path()
        if path is None:
            self._set_status(NO_CANONICAL_AUDIO, alert=True, urgent=True)
            return False
        span = self.span_to_play(wide)
        if span is None:
            self._set_status(NO_AUDIO, alert=True, urgent=True)
            return False
        start = int(span.start * 1000)
        self._stop_at_ms = int(span.end * 1000)
        description = f"{_seconds_phrase(span.duration)} from {spoken_duration(span.start)}"

        if path != self._loaded_path:
            self._loaded_path = path
            self._media_ready = False
            self._pending_play = (start, self._stop_at_ms, description)
            self._player.load(path)
            self._set_status(f"Loading {path.name}.", alert=not automatic)
            return True
        if not self._media_ready:
            self._pending_play = (start, self._stop_at_ms, description)
            return True
        self._pending_play = None
        self._player.seek_to(start)
        self._player.play()
        self._set_status(f"Playing {description}.", alert=not automatic)
        return True

    def _on_duration_changed(self, milliseconds: int) -> None:
        """Start what was waiting, now that the file is open.

        A length of nought is Qt saying it does not know yet, which it says
        once for every file as the source is set. Only a real length means the
        media can be seeked in.
        """
        if milliseconds <= 0:
            return
        self._media_ready = True
        pending = self._pending_play
        if pending is None:
            return
        self._pending_play = None
        start, stop_at, description = pending
        self._stop_at_ms = stop_at
        self._player.seek_to(start)
        self._player.play()
        # Not announced. Whoever asked for this has already been told the file
        # was being loaded, and saying so again when it arrives would talk
        # over the audio it is announcing.
        self._status_label.setText(f"Playing {description}.")

    def _on_player_error(self, message: str) -> None:
        """Say that the audio failed, rather than leaving a silence to be read.

        The loaded path is forgotten as well as the pending clip, so that
        asking again reloads the file instead of waiting forever for a
        readiness that will not now arrive.
        """
        self._pending_play = None
        self._media_ready = False
        self._loaded_path = None
        self._set_status(message, alert=True, urgent=True)

    def _on_position_changed(self, milliseconds: int) -> None:
        """Stop at the end of the region, since Qt only knows about whole files."""
        if self._stop_at_ms is None:
            return
        if milliseconds >= self._stop_at_ms:
            self._stop_at_ms = None
            self._player.pause()

    # -- Corrections -------------------------------------------------------

    def apply_replacement_to_word(self) -> bool:
        """Replace every occurrence of the selected word, in every recording.

        Each occurrence is rewritten from what it originally said rather than
        from what it currently says, which is what makes correcting a
        replacement give the same answer as having typed it correctly the
        first time. The occurrence keeps the form the services produced for
        the life of the project and it is never overwritten here.

        Nothing about the speaker or the timing of any of them is touched.
        Every message says so, because a change reaching files the person has
        not opened has to be explicit about what it did not do as well as
        about what it did.
        """
        row = self.current_row()
        if row is None:
            self._set_status(NO_WORD_SELECTED, alert=True, urgent=True)
            return False
        text = self._replacement_edit.text().strip()
        if not text:
            self._set_status(
                "The replacement cannot be empty. Type what was actually said.",
                alert=True,
                urgent=True,
            )
            return False

        group = self._state.group(row.group_id) if row.group_id is not None else None
        touched: list[Occurrence] = []
        missing = 0
        anything_changed = False
        # One recording at a time, corrected, handed on and let go of before
        # the next is opened. A word said in fifty files would otherwise hold
        # fifty transcripts at once, which is the whole folder.
        for recording_name, occurrences in self._by_recording(list(row.occurrences)):
            transcript = self._transcript(recording_name)
            if transcript is None:
                missing += len(occurrences)
                continue
            changed = False
            for occurrence in occurrences:
                # An occurrence with a replacement of its own has been decided
                # separately and keeps its own answer, which is the whole
                # point of having asked for this occurrence only.
                if occurrence.replacement and group is not None:
                    continue
                token = transcript.token_by_id(occurrence.token_id)
                if token is None:
                    missing += 1
                    continue
                touched.append(occurrence)
                if token.text == text:
                    continue
                transcript = transcript.with_correction(token.id, text=text)
                changed = True
            if changed:
                anything_changed = True
                self._hand_on(recording_name, transcript)

        # Nothing has been written at this point unless something actually
        # changed, so refusing here refuses a change that was never made. A
        # word that already reads the way the box says is refused rather than
        # quietly recording a decision, because the person almost certainly
        # meant to settle it rather than to replace it with itself, and
        # Correct as detected is what does that.
        if not anything_changed:
            self._set_status(
                "The replacement is unchanged. Use Correct as detected to settle this "
                "word as it stands.",
                alert=True,
                urgent=True,
            )
            return False

        for occurrence in touched:
            self._mark_replaced(occurrence, None if group is not None else text)
        if group is not None:
            group.replacement = text
            group.reviewed = True
            group.correct_as_detected = False
            self._update_rules(group, text)
        else:
            self._update_loose_rules(touched, text)

        message = (
            f"{text}. {self._affected_text(row)} The speaker and the timing of every "
            "one of them are unchanged."
        )
        if missing:
            message = (
                f"{message} {_counted(missing, 'occurrence')} could not be reached in "
                "the transcripts and were left alone."
            )
        self._after_change(message)
        return True

    def apply_replacement_to_occurrence(self) -> bool:
        """Replace this one occurrence, leaving the rest of the word alone."""
        occurrence = self.current_occurrence()
        token = self.current_token()
        if occurrence is None:
            self._set_status(NO_WORD_SELECTED, alert=True, urgent=True)
            return False
        if token is None:
            self._set_status(WORD_NOT_IN_TRANSCRIPT, alert=True, urgent=True)
            return False
        text = self._replacement_edit.text().strip()
        if not text:
            self._set_status(
                "The replacement cannot be empty. Type what was actually said.",
                alert=True,
                urgent=True,
            )
            return False
        if text == token.text and occurrence.replacement == text:
            self._set_status(
                "The replacement is unchanged. Use Confirm this occurrence as correct "
                "to settle it as it stands.",
                alert=True,
                urgent=True,
            )
            return False
        transcript = self._transcript(occurrence.recording_name)
        self._hand_on(
            occurrence.recording_name, transcript.with_correction(token.id, text=text)
        )
        self._mark_replaced(occurrence, text)
        self._after_change(
            f"{text} applied to this occurrence only, in {occurrence.recording_name}. "
            "The other occurrences of this word, and the speaker and the timing of this "
            "one, are unchanged."
        )
        return True

    def _mark_replaced(self, occurrence: Occurrence, replacement: str | None) -> None:
        """Record on an occurrence that a person has now decided about it.

        ``auto_applied`` is cleared, along with the rule it named. A word a
        rule answered on the person's behalf and a word they looked at
        themselves are different facts, and once they have looked at it the
        second is the true one.

        A pseudo-occurrence standing in for a word the pipeline flagged has no
        record in the project, so there is nothing to mark: the transcript
        already carries the correction, and it is where the reviewed state of
        such a word is read back from.
        """
        if self._state.occurrence(occurrence.id) is None:
            return
        occurrence.reviewed = True
        occurrence.correct_as_detected = False
        occurrence.auto_applied = False
        occurrence.applied_rule_id = None
        if replacement is not None:
            occurrence.replacement = replacement

    def _update_rules(self, group: WordGroup, replacement: str) -> None:
        """Teach the project what to do with each detected form of this word.

        One rule per distinct form the services produced, never one per group.
        A group holding ``Bosch``, ``Bosh`` and ``Bosche`` makes three rules,
        so that a file transcribed next week saying ``Bosh`` is answered by
        somebody who actually accepted a correction for ``Bosh``. One rule for
        the group would answer only the spelling that happened to be its name.

        Rules made by this group are updated rather than added to when the
        person changes their mind, and one whose form is no longer in the
        group is dropped. Left behind, it would go on applying a replacement
        that has been withdrawn, in files nobody has opened yet.
        """
        wanted: dict[tuple[str, str], str] = {}
        for occurrence in self._state.occurrences_of(group.id):
            form = occurrence.normalised_text or occurrence.detected_text.casefold()
            wanted.setdefault((form, occurrence.language), occurrence.detected_text)

        kept: list[ReplacementRule] = []
        for rule in self._state.rules:
            if rule.group_id != group.id:
                kept.append(rule)
                continue
            key = (rule.normalised_text, rule.language)
            if key in wanted:
                rule.replacement = replacement
                rule.matched_text = wanted.pop(key)
                kept.append(rule)
        for (form, language), detected in wanted.items():
            kept.append(
                ReplacementRule(
                    id=uuid.uuid4().hex,
                    matched_text=detected,
                    normalised_text=form,
                    replacement=replacement,
                    language=language,
                    created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                    group_id=group.id,
                )
            )
        self._state.rules = kept

    def _update_loose_rules(self, occurrences: list[Occurrence], replacement: str) -> None:
        """Teach the project from a correction made on a row that has no group.

        Three kinds of row have no group behind them: a word the pipeline
        flagged, an occurrence somebody deliberately kept on its own, and one
        that went stale. None of them ever taught the project anything, because
        the only place rules were made from was
        :meth:`_update_rules`, and that takes a group.

        The second block is where this mattered most, and the reason is worth
        stating. A word is flagged for provider disagreement, or for being a
        risky value, without any measured confidence of its own, so the
        low-confidence sweep never sees it and it never gets a group. Those are
        exactly the proper names and the amounts the services get *confidently*
        wrong. So the whole class of word the tenth decision was written for --
        every replacement the person accepts is applied to the next
        transcription -- was the one class that produced no rules at all.

        A rule made here answers the same detected form the person just
        corrected, and any earlier rule for that form is replaced rather than
        left beside it, whichever group it came from. Two rules answering one
        form would be decided by whichever the matching happened to reach
        first, which is not a thing anybody could reason about; and the person
        has this moment said what that form should read, which is later
        evidence than whatever said otherwise.
        """
        for occurrence in occurrences:
            form = occurrence.normalised_text or occurrence.detected_text.casefold()
            if not form:
                continue
            self._state.rules = [
                rule
                for rule in self._state.rules
                if not (rule.normalised_text == form and rule.language == occurrence.language)
            ]
            self._state.rules.append(
                ReplacementRule(
                    id=uuid.uuid4().hex,
                    matched_text=occurrence.detected_text,
                    normalised_text=form,
                    replacement=replacement,
                    language=occurrence.language,
                    created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                    group_id=None,
                )
            )

    def _drop_loose_rules(self, occurrences: tuple[Occurrence, ...]) -> None:
        """Forget what a row with no group taught, because it has been reversed.

        The mirror of :meth:`_update_loose_rules`, and needed for the same
        reason the group case already drops its own rules: a rule left standing
        would go on rewriting that form in files transcribed next week, under a
        decision the person has just withdrawn, and nothing would ever tell
        them.
        """
        forms = {
            (occurrence.normalised_text or occurrence.detected_text.casefold(),
             occurrence.language)
            for occurrence in occurrences
        }
        self._state.rules = [
            rule
            for rule in self._state.rules
            if rule.group_id is not None
            or (rule.normalised_text, rule.language) not in forms
        ]

    def correct_word_as_detected(self) -> bool:
        """Say the whole word is right as the services produced it, and settle it.

        Where a replacement had already been applied, the words go back to
        what they originally said and the rules that decision produced are
        dropped. A rule left standing would go on rewriting the same form in
        files transcribed later, under a decision the person has just
        reversed, and they would have no way of knowing.
        """
        row = self.current_row()
        if row is None:
            self._set_status(NO_WORD_SELECTED, alert=True, urgent=True)
            return False
        group = self._state.group(row.group_id) if row.group_id is not None else None

        missing = 0
        for recording_name, occurrences in self._by_recording(list(row.occurrences)):
            transcript = self._transcript(recording_name)
            if transcript is None:
                missing += len(occurrences)
                continue
            changed = False
            for occurrence in occurrences:
                token = transcript.token_by_id(occurrence.token_id)
                if token is None:
                    missing += 1
                    continue
                if occurrence.detected_text and token.text != occurrence.detected_text:
                    transcript = transcript.with_correction(
                        token.id, text=occurrence.detected_text
                    )
                transcript = with_confirmation(transcript, token.id)
                changed = True
                if self._state.occurrence(occurrence.id) is not None:
                    occurrence.reviewed = True
                    occurrence.correct_as_detected = True
                    occurrence.replacement = None
                    occurrence.auto_applied = False
                    occurrence.applied_rule_id = None
            if changed:
                self._hand_on(recording_name, transcript)

        if group is not None:
            group.reviewed = True
            group.correct_as_detected = True
            group.replacement = None
            self._state.rules = [
                rule for rule in self._state.rules if rule.group_id != group.id
            ]
        else:
            self._drop_loose_rules(row.occurrences)

        message = f"{row.word} confirmed as correct as detected. {self._affected_text(row)}"
        if missing:
            message = (
                f"{message} {_counted(missing, 'occurrence')} could not be reached in "
                "the transcripts and were left alone."
            )
        self._after_change(message)
        return True

    def isolate_occurrence(self) -> bool:
        """Take this occurrence out of its word, to be decided on its own."""
        row = self.current_row()
        occurrence = self.current_occurrence()
        if row is None or occurrence is None:
            self._set_status(NO_WORD_SELECTED, alert=True, urgent=True)
            return False
        group = self._state.group(row.group_id) if row.group_id is not None else None
        if group is None or self._state.occurrence(occurrence.id) is None:
            self._set_status(
                "This occurrence is already on its own, so there is nothing to take it "
                "out of.",
                alert=True,
                urgent=True,
            )
            return False
        group.occurrence_ids = [
            item for item in group.occurrence_ids if item != occurrence.id
        ]
        occurrence.isolated = True
        word = group.representative_text
        self._selected_group_key = loose_key(occurrence.id)
        self._after_change(
            f"{occurrence.detected_text} kept on its own, and no longer part of {word}. "
            f"{affected_summary(self._state, group.id)}"
        )
        return True

    def apply_speaker_correction(self) -> bool:
        """Attribute this one occurrence to the chosen speaker, and nothing else.

        Deliberately one occurrence at a time, whatever the word does. Two
        mentions of the same surname are as likely to be said by two different
        people as by one, so a group-wide speaker would be a guess dressed up
        as a decision.
        """
        occurrence = self.current_occurrence()
        token = self.current_token()
        if occurrence is None:
            self._set_status(NO_WORD_SELECTED, alert=True, urgent=True)
            return False
        if token is None:
            self._set_status(WORD_NOT_IN_TRANSCRIPT, alert=True, urgent=True)
            return False
        speaker = self.chosen_speaker()
        if not speaker:
            self._set_status(
                "No speaker has been chosen. Pick one from the list or type a label.",
                alert=True,
                urgent=True,
            )
            return False
        if speaker == token.speaker:
            self._set_status("The speaker is unchanged.", alert=True, urgent=True)
            return False
        transcript = self._transcript(occurrence.recording_name)
        self._hand_on(
            occurrence.recording_name,
            transcript.with_correction(token.id, speaker=speaker),
        )
        self._after_change(
            f"Speaker set to {speaker} for this occurrence. The text and the timing are "
            "unchanged, and no other occurrence of this word is affected."
        )
        return True

    def chosen_speaker(self) -> str:
        """The speaker label the correction should record.

        A speaker picked from the list records the label the services used
        rather than the friendly name, because that is what the rest of the
        transcript is keyed on. Anything typed that is not in the list is
        taken as a label in its own right.
        """
        text = self._speaker_box.currentText().strip()
        if not text:
            return ""
        index = self._speaker_box.findText(text)
        if index >= 0:
            data = self._speaker_box.itemData(index)
            if data:
                return str(data)
        return text

    def decide_timing(self, accepted: bool) -> bool:
        """Confirm or reject the span, leaving the text and the speaker alone."""
        occurrence = self.current_occurrence()
        token = self.current_token()
        if occurrence is None:
            self._set_status(NO_WORD_SELECTED, alert=True, urgent=True)
            return False
        if token is None:
            self._set_status(WORD_NOT_IN_TRANSCRIPT, alert=True, urgent=True)
            return False
        transcript = self._transcript(occurrence.recording_name)
        self._hand_on(
            occurrence.recording_name,
            with_timing_decision(transcript, token.id, accepted),
        )
        self._after_change(
            "Timing confirmed. The text and the speaker are unchanged."
            if accepted
            else "Timing rejected and marked uncertain. The text and the speaker are "
            "unchanged, and the item stays in the list."
        )
        return True

    def confirm_item(self) -> bool:
        """Settle this one occurrence exactly as it stands, which is the common case.

        Unlike correcting the whole word as detected, this changes no text at
        all, not even back. It says that the word as it now reads is right
        here, which is a different statement from saying the services had it
        right, and a person who has already replaced a word and then confirms
        one occurrence of it means the first.
        """
        occurrence = self.current_occurrence()
        token = self.current_token()
        if occurrence is None:
            self._set_status(NO_WORD_SELECTED, alert=True, urgent=True)
            return False
        if token is None:
            self._set_status(WORD_NOT_IN_TRANSCRIPT, alert=True, urgent=True)
            return False
        word = chosen_text(token)
        transcript = self._transcript(occurrence.recording_name)
        self._hand_on(occurrence.recording_name, with_confirmation(transcript, token.id))
        if self._state.occurrence(occurrence.id) is not None:
            occurrence.reviewed = True
            occurrence.correct_as_detected = True
        self._after_change(f"{word} confirmed as correct in {occurrence.recording_name}.")
        return True

    # -- Handing changes on, and saying what happened -----------------------

    def _hand_on(self, recording_name: str, transcript: Transcript) -> None:
        """Keep a corrected transcript, and tell whoever is storing it.

        There is no Save button on this screen and there is not going to be
        one. A correction is written through at the moment it is made, so that
        closing the window, losing power or a crash can never take back
        something the person watched happen.
        """
        self._cached_name = recording_name
        self._cached_transcript = transcript
        # The flagged words of this recording have changed: a confirmed word
        # leaves the block and a corrected one joins the settled list. Only
        # this recording is looked at again, because the rest of the folder is
        # not in memory and has not changed. This used to be done only for a
        # recording the window had itself read, because until the project
        # started holding the flagged words there was nothing to bring up to
        # date for any other. Now there is, and skipping it would leave the
        # project describing the word as it was before the person changed it.
        # It costs nothing: the corrected transcript is the argument.
        self._note_uncertainties(recording_name, transcript)
        if self._save_correction is not None:
            self._save_correction(recording_name, transcript)
        self.transcriptChanged.emit(recording_name, transcript)

    def _save_project(self, quiet: bool = False) -> bool:
        """Write the project, and say plainly when that did not work.

        A failed save is the one thing here that must never pass quietly. The
        person believes their decisions are kept, and unless they are told now
        they will find out only after closing the window that still held them.
        """
        if self._project_store.save(self._state):
            return True
        if not quiet:
            self._set_status(self._save_failure_text(), alert=True, urgent=True)
        return False

    def _save_failure_text(self) -> str:
        return (
            f"Your review could not be saved to {self._project_store.path}. The change "
            "is only in this window until that is put right."
        )

    def _save_and_say(self, message: str) -> str:
        """Save the project and say one sentence covering both. Returns what was said.

        Every path that changes the project has to do this, and the reason is
        that :meth:`_set_status` both writes the label and announces it, so two
        calls in a row do not give a person two messages: the second wipes the
        first off the screen and, on a polite announcement, cuts across it
        being read. Saving loudly and then saying what happened therefore
        destroys the more urgent of the two messages every time. Half the
        window used to do exactly that.

        The damaging case was processing the folder. Rule answers are written
        into the transcripts and saved before the project save is attempted, so
        a project save that failed left the transcripts on disk changed with no
        record of it in the project. On the next opening those words carry a
        strength of 1.0, the low-confidence sweep passes straight over them,
        and the review never mentions them again -- and the sentence saying the
        save had failed had already been overwritten by the sentence saying how
        many recordings were processed.

        So there is one sentence, the failure is the tail of it, and the whole
        thing is urgent when there was a failure to report.
        """
        saved = self._save_project(quiet=True)
        if not saved:
            message = f"{message} {self._save_failure_text()}"
        self._set_status(message, alert=True, urgent=not saved)
        return message

    def _after_change(self, message: str) -> None:
        """Save, rebuild the lists, and say what happened and where the person now is.

        The saving comes first and its answer is carried into the sentence,
        rather than being announced separately, so that a person is never told
        a change succeeded in one breath and failed in the next.
        """
        saved = self._save_project(quiet=True)
        self.refresh()
        occurrence = self.current_occurrence()
        if occurrence is not None:
            message = (
                f"{message} Now on "
                f"{spoken_occurrence_summary(occurrence, self._occurrence_model.group_replacement)}"
            )
        else:
            message = f"{message} {self._count_label.text()}"
        if not saved:
            message = f"{message} {self._save_failure_text()}"
        self._set_status(f"{message}{self._take_focus_note()}", alert=True, urgent=not saved)

    def _take_focus_note(self) -> str:
        """Say that the focus was caught, once, as part of whatever is being said."""
        if not self._focus_caught:
            return ""
        self._focus_caught = False
        return " The focus has moved to the word groups."

    # -- Status ------------------------------------------------------------

    def _set_status(self, message: str, alert: bool = False, urgent: bool = False) -> None:
        """Show a message in the status bar, and read it out if it matters."""
        self._status_label.setText(message)
        if alert:
            announce(self._status_label, message, urgent=urgent)
