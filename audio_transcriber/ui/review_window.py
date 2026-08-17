"""The window for working through everything the transcription could not settle.

A blended transcript is not a list of words with one confidence number
against each of them. Every word carries three separate answers, to three
separate questions, from three possibly different services: what was said,
when it was said, and who said it. This window is where a person settles
the ones that could not be settled without them, and it is built around
that separation rather than hiding it. The detail panel shows the text, the
timing and the speaker as three things, and the three corrections are three
buttons, because correcting a spelling is not a statement about the clock
and never should be.

The queue is the spine of it. It holds only what needs a person, it can be
narrowed by the reason a word was flagged and by how confident the
application is, and it always says how many items it is showing out of how
many there are. That count is announced when a filter changes, because a
filter that silently empties the list is indistinguishable from a broken
one.

Playing the audio is the heart of the screen, so it is the fastest thing
on it. A word without its sentence cannot be judged, so play always
includes several seconds either side, and it plays the wider containing
span when the word's own boundaries are not known, which is exactly when a
person most needs to hear it.

The window knows nothing about providers, reconciliation or storage. It
takes a finished :class:`Transcript`, something that can play audio, and a
callback to hand corrected transcripts back to. That is what lets the whole
screen be tested without a network, an API key or an audio file.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt, Signal
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
    QSplitter,
    QStatusBar,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from audio_transcriber.audio.player import AudioPlayer
from audio_transcriber.formatting import spoken_duration
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
from audio_transcriber.ui.accessibility import announce, describe
from audio_transcriber.ui.review_queue import (
    ReviewQueueModel,
    ReviewQueueView,
    chosen_text,
    spoken_summary,
)

#: How much audio to add either side of a word when it is played. A word on
#: its own is not judgeable: the sentence around it is what tells a person
#: whether "fifteen" or "fifty" is the one that makes sense.
CONTEXT_SECONDS = 3.0

#: The same again, for the second play button, when three seconds turned
#: out not to be enough to place what was being talked about.
WIDE_CONTEXT_SECONDS = 12.0

NOT_ATTRIBUTED = "Not attributed to a service"
NOT_REPORTED = "Not reported"
NO_SPEAKER = "No speaker attributed"
NO_RISK = "None recorded"
NO_LLM_DECISION = "The language model was not consulted."
NO_LANGUAGE_EVIDENCE = "No language evidence was recorded."
NO_TIMING = "No timing at all"
NO_ITEM_SELECTED = "No item selected"
NO_AUDIO = "This word has no audio behind it, so there is nothing to play."
NO_CANONICAL_AUDIO = "There is no audio file for this transcript, so nothing can be played."

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
QUEUE = "queue"
CANDIDATES = "candidates"
CANDIDATE_TEXT = "candidate_text"
TIMING = "timing"
SPEAKER = "speaker"
LANGUAGE = "language"
RISK = "risk"
DECISION = "decision"
PLAYBACK = "playback"
CORRECT_TEXT = "correct_text"
CORRECT_SPEAKER = "correct_speaker"
TIMING_DECISION = "timing_decision"
CONFIRM = "confirm"
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
    QUEUE: ControlNote(
        "Review queue",
        "The words needing a person. Up and Down move through them.",
        _reflowed(
            """
            Each row gives the time the word occurs, the text the application settled
            on, why it needs review, and how confident the application is. The
            confidence is a word in its own column rather than a colour, so it reads
            the same however you are looking at the screen.

            Moving the highlight fills in the detail panel and reads the item out. F3
            and Shift+F3 move to the next and previous item from anywhere in the window.
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
    CORRECT_TEXT: ControlNote(
        "Correcting the text",
        "Replaces the word. The timing and the speaker are left alone.",
        _reflowed(
            """
            Type what was actually said and apply it. What the word said before is kept,
            so the change can be undone and so the correction can teach the vocabulary
            and the provider statistics.

            Correcting the text says nothing about when the word was said or who said
            it. Those are separate answers with separate sources, and they are left
            exactly as they were.
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
        "Confirming the item",
        "Leaves the word exactly as it is and takes it out of the queue. F4.",
        _reflowed(
            """
            This is the common case and the fastest thing on the screen: you listened,
            you read the candidates, and the application had it right. One keystroke,
            F4, and the word is settled.

            Your approval is treated as evidence as strong as a correction, so the word
            becomes fully confident. Nothing is recorded as changed, because nothing
            changed, so there is nothing here for the vocabulary to learn.
            """
        ),
    ),
    NAVIGATE: ControlNote(
        "Moving through the queue",
        "F3 and Shift+F3 move to the next and previous item from anywhere.",
        _reflowed(
            """
            The whole item is read out when the highlight moves: its time, the word, why
            it needs review and how confident the application is. That is enough to work
            straight down the queue without going to look at the detail panel for every
            one.

            F6 moves between the panels of this window, and says which one you have
            landed on.
            """
        ),
    ),
}


def note_for(key: str) -> ControlNote:
    return _NOTES[key]


class ReviewWindow(QMainWindow):
    """Works through the words a transcript could not settle on its own."""

    transcriptChanged = Signal(object)
    """A correction was made, carrying the whole corrected transcript."""

    def __init__(
        self,
        transcript: Transcript,
        player: AudioPlayer,
        save_correction: Callable[[Transcript], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Review: {transcript.recording_name}")
        self._transcript = transcript
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
        # Whether moving the highlight should read the item out. It should
        # whenever a person moved it, and it should not when the window
        # moved it after a correction, because the correction is announced
        # with the new item already in the same sentence.
        self._announce_selection = True
        # Set when the window has had to take the focus off a control it was
        # about to switch off. Whoever is about to say what happened adds it
        # to their own sentence, so the two never talk over each other.
        self._focus_caught = False

        self._model = ReviewQueueModel(self)

        self._build_ui()
        self._build_menus()
        self._connect_signals()
        self._load_audio()
        self.show_transcript(transcript)
        self.resize(1280, 820)

    # -- Building the window ---------------------------------------------

    def _build_ui(self) -> None:
        self._splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._splitter.addWidget(self._build_left_side())
        self._splitter.addWidget(self._build_right_side())
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 1)
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
        side = QWidget(self)
        layout = QVBoxLayout(side)
        layout.setContentsMargins(8, 8, 4, 8)
        layout.addWidget(self._build_filters(), 1)

        self._queue_label = QLabel("Review &queue", side)
        self._queue = ReviewQueueView(side)
        self._queue.setModel(self._model)
        self._queue_label.setBuddy(self._queue)
        self._explain(self._queue, QUEUE)

        # This one carries the count, so it is left unnamed for the same
        # reason as the status label.
        self._count_label = QLabel(self._model.count_text(), side)
        self._count_label.setWordWrap(True)

        layout.addWidget(self._queue_label)
        layout.addWidget(self._queue, 2)
        layout.addWidget(self._count_label)
        return side

    def _build_filters(self) -> QWidget:
        """The reason and confidence filters, as real check boxes.

        Check boxes rather than a list of strings in a combo box, because a
        combo box would let one reason be picked at a time and could not
        show which of them are on without being opened. Seventeen of them
        is a lot to tab through, so they are grouped and scroll, and the
        arrow keys walk each group.
        """
        self._filters_group = QGroupBox("Filters", self)
        self._reason_boxes: dict[ReviewReason, QCheckBox] = {}
        self._confidence_boxes: dict[Confidence, QCheckBox] = {}

        contents = QWidget(self._filters_group)
        inner = QVBoxLayout(contents)
        inner.setContentsMargins(0, 0, 0, 0)

        reasons = QGroupBox("Reasons for review", contents)
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

        categories = QGroupBox("Confidence categories", contents)
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

        self._show_everything_button = QPushButton("Show &everything", contents)
        describe(
            self._show_everything_button,
            "Show everything",
            "Puts every filter back on, so the whole queue is on show again.",
        )
        self._note_keys[self._show_everything_button] = REASON_FILTERS

        inner.addWidget(reasons)
        inner.addWidget(categories)
        inner.addWidget(self._show_everything_button)
        inner.addStretch(1)

        # At a large Windows text size the filters are taller than the
        # window, so they scroll. Qt brings whatever takes focus into view,
        # so tabbing through still works exactly as it reads.
        scroll = QScrollArea(self._filters_group)
        scroll.setWidget(contents)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # A scroll area is a container, not somewhere to land. Giving it the
        # focus would put a stop on the way through that answers no keys.
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        outer = QVBoxLayout(self._filters_group)
        outer.addWidget(scroll)
        return self._filters_group

    def _build_right_side(self) -> QWidget:
        contents = QWidget(self)
        inner = QVBoxLayout(contents)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.addWidget(self._build_details(), 3)
        for group in (self._build_playback(), self._build_corrections(), self._build_notes()):
            group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            inner.addWidget(group)
        inner.addStretch(1)

        scroll = QScrollArea(self)
        scroll.setWidget(contents)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        return scroll

    def _build_details(self) -> QWidget:
        """The three answers, shown as three things rather than one summary.

        The text, the timing and the speaker each get their own group, with
        their own source and their own confidence, because that is what
        they are. Merging them into one line would hide the very thing the
        person is being asked to decide about.
        """
        self._details_group = QGroupBox("Item details", self)
        layout = QVBoxLayout(self._details_group)

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
            candidate_form, said, "Candidate text", "Candidate text", CANDIDATE_TEXT
        )
        said_layout.addLayout(candidate_form)

        self._use_candidate_button = QPushButton("&Use this candidate", said)
        describe(
            self._use_candidate_button,
            "Use this candidate",
            "Copies the highlighted candidate into the correction box, ready to apply.",
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

        decision_label = QLabel("What the language model decided", rest)
        self._decision_text = QPlainTextEdit(rest)
        self._decision_text.setReadOnly(True)
        self._decision_text.setFixedHeight(self.fontMetrics().lineSpacing() * 4)
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

        self._play_button = QPushButton("Play the &word again", self._playback_group)
        describe(
            self._play_button,
            "Play the word again",
            f"Plays the word with {CONTEXT_SECONDS:.0f} seconds either side of it. F5.",
        )
        self._note_keys[self._play_button] = PLAYBACK

        self._play_wide_button = QPushButton("Play with &more context", self._playback_group)
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
        self._corrections_group = QGroupBox("Corrections", self)
        layout = QVBoxLayout(self._corrections_group)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        text_label = QLabel("Corrected &text", self._corrections_group)
        self._text_edit = QLineEdit(self._corrections_group)
        describe(
            self._text_edit,
            "Corrected text",
            "What the word should say. Press Enter, or use the Apply button, to "
            "replace it. The timing and the speaker are left alone. F2 comes here.",
        )
        self._note_keys[self._text_edit] = CORRECT_TEXT
        text_label.setBuddy(self._text_edit)
        form.addRow(text_label, self._text_edit)

        speaker_label = QLabel("Spea&ker", self._corrections_group)
        self._speaker_box = QComboBox(self._corrections_group)
        self._speaker_box.setEditable(True)
        describe(
            self._speaker_box,
            "Speaker",
            "Who said this word. Choose one of the speakers found in the recording, "
            "or type a label of your own. The text and the timing are left alone.",
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

        self._apply_text_button = QPushButton("&Apply the text", self._corrections_group)
        describe(
            self._apply_text_button,
            "Apply the text",
            "Replaces the word with what is in the correction box. The timing and the "
            "speaker are left exactly as they were.",
        )
        self._note_keys[self._apply_text_button] = CORRECT_TEXT

        self._apply_speaker_button = QPushButton("App&ly the speaker", self._corrections_group)
        describe(
            self._apply_speaker_button,
            "Apply the speaker",
            "Attributes the word to the chosen speaker. The text and the timing are "
            "left exactly as they were.",
        )
        self._note_keys[self._apply_speaker_button] = CORRECT_SPEAKER

        self._confirm_timing_button = QPushButton("Con&firm the timing", self._corrections_group)
        describe(
            self._confirm_timing_button,
            "Confirm the timing",
            "Says the span is right. Reasons that are about the words are left "
            "standing, so the item stays in the queue if anything else is unsettled.",
        )
        self._note_keys[self._confirm_timing_button] = TIMING_DECISION

        self._reject_timing_button = QPushButton("&Reject the timing", self._corrections_group)
        describe(
            self._reject_timing_button,
            "Reject the timing",
            "Says the span is wrong. The numbers are kept so the word can still be "
            "played, but they are marked uncertain and the item stays in the queue.",
        )
        self._note_keys[self._reject_timing_button] = TIMING_DECISION

        self._confirm_button = QPushButton(
            "&Confirm this item as correct", self._corrections_group
        )
        describe(
            self._confirm_button,
            "Confirm this item as correct",
            "Leaves the word exactly as it is and takes it out of the queue. F4.",
        )
        self._note_keys[self._confirm_button] = CONFIRM

        self._next_button = QPushButton("&Next item", self._corrections_group)
        describe(self._next_button, "Next item", "Moves to the next item in the queue. F3.")
        self._note_keys[self._next_button] = NAVIGATE

        self._previous_button = QPushButton("Previou&s item", self._corrections_group)
        describe(
            self._previous_button,
            "Previous item",
            "Moves to the previous item in the queue. Shift+F3.",
        )
        self._note_keys[self._previous_button] = NAVIGATE

        for buttons in (
            (self._apply_text_button, self._apply_speaker_button),
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
        self._notes_text.setFixedHeight(self.fontMetrics().lineSpacing() * 7)
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
        """Put every action on a menu as well as on a button.

        The buttons are down the right-hand side, which is a long way from
        the queue. A menu bar is where a screen reader user looks for what
        a window can do, and it is where the shortcut keys are written down
        rather than having to be known.
        """
        menu_bar = self.menuBar()

        item_menu = menu_bar.addMenu("&Item")
        self._next_action = self._add_action(
            item_menu, "&Next Item", QKeySequence(Qt.Key.Key_F3), self.go_to_next_item
        )
        self._previous_action = self._add_action(
            item_menu, "&Previous Item", QKeySequence("Shift+F3"), self.go_to_previous_item
        )
        item_menu.addSeparator()
        self._correct_text_action = self._add_action(
            item_menu, "Correct the &Text", QKeySequence(Qt.Key.Key_F2), self.focus_text_correction
        )
        self._apply_text_action = self._add_action(
            item_menu, "&Apply the Text", None, self.apply_text_correction
        )
        self._apply_speaker_action = self._add_action(
            item_menu, "Apply the &Speaker", None, self.apply_speaker_correction
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
            "&Confirm This Item as Correct",
            QKeySequence(Qt.Key.Key_F4),
            self.confirm_item,
        )

        playback_menu = menu_bar.addMenu("&Playback")
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

        view_menu = menu_bar.addMenu("&View")
        self._add_action(
            view_menu, "&Next Panel", QKeySequence(Qt.Key.Key_F6), self.focus_next_panel
        )
        self._add_action(view_menu, "Show &Everything", None, self.show_everything)

    def _add_action(self, menu, text: str, shortcut, slot) -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def _connect_signals(self) -> None:
        self._queue.selectionModel().currentRowChanged.connect(self._on_row_changed)
        self._candidates.selectionModel().currentRowChanged.connect(self._on_candidate_changed)
        self._model.visibleCountChanged.connect(self._on_visible_count_changed)

        self._show_everything_button.clicked.connect(self.show_everything)
        self._use_candidate_button.clicked.connect(self.use_selected_candidate)
        self._play_button.clicked.connect(lambda: self.play_span(False))
        self._play_wide_button.clicked.connect(lambda: self.play_span(True))
        self._apply_text_button.clicked.connect(self.apply_text_correction)
        self._text_edit.returnPressed.connect(self.apply_text_correction)
        self._apply_speaker_button.clicked.connect(self.apply_speaker_correction)
        self._confirm_timing_button.clicked.connect(lambda: self.decide_timing(True))
        self._reject_timing_button.clicked.connect(lambda: self.decide_timing(False))
        self._confirm_button.clicked.connect(self.confirm_item)
        self._next_button.clicked.connect(self.go_to_next_item)
        self._previous_button.clicked.connect(self.go_to_previous_item)

        self._player.positionChanged.connect(self._on_position_changed)

    # -- The focus, and the note that follows it -------------------------

    def showEvent(self, event) -> None:
        """Start watching the focus while the window is actually on screen."""
        super().showEvent(event)
        self._watch_focus(True)

    def hideEvent(self, event) -> None:
        """Stop watching as soon as it is not.

        The watch is on the application, which outlives this window. Left
        connected, every review window ever opened would go on looking at
        every focus change anywhere in the application for the rest of its
        life.
        """
        self._watch_focus(False)
        super().hideEvent(event)

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

    def focus_next_panel(self) -> None:
        """Move the focus to the next panel, and say which one.

        Each panel names the control the focus should land on. Calling
        setFocus on the panel itself is no good: a plain container keeps
        the focus and answers no keys, while a group box passes it to a
        child that is not always the first one.
        """
        panels: list[tuple[str, QWidget, object]] = [
            ("Filters", self._filters_group, self._focus_first_filter),
            ("Review queue", self._queue, self._focus_queue),
            ("Item details", self._details_group, self._focus_details),
            ("Playback", self._playback_group, self._focus_playback),
            ("Corrections", self._corrections_group, self._focus_corrections),
            ("About this control", self._notes_group, self._focus_notes),
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

    def _focus_first_filter(self) -> None:
        first = next(iter(self._reason_boxes.values()), None)
        if first is not None:
            first.setFocus(Qt.FocusReason.TabFocusReason)

    def _focus_queue(self) -> None:
        self._queue.setFocus(Qt.FocusReason.TabFocusReason)

    def _focus_details(self) -> None:
        self._candidates.setFocus(Qt.FocusReason.TabFocusReason)

    def _focus_playback(self) -> None:
        self._focus_first_enabled([self._play_button, self._play_wide_button, self._span_edit])

    def _focus_corrections(self) -> None:
        self._focus_first_enabled(
            [self._text_edit, self._confirm_button, self._next_button]
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
        self._focus_queue()

    def focus_text_correction(self) -> None:
        """Put the focus in the correction box, with the word ready to replace."""
        if not self._text_edit.isEnabled():
            return
        self._text_edit.setFocus(Qt.FocusReason.OtherFocusReason)
        self._text_edit.selectAll()

    # -- The transcript ---------------------------------------------------

    @property
    def transcript(self) -> Transcript:
        """The transcript as it stands, corrections included."""
        return self._transcript

    def show_transcript(self, transcript: Transcript) -> None:
        """Take a transcript, fill the queue, and start on its first item."""
        self._transcript = transcript
        self._fill_speaker_box()
        self._model.set_tokens(transcript.tokens)
        if self._model.rowCount():
            self._select_row(0, announce_item=False)
        else:
            self._show_token(None)

    def current_token(self) -> FinalToken | None:
        return self._model.token_at(self._queue.selected_row())

    def _fill_speaker_box(self) -> None:
        """Offer the speakers of this recording, keeping what was typed.

        The item text is what a person calls the speaker and the item data
        is the label the services used, because a correction has to record
        the label rather than the friendly name.
        """
        typed = self._speaker_box.currentText()
        self._speaker_box.clear()
        for speaker in self._transcript.speakers:
            self._speaker_box.addItem(speaker.display_name, speaker.id)
        self._speaker_box.setCurrentText(typed)

    def _load_audio(self) -> None:
        """Give the player the one file every timestamp refers to."""
        audio = self._transcript.canonical_audio
        if audio is None:
            return
        self._player.load(audio.path)

    # -- Moving through the queue ------------------------------------------

    def _select_row(self, row: int, announce_item: bool = True) -> None:
        """Move the highlight, leaving the announcing to the one place that does it.

        Setting the current index raises the signal below, so the panel is
        filled in and the item read out from there. Doing it here as well
        would say every item twice.
        """
        self._announce_selection = announce_item
        try:
            self._queue.select_row(row)
        finally:
            self._announce_selection = True
        # A reset model sometimes lands on the same row it was already on,
        # which raises no signal, so the panel is brought up to date here
        # whether the signal came or not.
        self._show_token(self._model.token_at(row))

    def _on_row_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
        token = self._model.token_at(current.row()) if current.isValid() else None
        self._show_token(token)
        if token is not None and self._announce_selection:
            announce(self._queue, spoken_summary(token))

    def go_to_next_item(self) -> None:
        self._step(1)

    def go_to_previous_item(self) -> None:
        self._step(-1)

    def _step(self, delta: int) -> None:
        """Move up or down the queue, saying so when there is nowhere to go.

        Stopping at the ends rather than wrapping round matters here: a
        queue is worked through from one end to the other, and silently
        starting again at the top would have somebody reviewing the same
        words twice without noticing.
        """
        count = self._model.rowCount()
        if count == 0:
            self._set_status(self._model.count_text(), alert=True)
            return
        row = self._queue.selected_row()
        target = 0 if row < 0 else row + delta
        if target < 0:
            self._set_status("This is the first item in the queue.", alert=True)
            return
        if target >= count:
            self._set_status("This is the last item in the queue.", alert=True)
            return
        self._select_row(target)

    # -- The filters -------------------------------------------------------

    def _on_reason_toggled(self, reason: ReviewReason, shown: bool) -> None:
        self._model.set_reason_shown(reason, shown)
        self._filters_changed()

    def _on_confidence_toggled(self, confidence: Confidence, shown: bool) -> None:
        self._model.set_confidence_shown(confidence, shown)
        self._filters_changed()

    def show_everything(self) -> None:
        """Put every filter back on, in one change and one announcement."""
        self._setting_filters = True
        try:
            for box in list(self._reason_boxes.values()) + list(self._confidence_boxes.values()):
                box.setChecked(True)
            self._model.show_everything()
        finally:
            self._setting_filters = False
        self._filters_changed()

    def _filters_changed(self) -> None:
        """Say how many items are showing, because an empty list says nothing.

        A filter that has quietly hidden everything looks exactly like a
        broken queue. The count is on screen as well, so a sighted user and
        a screen reader user are told the same thing.
        """
        if self._setting_filters:
            return
        # The selection is settled first, because settling it is what may
        # take the focus off a control, and the sentence has to be able to
        # say so.
        self._keep_selection_in_range()
        self._set_status(f"{self._model.count_text()}{self._take_focus_note()}", alert=True)

    def _take_focus_note(self) -> str:
        """Say that the focus was caught, once, as part of whatever is being said."""
        if not self._focus_caught:
            return ""
        self._focus_caught = False
        return " The focus has moved to the review queue."

    def _keep_selection_in_range(self) -> None:
        """Land somewhere sensible after a filter changed what is on show."""
        if self._model.rowCount() == 0:
            self._show_token(None)
            return
        if self._queue.selected_row() < 0:
            self._select_row(0, announce_item=False)

    def _on_visible_count_changed(self, _shown: int, _total: int) -> None:
        text = self._model.count_text()
        self._count_label.setText(text)
        # The count also goes into the table's description, which a screen
        # reader reads after the name when the table takes focus. Somebody
        # tabbing into the queue then hears how much of it they are seeing
        # without having to go and find the label.
        note = note_for(QUEUE)
        describe(self._queue, note.title, f"{text} {note.summary}")
        self._queue.setToolTip(f"{text} {note.summary}")

    # -- The item on show --------------------------------------------------

    def _show_token(self, token: FinalToken | None) -> None:
        """Fill in every part of the detail panel for one word, or clear it."""
        self._candidates_model.set_rows(candidate_rows(self._transcript, token))
        if self._candidates_model.rowCount():
            self._candidates.setCurrentIndex(
                self._candidates_model.index(0, CANDIDATE_COLUMN_SERVICE)
            )
        else:
            self._set_value(self._candidate_text, "")

        if token is None:
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
                self._span_edit,
            ):
                self._set_value(edit, NO_ITEM_SELECTED)
            self._decision_text.setPlainText(NO_LLM_DECISION)
            self._text_edit.setText("")
            self._set_item_controls_enabled(False)
            return

        self._set_value(self._timing_edit, timing_text(token))
        self._set_value(self._timing_status_edit, _TIMING_STATUS_TEXT[token.timing_status])
        self._set_value(self._timing_source_edit, provider_name(token.timing_source))
        self._set_value(self._timing_confidence_edit, token.timing_confidence.display_name)
        self._set_value(self._alignment_edit, _ALIGNMENT_STATUS_TEXT[token.alignment_status])

        self._set_value(self._speaker_edit, speaker_text(self._transcript, token))
        self._set_value(self._speaker_source_edit, provider_name(token.speaker_source))
        self._set_value(self._speaker_confidence_edit, token.speaker_confidence.display_name)

        self._set_value(self._language_edit, token.language.display_name)
        self._set_value(self._language_evidence_edit, language_evidence_text(token))
        self._set_value(self._risk_edit, risk_text(token.risk_categories))
        self._decision_text.setPlainText(token.llm_decision or NO_LLM_DECISION)

        self._set_value(self._span_edit, self._span_description())
        self._text_edit.setText(token.text)
        self._text_edit.setCursorPosition(0)
        self._speaker_box.setCurrentText(self._speaker_box_text(token))
        self._set_item_controls_enabled(True)

    def _speaker_box_text(self, token: FinalToken) -> str:
        """What to show in the speaker box for a word.

        A speaker the recording knows about is shown by the name a person
        gave them; anything else is shown by its raw label. Both come back
        as the right label when the correction is applied, which is what
        stops the friendly name from being saved as though it were one.
        """
        if not token.speaker:
            return ""
        speaker = self._transcript.speaker_for(token.speaker)
        return speaker.display_name if speaker is not None else token.speaker

    def _span_description(self) -> str:
        """What pressing Play would actually play, said in full."""
        span = self.span_to_play()
        if self._transcript.canonical_audio is None:
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
        widgets: list[tuple[QWidget, bool]] = [
            (self._play_button, playable),
            (self._play_wide_button, playable),
            (self._use_candidate_button, enabled),
            (self._text_edit, enabled),
            (self._speaker_box, enabled),
            (self._apply_text_button, enabled),
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
            (self._apply_text_action, enabled),
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
            self._focus_queue()
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
        """Copy the highlighted candidate into the correction box."""
        text = self._candidate_text.text()
        if not text:
            self._set_status("No candidate is highlighted.", alert=True, urgent=True)
            return
        self._text_edit.setText(text)
        self._text_edit.setCursorPosition(0)
        self._set_status(
            f"{text} put in the correction box. Apply it to change the word.", alert=True
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
        audio = self._transcript.canonical_audio
        return span.padded(padding, padding, limit=audio.duration if audio else None)

    def play_span(self, wide: bool = False) -> bool:
        """Play the audio around the selected word. Returns whether it started."""
        if self._transcript.canonical_audio is None:
            self._set_status(NO_CANONICAL_AUDIO, alert=True, urgent=True)
            return False
        span = self.span_to_play(wide)
        if span is None:
            self._set_status(NO_AUDIO, alert=True, urgent=True)
            return False
        self._stop_at_ms = int(span.end * 1000)
        self._player.seek_to(int(span.start * 1000))
        self._player.play()
        self._set_status(
            f"Playing {_seconds_phrase(span.duration)} from {spoken_duration(span.start)}.",
            alert=True,
        )
        return True

    def _on_position_changed(self, milliseconds: int) -> None:
        """Stop at the end of the region, since Qt only knows about whole files."""
        if self._stop_at_ms is None:
            return
        if milliseconds >= self._stop_at_ms:
            self._stop_at_ms = None
            self._player.pause()

    # -- Corrections -------------------------------------------------------

    def apply_text_correction(self) -> bool:
        """Replace the word with what is in the correction box."""
        token = self.current_token()
        if token is None:
            return False
        text = self._text_edit.text().strip()
        if not text:
            self._set_status(
                "The correction cannot be empty. Type what was actually said.",
                alert=True,
                urgent=True,
            )
            return False
        if text == token.text:
            self._set_status(
                "The text is unchanged. Use Confirm this item as correct to settle it "
                "as it stands.",
                alert=True,
                urgent=True,
            )
            return False
        corrected = self._transcript.with_correction(token.id, text=text)
        self._apply(
            corrected,
            token.id,
            f"Text corrected to {text}. The timing and the speaker are unchanged.",
        )
        return True

    def apply_speaker_correction(self) -> bool:
        """Attribute the word to the chosen speaker, and nothing else."""
        token = self.current_token()
        if token is None:
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
        corrected = self._transcript.with_correction(token.id, speaker=speaker)
        self._apply(
            corrected,
            token.id,
            f"Speaker set to {speaker}. The text and the timing are unchanged.",
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
        token = self.current_token()
        if token is None:
            return False
        decided = with_timing_decision(self._transcript, token.id, accepted)
        message = (
            "Timing confirmed. The text and the speaker are unchanged."
            if accepted
            else "Timing rejected and marked uncertain. The text and the speaker are "
            "unchanged, and the item stays in the queue."
        )
        self._apply(decided, token.id, message)
        return True

    def confirm_item(self) -> bool:
        """Settle the word exactly as it stands, which is the common case."""
        token = self.current_token()
        if token is None:
            return False
        word = chosen_text(token)
        confirmed = with_confirmation(self._transcript, token.id)
        self._apply(confirmed, token.id, f"{word} confirmed as correct.")
        return True

    def _apply(self, transcript: Transcript, token_id: str, message: str) -> None:
        """Take a corrected transcript, hand it on, and say what happened.

        The change happens away from where the focus lands next, so it is
        always announced. A word that no longer needs review leaves the
        queue, and the selection moves to whatever took its place, so the
        one announcement says both what changed and where the person now
        is.
        """
        self._transcript = transcript
        self._fill_speaker_box()
        row_before = self._queue.selected_row()
        self._model.set_tokens(transcript.tokens)

        target = self._model.row_for_token_id(token_id)
        if target < 0:
            # The word has left the queue, so the row it occupied now holds
            # the next one along, which is where the work carries on.
            target = min(max(row_before, 0), self._model.rowCount() - 1)
        if target >= 0 and self._model.rowCount():
            self._select_row(target, announce_item=False)
            following = self._model.token_at(target)
            message = f"{message} Now on {spoken_summary(following)}"
        else:
            self._show_token(None)
            message = f"{message} {self._model.count_text()}"

        self._set_status(f"{message}{self._take_focus_note()}", alert=True)
        if self._save_correction is not None:
            self._save_correction(transcript)
        self.transcriptChanged.emit(transcript)

    # -- Status ------------------------------------------------------------

    def _set_status(self, message: str, alert: bool = False, urgent: bool = False) -> None:
        """Show a message in the status bar, and read it out if it matters."""
        self._status_label.setText(message)
        if alert:
            announce(self._status_label, message, urgent=urgent)
