"""The two tables a whole folder is reviewed from, and the words in their cells.

Reviewing one recording is a list. Reviewing a folder is two lists, because
the work has two levels: a *word*, which spans files, and an *occurrence* of
that word, which is one moment in one recording. Deciding what a surname
should say is a question about the word and is answered once; deciding who
said it, and whether the clock is right, is a question about the occurrence
and has a different answer every time. Two tables is what keeps those two
questions from being asked in the same breath.

Both are ``QTableView`` over a real model, for the reason
:mod:`~vox_verbatim.ui.review_queue` already gives: Qt hands a table to
Windows accessibility with genuine rows, columns and headers, and a
hand-drawn control would have to reproduce all of that and would get some of
it wrong.

Three decisions in here are worth explaining before the code.

**Every cell is a word or a number, and nothing is carried by position, by a
colour or by an icon.** That applies to the two things a reader is most
likely to be shown by a mark: whether an item has been reviewed, and *how* it
came to be reviewed. A word somebody judged and a word a project rule judged
on their behalf are different facts, so they are different sentences, and
both are in a column of their own.

**The first table holds two kinds of row, and each says which kind it is.**
Low-confidence groups come first and the words already flagged for review
follow them as a second block, so that working down the list stays
predictable. The block boundary is a visual fact, though, and a visual fact
is no use to somebody hearing the rows read out one at a time, so the reason
a row is in the list is a column rather than an inference from where it sits.

**An item flagged for review is described as a group of one.** It is not one:
it is a single word of a single transcript, with no project record behind it.
But the window has to play it, correct it, confirm it and describe it exactly
as it does a member of a real group, and giving it the same shape here means
one code path rather than two throughout the window, with the second one
inevitably being the one that gets a case wrong. :func:`uncertainty_occurrence`
builds that description. What it produces is never saved into the project.

The flagged words themselves *are* saved, though, as
:class:`~vox_verbatim.transcription.project.FlaggedItem`, and the pair of
functions that carry them across that boundary is at the top of this module.
:func:`flagged_in` takes them out of a transcript that is being read anyway,
and :func:`restored_token` turns a saved one back into the shape the rest of
the window works in. The second is what lets the window draw the whole block
without opening a single transcript, which is the difference between a folder
of fifty recordings opening in 0.04 seconds and taking 37.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableView, QWidget

from vox_verbatim.formatting import format_duration, spoken_duration
from vox_verbatim.transcription.diarisation import speaker_doubt_stretches
from vox_verbatim.transcription.confidence import (
    STRENGTH_NOT_MEASURED,
    strength_display,
    strength_percentage,
)
from vox_verbatim.transcription.model import (
    Confidence,
    FinalToken,
    Language,
    ReviewReason,
    ReviewStatus,
    RiskCategory,
    Transcript,
)
from vox_verbatim.transcription.normalise import normalise
from vox_verbatim.transcription.project import (
    FlaggedItem,
    Occurrence,
    ProjectState,
    SpeakerDoubtItem,
    WordGroup,
)
from vox_verbatim.ui.review_queue import reason_text

# -- The first table: the words -------------------------------------------

GROUP_COLUMN_WORD = 0
GROUP_COLUMN_WHEN = 1
GROUP_COLUMN_COUNT = 2
GROUP_COLUMN_CONFIDENCE = 3
GROUP_COLUMN_WHY = 4
GROUP_COLUMN_REVIEWED = 5
GROUP_COLUMN_TOTAL = 6

#: The columns the simple review window leaves out. What is left -- the
#: word, when it was said and why it is here -- is what a decision needs;
#: the rest is still in the model, and comes back with the details.
GROUP_DETAIL_COLUMNS = (GROUP_COLUMN_COUNT, GROUP_COLUMN_CONFIDENCE, GROUP_COLUMN_REVIEWED)

GROUP_COLUMN_TITLES = (
    "Word",
    "Time",
    "Occurrences",
    "Confidence",
    "Why it needs review",
    "Reviewed",
)

# -- The second table: where each word occurs -----------------------------

OCCURRENCE_COLUMN_FILE = 0
OCCURRENCE_COLUMN_WHEN = 1
OCCURRENCE_COLUMN_CONFIDENCE = 2
OCCURRENCE_COLUMN_LANGUAGE = 3
OCCURRENCE_COLUMN_REPLACEMENT = 4
OCCURRENCE_COLUMN_REVIEWED = 5
OCCURRENCE_COLUMN_TOTAL = 6

OCCURRENCE_COLUMN_TITLES = (
    "Source file",
    "When",
    "Confidence",
    "Language",
    "Replacement",
    "Reviewed",
)

#: Why a row is in the first table. A low-confidence group is there because
#: the application's own estimate of the words in it is weak; an uncertainty
#: item is there because some rule in the pipeline flagged it, and that rule
#: has a name which is worth more to the reader than the word "flagged".
WHY_LOW_CONFIDENCE = "Low confidence"
WHY_ISOLATED = "Low confidence, kept on its own"
WHY_STALE = "Not found in the transcript any more"

#: A group the person put together themselves, as opposed to one the analysis
#: suggested. The difference is worth a column of its own for the same reason
#: an isolated occurrence already says "kept on its own": both are statements
#: the person made about a word, and both survive every later regrouping,
#: whereas a suggested group is only the analysis's current opinion and is
#: rebuilt from scratch every time a threshold moves. Somebody deciding
#: whether to trust a grouping needs to know which of the two they are
#: looking at, and the fact was stored and preserved from the beginning
#: without ever being shown.
WHY_USER_CREATED = "Low confidence, grouped by you"

#: The reviewed state of a word, always as a sentence and never as an absence.
#: "Not reviewed" is said out loud rather than left as an empty cell, because
#: an empty cell is silent to a screen reader and a listener cannot tell
#: silence from a column that was not read.
NOT_REVIEWED = "Not reviewed"
REVIEWED_AS_DETECTED = "Correct as detected"
REVIEWED_AUTOMATICALLY = "Applied automatically from a project rule"
REVIEWED_CONFIRMED = "Confirmed as correct"
REVIEWED_CORRECTED = "Corrected by hand"
REVIEWED_PLAIN = "Reviewed"

NO_REPLACEMENT = "None set"
UNKNOWN_TIME_DISPLAY = "Unknown"
UNKNOWN_TIME_SPOKEN = "the time is not known"
NO_LANGUAGE = "Not known"

#: The invalid index, which is what Qt means by "the top of the model". It is
#: a named constant rather than a fresh ``QModelIndex()`` written into each
#: signature, which is how the older model classes in this package do it. An
#: invalid index holds no model and no row, so one of them is as good as a
#: thousand, and writing it this way also keeps the linter quiet about a
#: function call in a default argument. Qt itself always passes a real index,
#: so this is only ever the default for a Python caller.
TOP_LEVEL = QModelIndex()

#: Prefixes that keep the three kinds of row apart in one namespace. The
#: window remembers where the person was by this key and looks them up again
#: after the lists are rebuilt, so two rows must never be able to produce the
#: same one: a group identifier and an occurrence identifier are both uuid
#: hex, and without the prefix a saved position could land on the wrong row.
GROUP_KEY_PREFIX = "group:"
LOOSE_KEY_PREFIX = "loose:"
UNCERTAINTY_KEY_PREFIX = "token:"

# -- The third table: where the speaker is in doubt ------------------------

SPEAKER_DOUBT_COLUMN_FILE = 0
SPEAKER_DOUBT_COLUMN_FROM = 1
SPEAKER_DOUBT_COLUMN_TO = 2
SPEAKER_DOUBT_COLUMN_SPEAKERS = 3
SPEAKER_DOUBT_COLUMN_TOTAL = 4

SPEAKER_DOUBT_COLUMN_TITLES = (
    "Source file",
    "From",
    "To",
    "Speakers heard",
)

#: Said in place of a second speaker nobody could name, so the cell never
#: reads as one person with nothing to be in doubt about.
ANOTHER_SPEAKER = "another speaker"
UNKNOWN_SPEAKER = "Unknown speaker"


def flagged_in(recording_name: str, transcript: Transcript) -> list[FlaggedItem]:
    """Everything of one transcript that belongs in the second block, to be saved.

    This is called at the one moment a transcript is genuinely being read --
    while the analysis is walking the folder, or when a correction has just
    been written into a file that is therefore already in hand. Nothing here
    ever causes a read of its own, and it must not be made to: reading a folder
    to fill this list in is the 37 seconds the whole arrangement exists to
    avoid, and it would be spent on every opening rather than once.

    Two kinds of word come out. One still wants a person, and carries the
    reasons some rule in the pipeline recorded against it. The other has been
    settled, by somebody correcting it or confirming it, and is kept because
    asking to see reviewed words has to be able to show it without the folder
    being read again. A word that is neither is not in this list at all, which
    is the ordinary case for almost every word of a transcript.

    The text kept is what the word was *detected* as rather than what it says
    now, because that is the spelling the person recognises from the row they
    worked on, and because it is the one thing a later correction cannot
    recover.

    A word held only because its speaker is in doubt is left out as well. Its
    text is settled, and :func:`speaker_doubts_in` lists that doubt once for
    each stretch of speech instead.
    """
    items: list[FlaggedItem] = []
    for token in transcript.tokens:
        settled = not token.needs_review
        if not settled and not token.needs_word_review:
            continue
        if settled and not (
            token.human_corrected or token.review_status is ReviewStatus.CONFIRMED
        ):
            continue
        detected = token.original_text if token.original_text is not None else token.text
        items.append(
            FlaggedItem(
                recording_name=recording_name,
                token_id=token.id,
                text=detected,
                start=token.start,
                reasons=[reason.value for reason in token.review_reasons],
                confidence=token.confidence.value,
                risk_categories=[category.value for category in token.risk_categories],
                settled=settled,
            )
        )
    return items


def speaker_doubts_in(recording_name: str, transcript: Transcript) -> list[SpeakerDoubtItem]:
    """Every stretch of one transcript whose speaker is in doubt, to be saved.

    Called at the same moments as :func:`flagged_in`, for the same reason. The
    speakers are kept by the names the transcript gives them, because the
    window lists these without opening the transcript to look the names up.
    """
    items: list[SpeakerDoubtItem] = []
    for stretch in speaker_doubt_stretches(transcript.tokens):
        speakers: list[str] = []
        for label in stretch.speakers:
            speaker = transcript.speaker_for(label)
            speakers.append(speaker.display_name if speaker is not None else f"Speaker {label}")
        items.append(
            SpeakerDoubtItem(
                recording_name=recording_name,
                start=stretch.start,
                end=stretch.end,
                speakers=speakers,
                token_ids=list(stretch.token_ids),
            )
        )
    return items


def speakers_heard_text(item: SpeakerDoubtItem) -> str:
    """The two people a stretch might belong to, such as "Speaker A or Speaker B"."""
    names = list(item.speakers) or [UNKNOWN_SPEAKER]
    if len(names) < 2:
        names.append(ANOTHER_SPEAKER)
    return " or ".join(names)


def spoken_speaker_doubt_summary(item: SpeakerDoubtItem) -> str:
    """The whole row in one sentence, for the tooltip and the announcement."""
    return (
        f"{item.recording_name}. From {time_spoken(item.start)} to "
        f"{time_spoken(item.end)}. {speakers_heard_text(item)}."
    )


def speaker_doubt_count_text(count: int) -> str:
    """How many stretches are listed, said plainly even when there are none.

    An empty table is silent to a screen reader, and a listener cannot tell
    it from one that failed to load, so "none" is said in words too.
    """
    if count == 0:
        return "No speaker doubts. The services agree on who said every word."
    if count == 1:
        return "1 stretch of speech where the speaker is in doubt."
    return f"{count} stretches of speech where the speaker is in doubt."


def restored_token(item: FlaggedItem) -> FinalToken:
    """A saved flagged word, back in the shape the rest of the window works in.

    Everything downstream of here -- the reason and confidence filters, the
    rows, the announcements -- was written against
    :class:`~vox_verbatim.transcription.model.FinalToken`, and it stays
    that way rather than growing a second path for saved items. A second path
    would have to make the same decisions about filtering and wording all over
    again, and would be the one that got a case wrong, because it is the one
    nobody exercises while working normally.

    What comes back is honest about being a summary. The identifier, the text,
    the time, the reasons and the confidence category are what was saved. The
    fields that were not saved are left at their own defaults, so the word
    reads as having no measured strength, no known language and no candidates,
    which is exactly what this record knows about it. The moment the person
    selects the row, the window loads the transcript for the detail panel and
    every one of those is answered properly from the real word.

    The confidence category is put on all three parts of the word rather than
    one, because ``FinalToken.confidence`` is the weakest of the three and
    setting one would leave the other two at "unresolved" and report that
    instead. One saved category standing for all three is the same claim the
    saved value already made.

    An unsettled item is marked as pending as well as carrying its reasons.
    Some words are flagged by their status alone with no reason recorded
    against them, and without the status such a word would come back looking
    settled and quietly leave the list.
    """
    confidence = _confidence_value(item.confidence)
    return FinalToken(
        id=item.token_id,
        text=item.text,
        start=item.start,
        text_confidence=confidence,
        timing_confidence=confidence,
        speaker_confidence=confidence,
        review_status=ReviewStatus.SETTLED if item.settled else ReviewStatus.PENDING,
        review_reasons=[] if item.settled else _reason_values(item.reasons),
        risk_categories=_risk_values(item.risk_categories),
    )


def _confidence_value(value: str) -> Confidence:
    """A saved confidence category, or the weakest where it means nothing here.

    A category this version does not recognise was written by a newer one, and
    treating it as high confidence would hide the word. Falling to the weakest
    keeps it in front of the person, which is the only safe direction to be
    wrong in.
    """
    try:
        return Confidence(value)
    except ValueError:
        return Confidence.UNRESOLVED


def _reason_values(values: list[str]) -> list[ReviewReason]:
    """The saved reasons this version understands, in the order they were saved.

    A reason it does not understand is dropped rather than shown as a raw code,
    for the same reason :func:`language_name` refuses to: a code nobody can
    read is worse than an honest absence, and the word itself is still listed.
    """
    reasons: list[ReviewReason] = []
    for value in values:
        try:
            reasons.append(ReviewReason(value))
        except ValueError:
            continue
    return reasons


def _risk_values(values: list[str]) -> list[RiskCategory]:
    """The saved kinds of value this version understands, in their saved order.

    One it does not understand is dropped, as an unknown reason is, and the
    reason then reads in its general wording.
    """
    categories: list[RiskCategory] = []
    for value in values:
        try:
            categories.append(RiskCategory(value))
        except ValueError:
            continue
    return categories


def group_key(group_id: str) -> str:
    return f"{GROUP_KEY_PREFIX}{group_id}"


def loose_key(occurrence_id: str) -> str:
    return f"{LOOSE_KEY_PREFIX}{occurrence_id}"


def uncertainty_key(recording_name: str, token_id: str) -> str:
    return f"{UNCERTAINTY_KEY_PREFIX}{recording_name}:{token_id}"


@dataclass(frozen=True)
class GroupRow:
    """One word of the first table, with every cell already in words.

    The occurrences are carried on the row rather than looked up again when
    the second table is filled in. The two tables are then showing the same
    objects by construction, which is the only way to be sure that the count
    in the first table is the number of rows in the second.
    """

    key: str
    word: str
    word_spoken: str
    count: int
    file_count: int
    confidence: str
    confidence_spoken: str
    why: str
    reviewed: str
    occurrences: tuple[Occurrence, ...]
    group_id: str | None
    """The project group behind this row, where there is one.

    ``None`` for a loose occurrence and for an item flagged by the pipeline.
    Both of those are one word in one file with no group to speak for them,
    so the group-wide actions have nothing to act on and are switched off.
    """

    replacement: str | None = None
    """The group's replacement, which each member may override with its own."""


def language_name(value: str) -> str:
    """The display name of a stored language value.

    The value is a plain string on disk rather than an enumeration, so
    anything unrecognised is described as not known rather than shown as the
    raw code. A code nobody can read is worse than an honest absence.
    """
    try:
        language = Language(value)
    except ValueError:
        return NO_LANGUAGE
    if language is Language.UNKNOWN:
        return NO_LANGUAGE
    return language.display_name


def time_display(seconds: float | None) -> str:
    """The compact position for a table cell, such as ``1:32``."""
    return UNKNOWN_TIME_DISPLAY if seconds is None else format_duration(seconds)


def time_spoken(seconds: float | None) -> str:
    """The same position written out, for a screen reader to read."""
    return UNKNOWN_TIME_SPOKEN if seconds is None else spoken_duration(seconds)


def group_time_display(occurrences: tuple[Occurrence, ...]) -> str:
    """Where a word first occurs, and how many more times, for a table cell.

    The first occurrence is the one the second list opens on, so the time in
    this cell is the time that playing the word plays. "1:05 and 3 more"
    rather than a list of times, which would not fit and would not be read.
    """
    if not occurrences:
        return UNKNOWN_TIME_DISPLAY
    first = time_display(occurrences[0].start)
    more = len(occurrences) - 1
    return first if more == 0 else f"{first} and {more} more"


def group_time_spoken(occurrences: tuple[Occurrence, ...]) -> str:
    """The same, written out for a screen reader."""
    if not occurrences:
        return UNKNOWN_TIME_SPOKEN
    first = time_spoken(occurrences[0].start)
    more = len(occurrences) - 1
    return first if more == 0 else f"{first}, and {more} more"


def confidence_range(occurrences: tuple[Occurrence, ...]) -> tuple[str, str]:
    """The spread of confidence across a group, to show and to say.

    A range rather than an average, because an average is a number no member
    of the group actually has and the two ends are what tell a person whether
    the group is uniformly shaky or holds one weak spelling among strong ones.
    Where every member sits at the same figure the range collapses to that one
    figure, since "42% to 42%" is a sentence nobody would write by hand.

    Words with no saved strength are left out of the range rather than counted
    as nought, and a group where nobody has one is described as not measured.
    """
    strengths = [
        occurrence.confidence_strength
        for occurrence in occurrences
        if occurrence.confidence_strength is not None
    ]
    if not strengths:
        return STRENGTH_NOT_MEASURED, STRENGTH_NOT_MEASURED
    lowest = min(strengths)
    highest = max(strengths)
    if strength_display(lowest) == strength_display(highest):
        return strength_display(lowest), strength_percentage(lowest)
    return (
        f"{strength_display(lowest)} to {strength_display(highest)}",
        f"{strength_percentage(lowest)} to {strength_percentage(highest)}",
    )


def group_reviewed_text(group: WordGroup, occurrences: tuple[Occurrence, ...]) -> str:
    """How this word came to be settled, or that it has not been.

    Which kind it was matters as much as whether it happened. Somebody
    hearing the row read out must be able to tell a word they judged from a
    word a project rule judged for them, because only one of those is a
    decision they remember making.
    """
    if occurrences and all(occurrence.auto_applied for occurrence in occurrences):
        return REVIEWED_AUTOMATICALLY
    if group.correct_as_detected:
        return REVIEWED_AS_DETECTED
    if group.replacement:
        return f"Replaced with {group.replacement}"
    if group.reviewed:
        return REVIEWED_PLAIN
    return NOT_REVIEWED


def occurrence_reviewed_text(occurrence: Occurrence) -> str:
    """How this one occurrence was settled, in the same words as its group.

    Staleness is reported here as well, and before anything else, because an
    occurrence whose word is no longer in the transcript cannot be corrected
    at all. Saying "reviewed" about it would describe a decision that can no
    longer be applied to anything.
    """
    if occurrence.stale:
        return WHY_STALE
    if occurrence.auto_applied:
        return REVIEWED_AUTOMATICALLY
    if occurrence.correct_as_detected:
        return REVIEWED_AS_DETECTED
    if occurrence.replacement:
        return f"Replaced with {occurrence.replacement}"
    if occurrence.reviewed:
        return REVIEWED_PLAIN
    return NOT_REVIEWED


def uncertainty_occurrence(recording_name: str, token: FinalToken) -> Occurrence:
    """Describe a flagged word of a transcript the way a project word is described.

    Nothing here is saved. The occurrence exists so that the detail panel, the
    playback, the corrections and the announcements have one shape to work
    with, whether the word came out of the low-confidence sweep or out of a
    rule in the pipeline that flagged it for a stated reason.

    The reviewed fields are read back off the transcript rather than
    remembered, so they survive closing the window: a word a person confirmed
    carries :attr:`ReviewStatus.CONFIRMED`, and a word whose *text* they
    replaced carries ``text_corrected`` together with what it originally said.

    The replacement is taken from ``text_corrected`` and never from
    ``human_corrected``, and the difference is the whole reason the newer
    field exists. ``human_corrected`` means "a person decided something about
    this word", which a timing decision sets as readily as a correction does.
    Reading the replacement off it made rejecting a word's timing fill in the
    Replacement column with the word's own unchanged text, so the Occurrences
    table said "Replacement: contract" and "Reviewed: Replaced with contract"
    about somebody who had confirmed a clock and changed no letter of
    anything. That column exists precisely to keep the three answers apart,
    and it was the one place reporting them as the same answer.
    """
    detected = token.original_text if token.original_text is not None else token.text
    return Occurrence(
        id=uncertainty_key(recording_name, token.id),
        recording_name=recording_name,
        token_id=token.id,
        detected_text=detected,
        # Worked out from the detected spelling, not read off the token. The
        # token's own field describes what the word says *now*, which is the
        # right answer to a different question: after a correction it names the
        # replacement while ``detected_text`` beside it still names what the
        # services produced, and the two fields then describe two different
        # words. Every rule the project learns is keyed on this field, and a
        # rule keyed on the spelling this window last wrote would answer a
        # later file that already says the right thing and would never answer
        # the one that says the wrong thing. Correcting the same word twice
        # also left the first rule standing, because withdrawing it looked for
        # a form that no longer matched anything.
        normalised_text=normalise(detected),
        start=token.start,
        end=token.end,
        confidence_strength=token.confidence_strength,
        language=token.language.value,
        context_before="",
        context_after="",
        reviewed=not token.needs_review,
        correct_as_detected=token.review_status is ReviewStatus.CONFIRMED,
        replacement=token.text if token.text_corrected else None,
    )


def _uncertainty_reviewed_text(token: FinalToken) -> str:
    """How a flagged word was settled, or that it has not been.

    Still needing review comes first and beats everything else, because some
    of the decisions a person makes here are about one part of a word and
    leave the rest unsettled. Confirming the timing of a word whose spelling
    is still disputed marks it as touched by a person without settling it,
    and reading that as "corrected by hand" would take it out of the list
    with the argument about its spelling unfinished.

    A word that is settled without saying how is one restored from the project
    by :func:`restored_token`, whose record kept that somebody had finished
    with it and not which of the two ways they finished with it. "Reviewed" is
    what can honestly be said about it. Guessing at "confirmed as correct"
    would be a claim about what a person did, made on no evidence, in the one
    column that exists to tell their decisions apart, and it would be wrong
    every time the word had in fact been corrected.

    "Corrected by hand" is read off ``text_corrected`` rather than off
    ``human_corrected``, for the reason :func:`uncertainty_occurrence` gives.
    A word whose only human decision was to confirm its timing carries
    ``human_corrected`` and has had no letter of it changed, and it used to be
    reported here as having been corrected. It now falls through to
    "Reviewed", which is the true and much duller thing to say about it.
    """
    if token.needs_review:
        return NOT_REVIEWED
    if token.text_corrected:
        return REVIEWED_CORRECTED
    if token.review_status is ReviewStatus.CONFIRMED:
        return REVIEWED_CONFIRMED
    return REVIEWED_PLAIN


def build_group_rows(
    state: ProjectState,
    uncertainties: list[tuple[str, FinalToken]],
    show_reviewed: bool,
    show_other_uncertainties: bool,
) -> list[GroupRow]:
    """Every row of the first table, in the order a person works down it.

    Low-confidence words come first, weakest first, because the weakest word
    is the one most likely to be wrong and a queue is worked from the end that
    matters. The words the pipeline flagged follow as a second block rather
    than being mixed in among them, so that the list does not reorder itself
    around the reader as they go.

    A word the person has settled is left out unless they ask to see it. It is
    left out rather than shown greyed, struck through or moved to the bottom,
    because all three of those say "done" by appearance alone.

    Loose occurrences share the first block with the groups. Every weak word
    has a group of its own after the analysis has run, so the only loose ones
    are those the person deliberately took out of a group and those that went
    stale, and both must stay reachable: a word with no row cannot be played,
    corrected or confirmed, and would simply disappear.
    """
    rows: list[GroupRow] = []
    for group in state.groups:
        occurrences = tuple(state.occurrences_of(group.id))
        if not occurrences:
            continue
        reviewed = group_reviewed_text(group, occurrences)
        if reviewed != NOT_REVIEWED and not show_reviewed:
            continue
        display, spoken = confidence_range(occurrences)
        rows.append(
            GroupRow(
                key=group_key(group.id),
                word=group.representative_text,
                word_spoken=group.representative_text,
                count=len(occurrences),
                file_count=len({item.recording_name for item in occurrences}),
                confidence=display,
                confidence_spoken=spoken,
                why=WHY_USER_CREATED if group.user_created else WHY_LOW_CONFIDENCE,
                reviewed=reviewed,
                occurrences=occurrences,
                group_id=group.id,
                replacement=group.replacement,
            )
        )

    for occurrence in state.loose_occurrences():
        reviewed = occurrence_reviewed_text(occurrence)
        # A stale occurrence is never hidden as though it had been dealt with.
        # It is the one row in the list that describes a decision the project
        # could not keep, and the person has to be able to find it.
        if reviewed not in (NOT_REVIEWED, WHY_STALE) and not show_reviewed:
            continue
        display, spoken = confidence_range((occurrence,))
        rows.append(
            GroupRow(
                key=loose_key(occurrence.id),
                word=occurrence.detected_text,
                word_spoken=occurrence.detected_text,
                count=1,
                file_count=1,
                confidence=display,
                confidence_spoken=spoken,
                why=WHY_STALE if occurrence.stale else WHY_ISOLATED,
                reviewed=reviewed,
                occurrences=(occurrence,),
                group_id=None,
            )
        )

    rows.sort(key=_weakest_first)

    if not show_other_uncertainties:
        return rows

    for recording_name, token in uncertainties:
        reviewed = _uncertainty_reviewed_text(token)
        if reviewed != NOT_REVIEWED and not show_reviewed:
            continue
        occurrence = uncertainty_occurrence(recording_name, token)
        display, spoken = confidence_range((occurrence,))
        if display == STRENGTH_NOT_MEASURED:
            # A word flagged by a rule in the pipeline often carries no
            # measured strength at all, and "Not measured" on its own tells a
            # reader nothing about how bad it is. The category the pipeline
            # did record is the answer it actually has, so it stands in.
            display = spoken = token.confidence.display_name
        rows.append(
            GroupRow(
                key=uncertainty_key(recording_name, token.id),
                word=occurrence.detected_text or token.text,
                word_spoken=occurrence.detected_text or token.text,
                count=1,
                file_count=1,
                confidence=display,
                confidence_spoken=spoken,
                why=reason_text(token),
                reviewed=reviewed,
                occurrences=(occurrence,),
                group_id=None,
            )
        )
    return rows


def _weakest_first(row: GroupRow) -> tuple[int, float, str]:
    """Sort key for the low-confidence block: weakest first, unmeasured last.

    A word nobody measured is not a weak word, so it is not allowed to sort
    above one that was measured and found wanting. The text breaks ties, so
    that two runs over the same folder give the same order.
    """
    strengths = [
        occurrence.confidence_strength
        for occurrence in row.occurrences
        if occurrence.confidence_strength is not None
    ]
    if not strengths:
        return (1, 0.0, row.word.casefold())
    return (0, min(strengths), row.word.casefold())


def spoken_group_summary(row: GroupRow) -> str:
    """The whole row in one sentence, for when the highlight moves.

    Everything a sighted person sees on the row, in the order the columns are
    in, so that hearing it and reading it come to the same thing. The count of
    files is said as well as the count of occurrences, because a change to
    this word may reach files the person has never opened and the first they
    should hear of that is here.
    """
    return (
        f"{row.word_spoken}. {occurrence_phrase(row.count, row.file_count)}. "
        f"{row.confidence_spoken}. {row.why}. {row.reviewed}."
    )


def occurrence_phrase(count: int, files: int) -> str:
    """"8 occurrences across 2 files", correct for one of either.

    The blast radius of an actual change is always taken from
    :func:`~vox_verbatim.transcription.grouping.affected_summary`, so
    that the sentence on the screen and the sentence read out cannot drift
    apart. This one describes a row rather than a change, and is only ever
    used where nothing is about to happen.
    """
    words = "1 occurrence" if count == 1 else f"{count} occurrences"
    where = "in 1 file" if files == 1 else f"across {files} files"
    return f"{words} {where}"


def spoken_occurrence_summary(
    occurrence: Occurrence,
    group_replacement: str | None,
    previous: Occurrence | None = None,
) -> str:
    """One occurrence in one sentence: where it is, how weak, and what will happen to it.

    Six facts take between three and eight seconds to read out, and the whole
    six are only worth that when the person has just arrived at the word. Once
    they are working down its occurrences, most of the six are the same on
    every row: the occurrences of one word usually share a language, share the
    word's replacement, and are usually all unreviewed. Reading those out again
    on every arrow press is the same sentence over and over with two or three
    words changed in the middle of it, and it is the reason the audio ends up
    arriving while the reader is still talking.

    So ``previous`` is the row the highlight has just come from, and anything
    identical on the two rows is left out: it has not changed, it is still in
    its column for anyone who wants to go and read it, and saying it again
    tells the listener nothing they did not hear a second ago. The file and the
    position are always said, because those are what actually move and are what
    a person navigates by.

    Called with no ``previous`` -- arriving at a word, or announcing what a
    correction did -- the whole sentence is given, because in that case nothing
    was said a second ago to leave anything out of.
    """
    replacement = occurrence.replacement or group_replacement
    parts = [occurrence.recording_name, time_spoken(occurrence.start)]
    previous_replacement = (
        (previous.replacement or group_replacement) if previous is not None else None
    )
    if previous is None or (
        previous.confidence_strength != occurrence.confidence_strength
    ):
        parts.append(strength_percentage(occurrence.confidence_strength))
    if previous is None or previous.language != occurrence.language:
        parts.append(language_name(occurrence.language))
    if previous is None or previous_replacement != replacement:
        parts.append(f"Replacement {replacement or NO_REPLACEMENT}")
    reviewed = occurrence_reviewed_text(occurrence)
    if previous is None or occurrence_reviewed_text(previous) != reviewed:
        parts.append(reviewed)
    return "".join(f"{part}. " for part in parts).strip()


def low_confidence_count(rows: list[GroupRow]) -> int:
    """How many rows came out of the low-confidence analysis rather than the pipeline."""
    return sum(1 for row in rows if not row.key.startswith(UNCERTAINTY_KEY_PREFIX))


def group_count_text(
    word_groups: int,
    uncertainty_shown: int,
    uncertainty_total: int,
    showing_uncertainties: bool,
    hidden_reviewed: int,
) -> str:
    """How much of the folder is on show, in a full sentence.

    A list that has quietly emptied itself looks exactly like a broken one, so
    the count says plainly what is being kept back and why. Three separate
    things can keep a row out of this list: it has been reviewed, the whole
    second block is switched off, or a filter is hiding it. A person who
    cannot see the controls that did it needs to be told which, and telling
    them costs one sentence.
    """
    parts: list[str] = []
    if word_groups:
        parts.append("1 word group" if word_groups == 1 else f"{word_groups} word groups")
    if not showing_uncertainties and uncertainty_total:
        parts.append(
            "1 other uncertainty, hidden"
            if uncertainty_total == 1
            else f"{uncertainty_total} other uncertainties, hidden"
        )
    elif showing_uncertainties and uncertainty_shown == uncertainty_total:
        if uncertainty_total:
            parts.append(
                "1 other uncertainty"
                if uncertainty_total == 1
                else f"{uncertainty_total} other uncertainties"
            )
    elif showing_uncertainties:
        parts.append(f"{uncertainty_shown} of {uncertainty_total} other uncertainties")
    if not parts:
        return "There is nothing to review."
    sentence = f"Showing {' and '.join(parts)}."
    if hidden_reviewed:
        already = (
            "1 reviewed word is hidden."
            if hidden_reviewed == 1
            else f"{hidden_reviewed} reviewed words are hidden."
        )
        sentence = f"{sentence} {already}"
    return sentence


# -- The models -----------------------------------------------------------


class WordGroupModel(QAbstractTableModel):
    """The words waiting for a person, one row each, spanning every file."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._rows: list[GroupRow] = []

    def set_rows(self, rows: list[GroupRow]) -> None:
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def rows(self) -> list[GroupRow]:
        return list(self._rows)

    def row_at(self, row: int) -> GroupRow | None:
        if 0 <= row < len(self._rows):
            return self._rows[row]
        return None

    def row_for_key(self, key: str | None) -> int:
        """Where a word sits now, or ``-1`` when nothing is showing it.

        The lists are rebuilt from scratch after every change, so the window
        follows the person's place by this key rather than by holding on to a
        row number that the next rebuild would silently point at a neighbour.
        """
        if not key:
            return -1
        for position, row in enumerate(self._rows):
            if row.key == key:
                return position
        return -1

    def rowCount(self, parent: QModelIndex = TOP_LEVEL) -> int:
        if parent.isValid():
            return 0
        return len(self._rows)

    def columnCount(self, parent: QModelIndex = TOP_LEVEL) -> int:
        if parent.isValid():
            return 0
        return GROUP_COLUMN_TOTAL

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
        if not 0 <= section < GROUP_COLUMN_TOTAL:
            return None
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.AccessibleTextRole):
            return GROUP_COLUMN_TITLES[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self.row_at(index.row())
        if row is None:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return self._display_text(row, index.column())
        if role == Qt.ItemDataRole.AccessibleTextRole:
            return self._spoken_text(row, index.column())
        if role == Qt.ItemDataRole.ToolTipRole:
            return spoken_group_summary(row)
        if role == Qt.ItemDataRole.UserRole:
            return row
        return None

    @staticmethod
    def _display_text(row: GroupRow, column: int) -> str:
        if column == GROUP_COLUMN_WORD:
            return row.word
        if column == GROUP_COLUMN_WHEN:
            return group_time_display(row.occurrences)
        if column == GROUP_COLUMN_COUNT:
            return str(row.count)
        if column == GROUP_COLUMN_CONFIDENCE:
            return row.confidence
        if column == GROUP_COLUMN_WHY:
            return row.why
        if column == GROUP_COLUMN_REVIEWED:
            return row.reviewed
        return ""

    @staticmethod
    def _spoken_text(row: GroupRow, column: int) -> str:
        # The count is said with its noun, because a screen reader reading a
        # cell holding "8" says "eight" and the listener has to remember which
        # column they are in to know eight of what.
        if column == GROUP_COLUMN_WHEN:
            return group_time_spoken(row.occurrences)
        if column == GROUP_COLUMN_COUNT:
            return occurrence_phrase(row.count, row.file_count)
        if column == GROUP_COLUMN_CONFIDENCE:
            return row.confidence_spoken
        if column == GROUP_COLUMN_WORD:
            return row.word_spoken
        return WordGroupModel._display_text(row, column)


class OccurrenceModel(QAbstractTableModel):
    """Where one word occurs, one row per moment in one recording."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._rows: list[Occurrence] = []
        self._group_replacement: str | None = None

    def set_occurrences(
        self,
        occurrences: tuple[Occurrence, ...],
        group_replacement: str | None = None,
    ) -> None:
        self.beginResetModel()
        self._rows = list(occurrences)
        self._group_replacement = group_replacement
        self.endResetModel()

    def occurrence_at(self, row: int) -> Occurrence | None:
        if 0 <= row < len(self._rows):
            return self._rows[row]
        return None

    def row_for_id(self, occurrence_id: str | None) -> int:
        if not occurrence_id:
            return -1
        for position, occurrence in enumerate(self._rows):
            if occurrence.id == occurrence_id:
                return position
        return -1

    @property
    def group_replacement(self) -> str | None:
        return self._group_replacement

    def rowCount(self, parent: QModelIndex = TOP_LEVEL) -> int:
        if parent.isValid():
            return 0
        return len(self._rows)

    def columnCount(self, parent: QModelIndex = TOP_LEVEL) -> int:
        if parent.isValid():
            return 0
        return OCCURRENCE_COLUMN_TOTAL

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
        if not 0 <= section < OCCURRENCE_COLUMN_TOTAL:
            return None
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.AccessibleTextRole):
            return OCCURRENCE_COLUMN_TITLES[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        occurrence = self.occurrence_at(index.row())
        if occurrence is None:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return self._display_text(occurrence, index.column())
        if role == Qt.ItemDataRole.AccessibleTextRole:
            return self._spoken_text(occurrence, index.column())
        if role == Qt.ItemDataRole.ToolTipRole:
            return spoken_occurrence_summary(occurrence, self._group_replacement)
        if role == Qt.ItemDataRole.UserRole:
            return occurrence
        return None

    def _display_text(self, occurrence: Occurrence, column: int) -> str:
        if column == OCCURRENCE_COLUMN_FILE:
            return occurrence.recording_name
        if column == OCCURRENCE_COLUMN_WHEN:
            return time_display(occurrence.start)
        if column == OCCURRENCE_COLUMN_CONFIDENCE:
            return strength_display(occurrence.confidence_strength)
        if column == OCCURRENCE_COLUMN_LANGUAGE:
            return language_name(occurrence.language)
        if column == OCCURRENCE_COLUMN_REPLACEMENT:
            return occurrence.replacement or self._group_replacement or NO_REPLACEMENT
        if column == OCCURRENCE_COLUMN_REVIEWED:
            return occurrence_reviewed_text(occurrence)
        return ""

    def _spoken_text(self, occurrence: Occurrence, column: int) -> str:
        if column == OCCURRENCE_COLUMN_WHEN:
            return time_spoken(occurrence.start)
        if column == OCCURRENCE_COLUMN_CONFIDENCE:
            return strength_percentage(occurrence.confidence_strength)
        return self._display_text(occurrence, column)


class SpeakerDoubtModel(QAbstractTableModel):
    """The stretches whose speaker is in doubt, one row each, across every file."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._rows: list[SpeakerDoubtItem] = []

    def set_items(self, items: list[SpeakerDoubtItem]) -> None:
        self.beginResetModel()
        self._rows = list(items)
        self.endResetModel()

    def items(self) -> list[SpeakerDoubtItem]:
        return list(self._rows)

    def item_at(self, row: int) -> SpeakerDoubtItem | None:
        if 0 <= row < len(self._rows):
            return self._rows[row]
        return None

    def rowCount(self, parent: QModelIndex = TOP_LEVEL) -> int:
        if parent.isValid():
            return 0
        return len(self._rows)

    def columnCount(self, parent: QModelIndex = TOP_LEVEL) -> int:
        if parent.isValid():
            return 0
        return SPEAKER_DOUBT_COLUMN_TOTAL

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
        if not 0 <= section < SPEAKER_DOUBT_COLUMN_TOTAL:
            return None
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.AccessibleTextRole):
            return SPEAKER_DOUBT_COLUMN_TITLES[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        item = self.item_at(index.row())
        if item is None:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return self._display_text(item, index.column())
        if role == Qt.ItemDataRole.AccessibleTextRole:
            return self._spoken_text(item, index.column())
        if role == Qt.ItemDataRole.ToolTipRole:
            return spoken_speaker_doubt_summary(item)
        if role == Qt.ItemDataRole.UserRole:
            return item
        return None

    @staticmethod
    def _display_text(item: SpeakerDoubtItem, column: int) -> str:
        if column == SPEAKER_DOUBT_COLUMN_FILE:
            return item.recording_name
        if column == SPEAKER_DOUBT_COLUMN_FROM:
            return time_display(item.start)
        if column == SPEAKER_DOUBT_COLUMN_TO:
            return time_display(item.end)
        if column == SPEAKER_DOUBT_COLUMN_SPEAKERS:
            return speakers_heard_text(item)
        return ""

    @staticmethod
    def _spoken_text(item: SpeakerDoubtItem, column: int) -> str:
        if column == SPEAKER_DOUBT_COLUMN_FROM:
            return time_spoken(item.start)
        if column == SPEAKER_DOUBT_COLUMN_TO:
            return time_spoken(item.end)
        return SpeakerDoubtModel._display_text(item, column)


# -- The views ------------------------------------------------------------


class ReviewTable(QTableView):
    """A table worked entirely from the keyboard, shared by both lists.

    Both lists want exactly the same behaviour, so it is written once. Tab
    leaves the table rather than walking across its cells, which keeps the way
    through the window predictable; the arrow keys still reach every cell for
    anyone who wants them.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.setWordWrap(False)
        self.setSortingEnabled(False)
        self.setTabKeyNavigation(False)
        self.verticalHeader().setVisible(False)
        header = self.horizontalHeader()
        header.setSectionsClickable(False)
        header.setHighlightSections(False)

    def selected_row(self) -> int:
        index = self.currentIndex()
        return index.row() if index.isValid() else -1

    def select_row(self, row: int) -> None:
        """Highlight a row and bring it into view."""
        model = self.model()
        if model is None or not 0 <= row < model.rowCount():
            return
        index = model.index(row, 0)
        self.setCurrentIndex(index)
        self.scrollTo(index, QAbstractItemView.ScrollHint.EnsureVisible)

    def _stretch(self, column: int) -> None:
        """Give one column the room left over, sizing the rest to their text.

        Column widths can only be set once the header knows how many sections
        it has, which is only true after a model has been attached.
        """
        header = self.horizontalHeader()
        for section in range(header.count()):
            header.setSectionResizeMode(
                section,
                QHeaderView.ResizeMode.Stretch
                if section == column
                else QHeaderView.ResizeMode.ResizeToContents,
            )


class WordGroupView(ReviewTable):
    """The first list. The reason a word is here takes the spare room."""

    def setModel(self, model) -> None:
        super().setModel(model)
        if model is not None:
            self._stretch(GROUP_COLUMN_WHY)

    def set_simple(self, simple: bool) -> None:
        """Show only the word, its time and its reason, or every column.

        The columns are hidden rather than taken out of the model, so the
        cells keep their numbers and a screen reader is told the real column
        count of what is on the screen.
        """
        for column in GROUP_DETAIL_COLUMNS:
            self.setColumnHidden(column, simple)


class OccurrenceView(ReviewTable):
    """The second list. The file name takes the spare room, being the longest."""

    def setModel(self, model) -> None:
        super().setModel(model)
        if model is not None:
            self._stretch(OCCURRENCE_COLUMN_FILE)


class SpeakerDoubtView(ReviewTable):
    """The third list. The speakers take the spare room, being the longest."""

    def setModel(self, model) -> None:
        super().setModel(model)
        if model is not None:
            self._stretch(SPEAKER_DOUBT_COLUMN_SPEAKERS)
