"""Everything a folder of recordings remembers about its own review.

A folder is a project. The state file sits in the audio folder itself,
beside the recordings rather than in the user's profile, so that it travels
with them when they are copied to another machine, moved to a bigger disk or
restored from a backup. That mirrors the transcript folders, which already
sit beside the recordings they explain, and it means a person who hands
somebody else a memory stick hands over the review as well.

**Nothing leaks between projects.** Two folders are two states with nothing
shared between them. A replacement rule created while reviewing one folder
is invisible in the other, and so are its groups, its occurrences and its
settings. This is the point of the whole design rather than an incidental
consequence of where the file lives, so it is worth saying what it protects
against: the obvious optimisation here is a cache keyed on the word, so that
having decided ``Bosh`` means ``Bosch`` once the application never has to ask
again. That cache would be wrong. The same sound is a different word in a
different recording, and a correction accepted for one client's interview has
no business rewriting another client's. Any cache added here must be keyed on
the folder, never on the word. The global vocabulary is the place for
knowledge that genuinely is universal, and it already exists.

A damaged file must never stop the Review window opening. :meth:`ProjectStore.load`
returns an empty state rather than ``None`` and never raises, whatever is in
the file. A field of the wrong type falls back to its own default, one field
at a time, exactly as :class:`~vox_verbatim.session.SessionState` does.
Cross-references between the lists are a different kind of damage, the kind a
hand edit produces, and each one is repaired on load in the way described at
:func:`ProjectState._repair`.

A failed save is not a silent event. :meth:`ProjectStore.save` returns whether
it worked and the caller is responsible for saying so out loud, for the same
reason a lost correction must not pass quietly: the person has done work, and
being told it did not stick is the only thing that lets them do it again.

Two identifiers appear throughout and they are not the same kind of thing.

``Occurrence.id`` is this layer's own identifier. It is made once, when the
occurrence is first found, and it is stable for the life of the project.
Groups, saved decisions and the "where had I got to" markers all refer to it.

``Occurrence.token_id`` is a pointer at a :class:`~vox_verbatim.transcription.model.FinalToken`
in a transcript. Those identifiers are fresh uuids every time a recording is
transcribed again, so ``token_id`` goes stale the moment a transcript is
regenerated and is never an identity. Re-matching after a re-transcription
works on the recording, the text, the time and the surrounding context, and
gives the occurrence a new ``token_id``; it never works the other way round.
Treating ``token_id`` as an identity would attach somebody's correction to a
word they never looked at, which is worse than losing the correction.

:class:`FlaggedItem` also points at a token, and its pointer goes stale in
exactly the same way, but what is done about that is the opposite and the
difference is worth stating here rather than only at the class. An occurrence
carries a decision somebody made, so it is worth a careful search to keep;
a flagged item carries nothing but what a transcript said, so it is thrown
away and read afresh whenever that transcript is read. Re-matching a flagged
item would be effort spent preserving something that costs nothing to rebuild.

Nothing here imports Qt, so the whole layer can be tested without a desktop.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vox_verbatim import DISTRIBUTION_NAME
from vox_verbatim.json_store import (
    JsonReadStatus,
    read_json_object,
    read_json_object_status,
    write_json_object,
)
from vox_verbatim.transcription.model import Confidence, Language
from vox_verbatim.transcription.normalise import normalise

_log = logging.getLogger(__name__)

#: The state file, kept in the audio folder itself. The name says which
#: application wrote it, because it sits among the user's own recordings
#: rather than in a folder of ours, and somebody who finds it there should be
#: able to tell what it belongs to before they decide whether to delete it.
PROJECT_FILE_NAME = f"{DISTRIBUTION_NAME}-project.json"

#: Bumped only if the on-disk shape changes in a way that needs migrating.
#:
#: Version 2 changed no shape. It marks the files written after the default
#: wait before automatic playback went from two seconds to none, so that a
#: file still holding the old default can be told apart from one where the
#: person chose two seconds themselves; see :meth:`ProjectState.from_dict`.
PROJECT_FORMAT_VERSION = 2

#: The last format version whose files were written under the old default
#: wait of two seconds.
_LAST_VERSION_WITH_OLD_DELAY_DEFAULT = 1

#: The default wait before automatic playback, up to format version 1.
_OLD_DEFAULT_DELAY_SECONDS = 2

#: Below this strength a word is weak enough to want a person. The figure is
#: the same as ``confidence.REVIEW_SUGGESTED_THRESHOLD``, which is where it
#: comes from, but it is written out here rather than imported: this one is a
#: project setting the user can move, and the other is the boundary of a
#: category the pipeline records in every transcript. Tying them together
#: would mean that a user nudging their own review threshold silently changed
#: what the pipeline had already written down.
DEFAULT_MINIMUM_CONFIDENCE = 0.55

#: How far two confidences may differ and still be considered the same word.
#: Five percentage points.
DEFAULT_GROUPING_TOLERANCE = 0.05

#: How long the Review window waits before playing the occurrence the person
#: has just moved to, in whole seconds.
#:
#: None, by default. A screen reader is reading the row out as the person
#: arrives, and audio that starts before it has finished talks over it. How
#: long that takes is a property of the listener rather than of the
#: application, so a person who needs a wait sets one. It used to be two
#: seconds for everybody, which made reviewing hundreds of words slow for
#: every person who did not need it.
#:
#: No wait does not mean holding the Down arrow queues a clip for every row.
#: The window still waits a moment of its own for the arrow to stop; see
#: ``SETTLE_MILLISECONDS`` in the review window.
DEFAULT_AUTO_PLAY_DELAY_SECONDS = 0

#: The longest wait the setting will accept. Half a minute is far more than
#: anybody needs and is chosen only to be an obvious mistake rather than a
#: plausible one; a value beyond it is damage rather than a preference.
MAXIMUM_AUTO_PLAY_DELAY_SECONDS = 30

#: What a language field says when the language was never worked out. Taken
#: from the model so that the one spelling is used everywhere.
_UNKNOWN_LANGUAGE = Language.UNKNOWN.value

#: What a confidence category says when none was recorded. "Unresolved" is the
#: model's own word for a value nothing could settle, which is exactly what a
#: missing category is, so a flagged item that has lost its own reads as the
#: weakest rather than as the strongest.
_UNRESOLVED_CONFIDENCE = Confidence.UNRESOLVED.value


# -- Reading loose JSON carefully ----------------------------------------
#
# The same one-line type guards that ``transcription/store.py`` uses on a
# transcript. They are written out again rather than shared because they are
# private to that module and because a project file and a transcript have
# nothing to do with each other; a rule relaxed one day to let an odd
# transcript load should not quietly change how a project file degrades.


def _text(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _prompt(value: Any) -> str:
    """A prompt kept as typed, or empty when there is no real text in it.

    Not stripped, because line breaks and indentation are part of how a
    person laid the prompt out. Blank text means no prompt of its own.
    """
    if isinstance(value, str) and value.strip():
        return value
    return ""


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value != "" else None


def _flag(value: Any, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def _integer(value: Any, default: int = 0) -> int:
    # ``bool`` is an ``int`` in Python, so ``True`` would otherwise be read
    # back as the count 1.
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _fraction(value: Any, default: float) -> float:
    """Return ``value`` as a number between nought and one, or ``default``.

    Out of range is not the same as the wrong type, and it is not treated the
    same way. A threshold of 5.0 or of -1 has a type the file is entitled to
    hold; what it does not have is any useful meaning, because it would put
    either every word or no word at all in front of the person. Pulling it to
    the nearest end of the range keeps the setting the user was reaching for
    instead of throwing it away.
    """
    number = _optional_number(value)
    if number is None:
        return default
    return min(1.0, max(0.0, number))


def _whole_seconds(value: Any, default: int, most: int) -> int:
    """Return ``value`` as a whole number of seconds within a sensible range.

    The two kinds of trouble are told apart in the same way :func:`_fraction`
    tells them apart, and for the same reason. A value of the wrong type says
    nothing at all and falls back to the default. A number out of range is
    something the person was plainly reaching for, so it is pulled to the
    nearest end rather than discarded, and a fractional number is rounded
    rather than refused: somebody who wrote 2.5 wants about two and a half
    seconds and would be baffled to be given the default instead.
    """
    number = _optional_number(value)
    if number is None:
        return default
    return min(most, max(0, round(number)))


def _language(value: Any) -> str:
    return _text(value, _UNKNOWN_LANGUAGE) or _UNKNOWN_LANGUAGE


def _whole_number_map(value: Any) -> dict[str, int]:
    """A stored mapping of names to whole numbers, keeping only what is both.

    Anything else is dropped rather than repaired, and dropping is the safe
    direction here: the only caller reads a missing entry as "nothing is known
    about that recording", which costs a read it might not have needed. Coaxing
    a number out of a value that is not one would instead let a hand-edited or
    half-written file claim knowledge it does not have, and this mapping is
    what decides that a transcript need not be looked at.
    """
    if not isinstance(value, dict):
        return {}
    found: dict[str, int] = {}
    for name, number in value.items():
        # ``bool`` is an ``int`` in Python, so ``True`` would otherwise be
        # read back as the number 1.
        if not isinstance(name, str) or isinstance(number, bool) or not isinstance(number, int):
            continue
        found[name] = number
    return found


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


# -- The pieces of a project ---------------------------------------------


@dataclass
class Occurrence:
    """One weak word in one recording, and what the person decided about it.

    ``id`` is this layer's own identifier, a uuid4 hex string made once when
    the word is first found and never changed afterwards. Everything that
    refers to an occurrence, from a group's membership to the marker saying
    where the person had got to, refers to this.

    ``token_id`` is a pointer at the :class:`~vox_verbatim.transcription.model.FinalToken`
    this occurrence currently stands for. It is **not** an identity and must
    never be used as one. Token identifiers are regenerated as fresh uuids
    every time a recording is transcribed again, so this field goes stale as
    soon as a transcript is redone and is rewritten by the re-matching step.
    An occurrence that could not be re-matched is marked ``stale`` and left
    alone, because attaching a saved correction to whichever word happened to
    be nearest would apply somebody's decision to a word they never saw.
    """

    id: str
    recording_name: str
    """The audio file name, such as ``Interview 01.m4a``."""

    token_id: str
    """The ``FinalToken`` identifier this occurrence points at just now."""

    detected_text: str
    """What the services actually produced, kept unchanged.

    Decision 7 of the specification rewrites each occurrence from what it
    originally said rather than from what it currently says, so this field is
    the one a replacement is applied to and it must not be overwritten with a
    correction.
    """

    normalised_text: str
    start: float | None
    end: float | None
    confidence_strength: float | None
    """The application's own combined estimate, nought to one, or ``None``
    where it was never worked out."""

    language: str
    """A :class:`~vox_verbatim.transcription.model.Language` value, or
    ``"unknown"`` where the language was never established."""

    context_before: str
    """A few words each side, kept for re-matching after a re-transcription."""

    context_after: str
    reviewed: bool = False
    correct_as_detected: bool = False
    replacement: str | None = None
    """A replacement for this one occurrence, which overrides its group's."""

    isolated: bool = False
    """The person took this word out of its group and wants it left out."""

    stale: bool = False
    """It could not be re-matched after the recording was transcribed again."""

    auto_applied: bool = False
    """A project rule answered this occurrence; nobody looked at it.

    A file transcribed next week is answered by what the project already
    knows, so a word whose form somebody has already accepted a replacement
    for is corrected without asking again. The flag records that this is what
    happened, so the window can show the difference between a decision a
    person made and one the project made on their behalf.
    """

    applied_rule_id: str | None = None
    """Which rule answered this occurrence, so it can be found again.

    ``auto_applied`` says that a rule fired; this says which one. The two are
    kept apart because finding the rule again by matching its text and its
    language is a lookup rather than a record, and it gives the wrong answer
    as soon as two rules overlap through the unknown-language fallback in
    :meth:`ProjectState.rule_for`. A person undoing an automatic correction
    has to reach the exact rule that made it, both to put that rule right and
    to keep its ``occurrence_count`` honest, and a lookup that is usually
    right is not good enough for something that rewrites words quietly.
    """

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> Occurrence | None:
        """Rebuild an occurrence, or return ``None`` if it cannot be trusted.

        Only the identifier is required. Inventing one for an occurrence that
        has lost its own would be worse than dropping it: no group, no marker
        and no saved decision refers to an identifier this module made up a
        moment ago, so the occurrence would come back as a loose word with no
        history, looking like a fresh finding rather than like damage. The
        analysis finds the word again on the next run in any case.

        The comparison form is worked out again from ``detected_text``, as
        :meth:`ReplacementRule.from_dict` does for a rule. The review window
        finds the rule an occurrence taught by comparing the two forms, so if
        only the rule's form were brought up to date, a project saved before
        the comparison forms changed would load with the two disagreeing, and
        reversing a correction would leave its rule behind to go on applying.
        """
        if not isinstance(data, dict):
            return None
        identifier = _optional_text(data.get("id"))
        if identifier is None:
            _log.warning("An occurrence with no identifier was ignored.")
            return None
        return cls(
            id=identifier,
            recording_name=_text(data.get("recording_name")),
            token_id=_text(data.get("token_id")),
            detected_text=_text(data.get("detected_text")),
            normalised_text=(
                normalise(_text(data.get("detected_text"))) or _text(data.get("normalised_text"))
            ),
            start=_optional_number(data.get("start")),
            end=_optional_number(data.get("end")),
            confidence_strength=_optional_number(data.get("confidence_strength")),
            language=_language(data.get("language")),
            context_before=_text(data.get("context_before")),
            context_after=_text(data.get("context_after")),
            reviewed=_flag(data.get("reviewed")),
            correct_as_detected=_flag(data.get("correct_as_detected")),
            replacement=_optional_text(data.get("replacement")),
            isolated=_flag(data.get("isolated")),
            stale=_flag(data.get("stale")),
            auto_applied=_flag(data.get("auto_applied")),
            applied_rule_id=_optional_text(data.get("applied_rule_id")),
        )


@dataclass
class FlaggedItem:
    """One word some rule in the pipeline flagged, remembered rather than re-found.

    These are the second block of the Review window's first list: the words
    that carry a stated reason for needing a person, as opposed to the words
    the low-confidence sweep found by their number alone. They are a property
    of a transcript rather than a decision anybody made, so at first sight the
    obvious thing is to read the transcripts and pick them out when the window
    opens. **That is the expensive mistake, and it has a measurement against
    it.** A folder of fifty hour-long recordings took 37 seconds to read and
    held about 1.6 GB of live objects; the same folder opens in 0.04 seconds
    when the project file is treated as the index and the transcripts are read
    only on demand. The low-confidence occurrences already live here for that
    reason and these belong here for exactly the same one.

    What is kept is what the two lists need to draw a row without opening
    anything: which recording it is in, which word of that recording, what it
    says, where it sits, why it was flagged, how confident the pipeline was,
    and whether somebody has since finished with it. Everything else about the
    word -- the services' candidates, the timing evidence, the language
    reasoning -- comes out of the transcript when the person selects it, which
    is one read of one file at the moment it is actually wanted.

    ``token_id`` is a pointer, in the same sense and with the same dangers as
    :attr:`Occurrence.token_id`, and the module docstring says why it is
    treated differently. It is never re-matched. A transcript that has been
    regenerated is a transcript that has been rewritten, so it is read again by
    whoever analyses the folder next, and the flagged items of that recording
    are replaced wholesale by what it now says. An entry pointing at a word
    that has gone therefore cannot survive the next look at its recording, and
    until then the window checks the pointer against the transcript it has to
    load anyway before it lets the person do anything to the word.
    """

    recording_name: str
    """The audio file name, such as ``Interview 01.m4a``."""

    token_id: str
    """The ``FinalToken`` this item stood for when its recording was last read."""

    text: str
    """The word as it was detected, which is what the first list shows."""

    start: float | None = None
    reasons: list[str] = field(default_factory=list)
    """:class:`~vox_verbatim.transcription.model.ReviewReason` values.

    Kept as plain strings, as ``Occurrence.language`` is, so that a reason this
    version has never heard of survives being written by a newer one and is
    simply not recognised when the window comes to name it.
    """

    confidence: str = _UNRESOLVED_CONFIDENCE
    """A :class:`~vox_verbatim.transcription.model.Confidence` value.

    A flagged word very often has no measured strength at all, because it was
    flagged by a rule rather than by assessment, and "not measured" tells a
    reader nothing about how bad it is. The category is the answer the pipeline
    actually had, so it is what the confidence column falls back to.
    """

    risk_categories: list[str] = field(default_factory=list)
    """:class:`~vox_verbatim.transcription.model.RiskCategory` values.

    Kept so that the window can say what kind of value a flagged word is,
    such as a date, without opening its transcript. Plain strings for the
    reason :attr:`reasons` is.
    """

    settled: bool = False
    """Somebody has finished with this word: they corrected it or confirmed it.

    Which of the two is deliberately not kept. It is written into the
    transcript, which is where a claim about what a person did belongs, and the
    window says only "Reviewed" until that transcript is open to say more. The
    flag is here so that a settled word can be left out of the list without the
    transcript being read to discover that it should be.
    """

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> FlaggedItem | None:
        """Rebuild a flagged item, or return ``None`` if it points at nothing.

        The recording and the token are both required, and neither can be
        invented. Together they are how the window keys the row, how it finds
        the word again in the transcript and how it tells this item apart from
        an occurrence describing the same word, so an item missing either is
        not a damaged row but no row at all. Nothing is lost by dropping it:
        unlike an occurrence, it holds no decision, and the next analysis of
        that recording finds the word again.

        Everything else degrades to its own default. A word with no text is
        not damage at all and must not be dropped: the services sometimes
        disagree so completely that nothing was chosen, and that word is
        precisely one somebody needs to look at.
        """
        if not isinstance(data, dict):
            return None
        recording_name = _optional_text(data.get("recording_name"))
        token_id = _optional_text(data.get("token_id"))
        if recording_name is None or token_id is None:
            _log.warning("A flagged word with no recording or no word to point at was ignored.")
            return None
        return cls(
            recording_name=recording_name,
            token_id=token_id,
            text=_text(data.get("text")),
            start=_optional_number(data.get("start")),
            reasons=_string_list(data.get("reasons")),
            confidence=_text(data.get("confidence"), _UNRESOLVED_CONFIDENCE)
            or _UNRESOLVED_CONFIDENCE,
            risk_categories=_string_list(data.get("risk_categories")),
            settled=_flag(data.get("settled")),
        )


@dataclass
class SpeakerDoubtItem:
    """One stretch of speech whose speaker is in doubt, remembered for the window.

    A word whose text every service agrees on is not a word to review just
    because the services heard it from different people, so these are kept
    apart from :class:`FlaggedItem`. One item stands for a whole stretch of
    consecutive words, which is how the doubt arises: the second opinion
    disagrees about a passage, and listing it once for every word in it
    filled the word list with hundreds of rows that all said the same thing.

    It is saved for the reason flagged items are: the window lists these on a
    folder whose transcripts nobody has opened. They are replaced wholesale
    whenever their recording is read again, exactly as flagged items are.
    """

    recording_name: str
    start: float | None = None
    end: float | None = None
    speakers: list[str] = field(default_factory=list)
    """The speaker the words were given, then the other one a service heard.

    Shorter than two where no other speaker could be named.
    """

    token_ids: list[str] = field(default_factory=list)
    """The words of the stretch, as they were when the recording was read."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> SpeakerDoubtItem | None:
        """Rebuild an item, or return ``None`` if it points at nothing.

        A stretch with no recording or no words in it cannot be played or
        found again, and the next analysis of that recording finds it afresh,
        so it is dropped rather than shown as a row that does nothing.
        """
        if not isinstance(data, dict):
            return None
        recording_name = _optional_text(data.get("recording_name"))
        token_ids = [item for item in _string_list(data.get("token_ids")) if item]
        if recording_name is None or not token_ids:
            _log.warning("A speaker doubt with no recording or no words was ignored.")
            return None
        return cls(
            recording_name=recording_name,
            start=_optional_number(data.get("start")),
            end=_optional_number(data.get("end")),
            speakers=[item for item in _string_list(data.get("speakers")) if item],
            token_ids=token_ids,
        )


@dataclass
class WordGroup:
    """Several occurrences that probably mean the same intended word."""

    id: str
    representative_text: str
    """The form shown for the group: the most common detected form, and the
    lowest-confidence one where there is a tie."""

    occurrence_ids: list[str]
    """The ``Occurrence.id`` of every member, never a ``token_id``."""

    replacement: str | None = None
    reviewed: bool = False
    correct_as_detected: bool = False
    user_created: bool = False
    """The person made this group by hand, so reprocessing keeps its members
    rather than rebuilding it."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> WordGroup | None:
        """Rebuild a group, or return ``None`` if it has no identifier.

        As with an occurrence, a group that has lost its identifier cannot be
        repaired by giving it a new one, because the identifier is how the
        window and the saved marker refer to it.
        """
        if not isinstance(data, dict):
            return None
        identifier = _optional_text(data.get("id"))
        if identifier is None:
            _log.warning("A word group with no identifier was ignored.")
            return None
        return cls(
            id=identifier,
            representative_text=_text(data.get("representative_text")),
            occurrence_ids=_string_list(data.get("occurrence_ids")),
            replacement=_optional_text(data.get("replacement")),
            reviewed=_flag(data.get("reviewed")),
            correct_as_detected=_flag(data.get("correct_as_detected")),
            user_created=_flag(data.get("user_created")),
        )


@dataclass
class ReplacementRule:
    """One detected form this project has been told what to do with.

    Rules are made one per distinct detected form, not one per group. A group
    holding ``Bosch``, ``Bosh`` and ``Bosche`` produces three rules when its
    replacement is accepted. That is deliberate and it is what makes the rules
    useful later: a file transcribed next week which says ``Bosh`` is answered
    by a person who actually accepted a correction for ``Bosh``. A form nobody
    has ever accepted a replacement for is never rewritten silently; it goes
    into the review queue instead, which is the only honest thing to do with a
    spelling the person has not seen.

    ``group_id`` records which group the rule came from, so that changing that
    group's replacement can find its rules again and update them. Without it,
    editing a decision would leave the old rules behind to keep applying the
    replacement the person had just changed their mind about.
    """

    id: str
    matched_text: str
    """The detected form this rule answers, spelled as the services spelled it."""

    normalised_text: str
    replacement: str
    language: str
    created_at: str
    """ISO 8601."""

    occurrence_count: int = 0
    """How many occurrences this rule has answered so far."""

    group_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> ReplacementRule | None:
        """Rebuild a rule, or return ``None`` if it could do nothing.

        A rule with no identifier is given a fresh one rather than being
        dropped, which is the opposite of what happens to an occurrence or a
        group. The asymmetry is deliberate: nothing refers to a rule by its
        identifier, so a new one costs nothing, while dropping the rule would
        throw away a correction the person accepted. What cannot be repaired
        is a rule with nothing to match or nothing to put in its place, since
        that rule could never do anything but confuse whoever read the list.

        The comparison form is worked out again from ``matched_text`` rather
        than read from the file. The stored form is whatever the code said
        when the rule was saved, and :meth:`ProjectState.rule_for` matches
        against it exactly, so once the comparison forms change -- as they
        did when "ja" came to compare as "yeah" -- every older rule would
        stop matching and nothing would say so. The stored form is kept only
        for text that normalises to nothing, where the window saved a
        case-folded form of its own.
        """
        if not isinstance(data, dict):
            return None
        matched = _optional_text(data.get("matched_text"))
        replacement = _optional_text(data.get("replacement"))
        if matched is None or replacement is None:
            _log.warning("A replacement rule with no text to match or apply was ignored.")
            return None
        identifier = _optional_text(data.get("id"))
        if identifier is None:
            identifier = _new_id()
            _log.warning("A replacement rule for %r had no identifier; giving it one.", matched)
        return cls(
            id=identifier,
            matched_text=matched,
            normalised_text=(
                normalise(matched) or _text(data.get("normalised_text")) or matched.casefold()
            ),
            replacement=replacement,
            language=_language(data.get("language")),
            created_at=_text(data.get("created_at")),
            occurrence_count=_integer(data.get("occurrence_count")),
            group_id=_optional_text(data.get("group_id")),
        )


@dataclass
class LearnedName:
    """A name this folder's reviews taught, and the ways the services got it wrong.

    A person correcting ``Bosh`` to ``Bosch`` while listening to their own
    recording has said, with the best evidence there is, that ``Bosch`` is a
    word spoken in this folder. It is kept here, in the folder's own project
    file, rather than in the shared vocabulary, for the reason the module
    docstring gives: one client's surname has no business being suggested to
    the services when another client's recordings are transcribed.

    ``wrong_forms`` are the spellings the services wrote instead. The correct
    text never appears among them.
    """

    text: str
    language: str
    """A :class:`~vox_verbatim.transcription.model.Language` value, or
    ``"unknown"`` where the language was never established."""

    wrong_forms: list[str] = field(default_factory=list)
    learned_at: str = ""
    """When the name was first learned, ISO 8601."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> LearnedName | None:
        """Rebuild a learned name, or return ``None`` if it has no text.

        A name with no text could teach nothing, so it is dropped. Everything
        else falls back to its own default.
        """
        if not isinstance(data, dict):
            return None
        text = _text(data.get("text")).strip()
        if not text:
            _log.warning("A learned name with no text was ignored.")
            return None
        return cls(
            text=text,
            language=_language(data.get("language")),
            wrong_forms=_string_list(data.get("wrong_forms")),
            learned_at=_text(data.get("learned_at")),
        )

    @property
    def key(self) -> tuple[str, str]:
        """What two entries are compared on to decide they are one name.

        Capitals and surrounding space are set aside, because ``bosch`` and
        ``Bosch`` are one name written twice. Casefolding leaves accented
        letters alone, so ``Müller`` and ``Mueller`` stay two names. The
        language is part of the key, so the same spelling learned in two
        languages is two entries.
        """
        return (self.text.strip().casefold(), self.language)


def merge_learned_names(existing: list[LearnedName], new: list[LearnedName]) -> list[LearnedName]:
    """Return ``existing`` with ``new`` added, never listing one name twice.

    A name already listed keeps its place, its spelling and the time it was
    first learned, and gains any wrong forms it did not have. Wrong forms are
    compared without regard to capitals, and the correct text itself is never
    kept as a wrong form. Neither argument is changed: the entries returned
    are new objects. The order is the order names were first learned in.
    """
    merged: list[LearnedName] = []
    positions: dict[tuple[str, str], int] = {}
    for name in [*existing, *new]:
        text = name.text.strip()
        if not text:
            continue
        position = positions.get(name.key)
        if position is None:
            position = len(merged)
            positions[name.key] = position
            merged.append(LearnedName(text=text, language=name.language,
                                      learned_at=name.learned_at))
        target = merged[position]
        if not target.learned_at and name.learned_at:
            target.learned_at = name.learned_at
        known = {form.casefold() for form in target.wrong_forms}
        known.add(target.text.casefold())
        for form in name.wrong_forms:
            form = form.strip()
            if form and form.casefold() not in known:
                target.wrong_forms.append(form)
                known.add(form.casefold())
    return merged


@dataclass
class ProjectSettings:
    """The choices the person made about how this folder is reviewed."""

    minimum_confidence: float = DEFAULT_MINIMUM_CONFIDENCE
    grouping_tolerance: float = DEFAULT_GROUPING_TOLERANCE
    show_other_uncertainties: bool = True
    show_reviewed: bool = False
    play_automatically: bool = True

    show_details: bool = False
    """Whether the review window shows everything, or only what a decision needs.

    Off by default, so a folder opens on the simple window: the words, the
    choices for the one selected, and playback. Kept with the folder like
    the other review toggles, so a person who wants the details sees them
    every time they come back to it.
    """

    auto_play_delay_seconds: int = DEFAULT_AUTO_PLAY_DELAY_SECONDS
    """How long to wait before playing the occurrence just moved to.

    Whole seconds, because this is a number a person sets for themselves and
    a spin box counting in milliseconds asks them to think in units nobody
    thinks in. Nought means play at once, which is what somebody who has
    turned their screen reader's speech off will want.
    """

    smoothing_prompt: str = ""
    """This folder's own style prompt for the smooth transcript, or empty.

    Empty means the folder uses the app's prompt from Settings. It is kept
    with the folder, because the right style belongs to the recordings: a
    folder of family interviews in Afrikaans wants a different prompt from
    the rest, and the folder carries it when it is handed to someone else.
    """

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> ProjectSettings:
        """Rebuild the settings, falling back field by field.

        This never returns ``None``. There is always a usable set of settings,
        because refusing to open a folder over a mistyped number would be an
        absurd thing to do to somebody who only wants to carry on reviewing.
        """
        if not isinstance(data, dict):
            return cls()
        return cls(
            minimum_confidence=_fraction(
                data.get("minimum_confidence"), DEFAULT_MINIMUM_CONFIDENCE
            ),
            grouping_tolerance=_fraction(
                data.get("grouping_tolerance"), DEFAULT_GROUPING_TOLERANCE
            ),
            show_other_uncertainties=_flag(data.get("show_other_uncertainties"), True),
            show_reviewed=_flag(data.get("show_reviewed"), False),
            play_automatically=_flag(data.get("play_automatically"), True),
            show_details=_flag(data.get("show_details"), False),
            auto_play_delay_seconds=_whole_seconds(
                data.get("auto_play_delay_seconds"),
                DEFAULT_AUTO_PLAY_DELAY_SECONDS,
                MAXIMUM_AUTO_PLAY_DELAY_SECONDS,
            ),
            smoothing_prompt=_prompt(data.get("smoothing_prompt")),
        )


@dataclass
class ProjectState:
    """Everything one folder remembers, and the small lookups over it.

    The lookups are methods rather than something every caller works out for
    itself, so that "which group is this word in" has one answer in one place
    instead of four windows each rummaging through the lists their own way.
    """

    settings: ProjectSettings = field(default_factory=ProjectSettings)
    groups: list[WordGroup] = field(default_factory=list)
    occurrences: list[Occurrence] = field(default_factory=list)
    """Every occurrence in the project, whether it is in a group or not."""

    flagged: list[FlaggedItem] = field(default_factory=list)
    """Every word the pipeline flagged, in every recording that has been read.

    This is the index the second block of the Review window's first list is
    drawn from, and it is why that block is complete on a folder whose
    transcripts nobody has opened. :class:`FlaggedItem` says what it costs to
    find these by reading instead.

    A recording appears here once it has been analysed and its entries are
    then replaced in full every time it is analysed again, so an empty list
    for a recording that has been read means that recording has no flagged
    words rather than that nobody has looked.
    """

    speaker_doubts: list[SpeakerDoubtItem] = field(default_factory=list)
    """Every stretch whose speaker is in doubt, in every recording that has been read.

    Kept and replaced per recording exactly as :attr:`flagged` is.
    """

    rules: list[ReplacementRule] = field(default_factory=list)
    processed_at: str = ""
    """When the analysis last ran, ISO 8601.

    This says when, for anyone who wants to know when, and it is what tells an
    unreviewed folder apart from a reviewed one. It is deliberately not what
    decides whether a transcript has to be read again; ``transcript_times``
    is, and the difference between the two is written up there.
    """

    transcript_times: dict[str, int] = field(default_factory=dict)
    """What each recording's transcript file said its own age was when it was analysed.

    Keyed on the recording's file name with its extension, as everything else
    about a recording is, and holding one whole number built from the
    transcript's modification time, size and file ID. The main window makes
    the number and says why the time alone is not enough; the name of this
    field is older than that, and is kept so that existing project files still
    load. The number is never interpreted here and never compared with a wall
    clock. It is only ever compared with the same file's number later on, and
    the single question asked of it is whether the two are the same.

    That is what makes it trustworthy where a time of day is not. A stamp
    saying when the analysis ran comes from the system clock, and the time a
    file carries comes from the filesystem, and on Windows those two do not
    agree: a transcript written a moment *after* an analysis finished can
    honestly report an age a moment *before* it, by up to about ten
    milliseconds, so asking "is the file newer than the analysis" answers no
    about a file that has genuinely changed. Asking instead whether the file
    is still the one that was read has no clock in it at all.

    A recording with no entry here has never been analysed, or was analysed
    and could not be read, and either way is read again. An entry is dropped
    when its recording leaves the folder, so this never grows into a record of
    files that are gone.
    """

    last_group_id: str | None = None
    """The group the person was last on, so reopening puts them back there."""

    last_occurrence_id: str | None = None
    """The occurrence the person was last on."""

    learned_names: list[LearnedName] = field(default_factory=list)
    """The names this folder's reviews taught, each listed once.

    Saved through :meth:`ProjectStore.save_keeping_learned_names` rather than
    plain :meth:`ProjectStore.save` wherever the file may have changed since
    it was loaded, so that a name somebody removed is not written back.
    """

    # -- The lookups the window needs
    #
    # These walk the lists rather than keeping an index beside them, and that
    # is on purpose. The window adds, removes and regroups occurrences while
    # the person works, so an index would have to be rebuilt on every change
    # or it would go stale. A stale index here does not merely give a slow
    # answer, it gives the wrong word, which is the one failure this whole
    # layer exists to prevent. The lists hold the weak words of one folder,
    # not every word of it, so walking them costs nothing worth having.

    def occurrence(self, occurrence_id: str) -> Occurrence | None:
        """The occurrence with this identifier, or ``None`` if it is gone."""
        for item in self.occurrences:
            if item.id == occurrence_id:
                return item
        return None

    def group(self, group_id: str) -> WordGroup | None:
        """The group with this identifier, or ``None`` if it is gone."""
        for item in self.groups:
            if item.id == group_id:
                return item
        return None

    def group_of(self, occurrence_id: str) -> WordGroup | None:
        """The group this occurrence belongs to, or ``None`` if it is loose."""
        for item in self.groups:
            if occurrence_id in item.occurrence_ids:
                return item
        return None

    def occurrences_of(self, group_id: str) -> list[Occurrence]:
        """The members of this group, in the order the group lists them.

        A membership naming an occurrence that is not there is skipped rather
        than reported, because :meth:`_repair` has already removed those on
        load and anything appearing here afterwards was made by a caller that
        will see the shortfall in what it gets back.
        """
        found = self.group(group_id)
        if found is None:
            return []
        members = [self.occurrence(item) for item in found.occurrence_ids]
        return [item for item in members if item is not None]

    def loose_occurrences(self) -> list[Occurrence]:
        """Every occurrence that no group holds, in project order.

        Both kinds of loose word end up here: one the person deliberately took
        out of a group, which carries ``isolated``, and one the analysis never
        found company for. The flag is the person's intention and is what stops
        reprocessing gathering the word up again; belonging to no group is the
        present fact, and it is the fact the window has to draw.
        """
        claimed = {
            occurrence_id for group in self.groups for occurrence_id in group.occurrence_ids
        }
        return [item for item in self.occurrences if item.id not in claimed]

    def rule_for(self, normalised_text: str, language: str) -> ReplacementRule | None:
        """The rule that answers this word, or ``None`` if there is not one.

        The text must match exactly once normalised; nothing here guesses.
        The language is more forgiving, and matches when the two agree or when
        either side never established a language at all. That is the same
        principle the grouping engine works to: an unknown language is missing
        information, not a difference, and letting it block a rule would mean a
        correction the person accepted stopped working on the recordings where
        the language could not be detected, which is precisely where the words
        are hardest and the help is most wanted. A rule in the same language is
        still preferred over one where either side is unknown, so a deliberate
        German rule is never passed over for a vaguer one.
        """
        fallback: ReplacementRule | None = None
        for rule in self.rules:
            if rule.normalised_text != normalised_text:
                continue
            if rule.language == language:
                return rule
            unknown_on_either_side = (
                rule.language == _UNKNOWN_LANGUAGE or language == _UNKNOWN_LANGUAGE
            )
            if fallback is None and unknown_on_either_side:
                fallback = rule
        return fallback

    # -- On the disk

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": PROJECT_FORMAT_VERSION,
            "settings": self.settings.to_dict(),
            "groups": [item.to_dict() for item in self.groups],
            "occurrences": [item.to_dict() for item in self.occurrences],
            "flagged": [item.to_dict() for item in self.flagged],
            "speaker_doubts": [item.to_dict() for item in self.speaker_doubts],
            "rules": [item.to_dict() for item in self.rules],
            "processed_at": self.processed_at,
            "transcript_times": dict(self.transcript_times),
            "last_group_id": self.last_group_id,
            "last_occurrence_id": self.last_occurrence_id,
            "learned_names": [item.to_dict() for item in self.learned_names],
        }

    @classmethod
    def from_dict(cls, data: Any) -> ProjectState:
        """Rebuild a project from loaded JSON, repairing what does not add up.

        This never returns ``None`` and never raises. Every field falls back
        to its own default one at a time, and the cross-references between the
        lists are then put right by :meth:`_repair`.
        """
        if not isinstance(data, dict):
            return cls()

        version = _integer(data.get("version"), PROJECT_FORMAT_VERSION)
        if version > PROJECT_FORMAT_VERSION:
            # Reading it anyway is the right call. The fields this version
            # knows about are still there, and refusing would leave somebody
            # who opened the folder on a newer machine yesterday unable to
            # review it on this one today.
            _log.warning(
                "This project file was written by a newer version of the application "
                "(format %s); reading what can be understood.",
                version,
            )

        state = cls(settings=ProjectSettings.from_dict(data.get("settings")))
        if (
            version <= _LAST_VERSION_WITH_OLD_DELAY_DEFAULT
            and state.settings.auto_play_delay_seconds == _OLD_DEFAULT_DELAY_SECONDS
        ):
            # Two seconds in a file this old is the old default rather than a
            # choice, as far as can be told, and keeping it would leave every
            # folder reviewed before the change as slow as it was. Any other
            # value was chosen by the person and stays. The file is written
            # back at the current version, so this happens only once.
            state.settings.auto_play_delay_seconds = DEFAULT_AUTO_PLAY_DELAY_SECONDS
        state.occurrences = _unique_by_id(
            _each(data.get("occurrences"), Occurrence.from_dict), "occurrence"
        )
        state.groups = _unique_by_id(_each(data.get("groups"), WordGroup.from_dict), "word group")
        state.flagged = _each(data.get("flagged"), FlaggedItem.from_dict)
        state.speaker_doubts = _each(data.get("speaker_doubts"), SpeakerDoubtItem.from_dict)
        state.rules = _each(data.get("rules"), ReplacementRule.from_dict)
        state.processed_at = _text(data.get("processed_at"))
        state.transcript_times = _whole_number_map(data.get("transcript_times"))
        state.last_group_id = _optional_text(data.get("last_group_id"))
        state.last_occurrence_id = _optional_text(data.get("last_occurrence_id"))
        state.learned_names = _learned_names(data)
        state._repair()
        return state

    def _repair(self) -> None:
        """Put right the things a hand-edited file gets wrong.

        Four kinds of damage are possible once the fields themselves have
        been read, and each is repaired quietly, predictably and the same way
        every time, so that opening the same damaged file twice gives the same
        project twice.

        A group naming an occurrence that is not there loses that name. The
        group is what is left of it, which is the most of the person's work
        that can be kept. A group left holding nothing is dropped altogether,
        even one the person made by hand: a group is no more than its members,
        and an empty one would be a row in the window that cannot be opened,
        played or corrected, with a replacement waiting to be applied to
        nothing.

        An occurrence claimed by two groups stays in the first group that
        lists it and is taken out of the others. First is chosen because it is
        the only rule that gives the same answer every time; picking the
        better-fitting group would be a guess dressed up as a repair. Leaving
        the word in both would be worse than either, because decision 7
        rewrites each occurrence from what it originally said, so the word
        would be rewritten twice and which replacement survived would depend
        on the order the groups happened to be applied in, differ between two
        runs over the same file, and be invisible to everybody.

        A pointer at something that is no longer there is cleared, and these
        are not damage at all, which is why they are done without a warning
        while the two above are logged. There are two of them. A marker saying
        where the person had got to goes out of date whenever reprocessing
        rebuilds the automatic groups, which happens between one ordinary
        session and the next, and the person simply starts at the beginning.
        An occurrence naming the rule that corrected it loses that name when
        the rule has since been deleted, which is an ordinary thing for a
        person to do to a rule they no longer want. ``auto_applied`` is left
        standing in that case, because the correction did happen and saying
        otherwise would be a lie about the word; what is gone is only the way
        back to the rule behind it.

        Two flagged items naming the same word of the same recording are the
        last kind, and only the first is kept. The pair is what the window
        keys the row by, so the second could never be selected, played or
        settled; it would show as a row that does nothing, and it would be
        counted, so the sentence saying how much is on show would overstate
        the work waiting by one.
        """
        known = {item.id for item in self.occurrences}
        claimed: set[str] = set()
        kept: list[WordGroup] = []
        for group in self.groups:
            members: list[str] = []
            for occurrence_id in group.occurrence_ids:
                if occurrence_id not in known:
                    _log.warning(
                        "The word group %r names an occurrence that is not in this project "
                        "(%s); leaving it out.",
                        group.representative_text,
                        occurrence_id,
                    )
                    continue
                if occurrence_id in claimed:
                    _log.warning(
                        "The occurrence %s is claimed by more than one word group; leaving it "
                        "in the first and out of %r.",
                        occurrence_id,
                        group.representative_text,
                    )
                    continue
                members.append(occurrence_id)
                claimed.add(occurrence_id)
            if not members:
                _log.warning(
                    "The word group %r has no occurrences left in it; dropping it.",
                    group.representative_text,
                )
                continue
            group.occurrence_ids = members
            kept.append(group)
        self.groups = kept

        if self.last_group_id is not None and self.group(self.last_group_id) is None:
            self.last_group_id = None
        if self.last_occurrence_id is not None and self.occurrence(self.last_occurrence_id) is None:
            self.last_occurrence_id = None

        rule_ids = {rule.id for rule in self.rules}
        for occurrence in self.occurrences:
            applied = occurrence.applied_rule_id
            if applied is not None and applied not in rule_ids:
                occurrence.applied_rule_id = None

        seen_words: set[tuple[str, str]] = set()
        kept_flagged: list[FlaggedItem] = []
        for item in self.flagged:
            where = (item.recording_name, item.token_id)
            if where in seen_words:
                _log.warning(
                    "The word %s of %s is flagged twice; keeping the first.",
                    item.token_id,
                    item.recording_name,
                )
                continue
            seen_words.add(where)
            kept_flagged.append(item)
        self.flagged = kept_flagged

        # The same stretch listed twice is the same fault as a word flagged
        # twice: a row that duplicates another and overstates the work left.
        seen_stretches: set[tuple[str, tuple[str, ...]]] = set()
        kept_doubts: list[SpeakerDoubtItem] = []
        for doubt in self.speaker_doubts:
            stretch = (doubt.recording_name, tuple(doubt.token_ids))
            if stretch in seen_stretches:
                _log.warning(
                    "A speaker doubt in %s is listed twice; keeping the first.",
                    doubt.recording_name,
                )
                continue
            seen_stretches.add(stretch)
            kept_doubts.append(doubt)
        self.speaker_doubts = kept_doubts


def _each(value: Any, build: Any) -> list[Any]:
    """Build every item of a stored list, dropping the ones that cannot be."""
    if not isinstance(value, list):
        return []
    found = [build(item) for item in value]
    return [item for item in found if item is not None]


def _unique_by_id(items: list[Any], description: str) -> list[Any]:
    """Keep the first of anything sharing an identifier with something earlier.

    Two occurrences with one identifier is damage rather than a fact anything
    could act on, because every lookup by that identifier would answer with
    whichever of them the search reached first. Keeping the first and dropping
    the rest at least makes that answer the same one every time.
    """
    seen: set[str] = set()
    kept: list[Any] = []
    for item in items:
        if item.id in seen:
            _log.warning("Two entries share the %s identifier %s; keeping the first.",
                         description, item.id)
            continue
        seen.add(item.id)
        kept.append(item)
    return kept


def _learned_names(data: dict[str, Any]) -> list[LearnedName]:
    """The learned names in a loaded project file, each listed once."""
    return merge_learned_names([], _each(data.get("learned_names"), LearnedName.from_dict))


def read_learned_names(folder: Path | str) -> tuple[list[LearnedName], JsonReadStatus]:
    """Read only the learned names from a folder's project file, as it is now.

    The status says whether the file was read, is not there, or is there and
    could not be read. The list is empty unless it was read. Nothing else in
    the file is built, so this is cheap enough to call before every save.
    """
    data, status = read_json_object_status(Path(folder) / PROJECT_FILE_NAME)
    if data is None:
        return [], status
    return _learned_names(data), status


def remove_learned_name(folder: Path | str, text: str, language: str) -> bool:
    """Take one learned name out of a folder's project file on disk.

    Returns whether a name was removed and the file was written. The file is
    read and written back as raw JSON, so every other key in it is kept
    exactly as it was, including any this version does not know about. A file
    that is missing, locked or damaged is not written at all, because writing
    it would replace somebody's whole project with one list.
    """
    path = Path(folder) / PROJECT_FILE_NAME
    data, status = read_json_object_status(path)
    if data is None or status is not JsonReadStatus.READ:
        return False
    entries = data.get("learned_names")
    if not isinstance(entries, list):
        return False
    wanted = LearnedName(text=text, language=_language(language)).key
    kept = []
    for entry in entries:
        name = LearnedName.from_dict(entry)
        if name is not None and name.key == wanted:
            continue
        kept.append(entry)
    if len(kept) == len(entries):
        return False
    data["learned_names"] = kept
    return write_json_object(path, data)


def _new_id() -> str:
    """A fresh identifier for this layer's own use."""
    return uuid.uuid4().hex


class ProjectStore:
    """The project file that belongs to one folder of recordings.

    Making a store creates nothing. A folder that has never been reviewed
    costs nothing to ask about and is left exactly as it was found, which
    matters because the folder belongs to the user rather than to us.
    """

    def __init__(self, folder: Path | str) -> None:
        self._folder = Path(folder)

    @property
    def folder(self) -> Path:
        return self._folder

    @property
    def path(self) -> Path:
        return self._folder / PROJECT_FILE_NAME

    def load(self) -> ProjectState:
        """Return the saved project, or an empty one if there is nothing usable.

        A missing file, an unreadable one, one that is not JSON, one that is
        JSON but not an object, and one that is an object full of nonsense all
        come back as a usable project, with a line in the log where something
        was thrown away. Nothing here returns ``None`` and nothing here raises,
        because the Review window must always open: a person who cannot open
        the window cannot even see what they have lost, let alone do it again.
        """
        data = read_json_object(self.path)
        if data is None:
            return ProjectState()
        try:
            return ProjectState.from_dict(data)
        # Everything above is meant to degrade rather than raise, so nothing
        # should reach here. The catch is deliberately as wide as it can be
        # made anyway: whatever a hand edit does to the file, the window opens.
        except Exception:  # noqa: BLE001
            _log.warning("%s could not be understood; starting an empty project.", self.path,
                         exc_info=True)
            return ProjectState()

    def save(self, state: ProjectState) -> bool:
        """Write the project, returning whether it worked.

        The write goes through a temporary file that is moved into place, so a
        save interrupted by a crash or a full disk leaves the previous project
        intact rather than a truncated file where the person's review used to
        be. The answer must not be ignored: a correction that did not reach the
        disk is work the person will never know they have to do again unless
        somebody tells them now.
        """
        return write_json_object(self.path, state.to_dict())

    def save_keeping_learned_names(
        self, state: ProjectState, learned_here: list[LearnedName]
    ) -> bool:
        """Save the project without bringing back a name removed from the file.

        ``learned_here`` is what the caller has learned since its last save.
        The names written are the ones in the file as it is on disk now, with
        those added, rather than the ones in ``state``. This is the failure it
        prevents: the Review window holds the project for as long as it is
        open, and a person may remove a name from the file in the meantime.
        Saving the window's copy would write that name back without anybody
        noticing.

        When the file is missing or cannot be read there is nothing newer to
        respect, so the names in ``state`` are kept. ``state.learned_names``
        is set to what was written. Returns whether the save worked.
        """
        disk_names, status = read_learned_names(self._folder)
        if status is JsonReadStatus.READ:
            state.learned_names = merge_learned_names(disk_names, learned_here)
        else:
            state.learned_names = merge_learned_names(state.learned_names, learned_here)
        return self.save(state)
