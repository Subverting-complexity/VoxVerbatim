"""The words the services would otherwise get wrong, and what the user knows about them.

Speech-to-text services are good at ordinary words and weak at exactly the
words that matter most in a recording: a client's surname, a product nobody
else sells, an acronym that sounds like an everyday word. Every service
offers some way of being told those words beforehand, and being told makes a
real difference to whether they come back spelled correctly. This module is
where that knowledge is kept.

It is kept at four levels rather than in one list, because the same word is
not equally likely in every recording. A surname that is almost certain in a
recording of one client's meeting is a distraction in a recording of
somebody else's, and a long list of distractions makes a service worse
rather than better. So the user keeps a global list, a list per client, a
list per project and a list per speaker, and chooses which of them apply
before a recording is transcribed.

Beside the lists sit the corrections a person has actually made. Those are
the strongest thing here. A word the user typed while listening to their own
audio is evidence of two things at once: that the right spelling is worth
listening for next time, and that a particular service gets that word wrong,
which section 23 of the specification wants recorded so that a service's
recurring mistakes can be weighed later.

Nothing here depends on Qt or on any provider library, so it can be tested
on its own.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vox_verbatim.json_store import read_json_object, write_json_object
from vox_verbatim.transcription.model import Language, Provider
from vox_verbatim.transcription.normalise import (
    EquivalenceKind,
    _compound_form,
    _has_german_letter,
    are_equivalent,
    equivalence_kind,
    normalise,
)

if TYPE_CHECKING:
    from vox_verbatim.transcription.project import LearnedName

VOCABULARY_FILE_NAME = "vocabulary.json"

#: Bumped only if the shape on disk changes in a way that needs migrating.
VOCABULARY_FORMAT_VERSION = 1

#: How many times a correction must have been seen before it is offered to the
#: services as a term to listen out for. One correction can be a typo or a
#: one-off; the same correction twice is a pattern worth acting on.
MINIMUM_CORRECTION_OCCURRENCES = 2


class TermCategory(str, Enum):
    """What kind of thing a term is.

    The category is not used to change how a term is sent to a service. It is
    there so the user can find and manage a long list, and so that a
    disagreement over a term the user marked as a person's name can be
    treated more carefully than a disagreement over an ordinary word.
    """

    PERSON = "person"
    COMPANY = "company"
    PRODUCT = "product"
    PLACE = "place"
    ACRONYM = "acronym"
    TECHNICAL = "technical"
    OTHER = "other"

    @property
    def display_name(self) -> str:
        return self.value.capitalize()

    @classmethod
    def from_value(cls, value: Any) -> "TermCategory | None":
        """Return the category named by ``value``, or ``None`` if it names none."""
        if not isinstance(value, str):
            return None
        try:
            return cls(value.strip().lower())
        except ValueError:
            return None


class VocabularyLevel(str, Enum):
    """How narrowly a list of terms applies.

    They are declared from the widest to the narrowest, and :attr:`specificity`
    depends on that order. Specificity is what decides which terms survive
    when a service will not accept the whole list.
    """

    GLOBAL = "global"
    CLIENT = "client"
    PROJECT = "project"
    SPEAKER = "speaker"

    @property
    def display_name(self) -> str:
        return self.value.capitalize()

    @property
    def specificity(self) -> int:
        """How narrow this level is, with a larger number meaning narrower."""
        return list(VocabularyLevel).index(self)

    @classmethod
    def from_value(cls, value: Any) -> "VocabularyLevel | None":
        if not isinstance(value, str):
            return None
        try:
            return cls(value.strip().lower())
        except ValueError:
            return None


def _language_from_value(value: Any) -> Language | None:
    """Return the language named by ``value``, or ``None`` if it names none.

    A term with no language is the normal case: most names are spelled the
    same whatever is being spoken around them. ``None`` therefore means "this
    term belongs everywhere", not "unknown".
    """
    if not isinstance(value, str) or not value.strip():
        return None
    language = Language.from_code(value)
    return None if language is Language.UNKNOWN else language


def _clean_text_list(value: Any) -> tuple[str, ...]:
    """Return the usable strings in ``value``, in order and without repeats."""
    if not isinstance(value, list):
        return ()
    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return tuple(cleaned)


def _clean_count(value: Any, default: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return default
    return value


def _plain_form(text: str) -> str:
    """The form used to decide whether two recorded strings are the same string.

    This module compares text in two quite different ways, and confusing them
    causes real damage, so they are named apart.

    Asking "did a service say this term?" is a question about sound, and
    :func:`~vox_verbatim.transcription.normalise.are_equivalent` is the
    right tool: "Muller" and "Müller" and "twenty five" and "25" are all the
    same spoken evidence, and a vocabulary that could not see that would miss
    most of what it exists to catch.

    Asking "is this the same string I already wrote down?" is a different
    question, and the full comparison is far too strong for it. The user
    records "Mueller" as a thing the services wrongly produce for "Müller",
    and those two normalise to exactly the same form. Judge them by that form
    and the mistake becomes indistinguishable from the correction, which
    destroys the one piece of knowledge the user was trying to record. So
    stored strings are compared on this form instead: surrounding space and
    capitals set aside, and nothing else. Note that ``casefold`` leaves the
    German letters where they are, which is precisely why it is safe here.
    """
    return text.strip().casefold()


@dataclass(frozen=True)
class VocabularyTerm:
    """One word or phrase the user wants the services to listen out for."""

    text: str
    """Exactly how the term should be spelled in the finished transcript."""

    category: TermCategory | None = None
    language: Language | None = None
    """The language this term belongs to, where it belongs to only one."""

    pronunciation_hints: tuple[str, ...] = ()
    """How the term sounds, for the services that can use that."""

    common_misrecognitions: tuple[str, ...] = ()
    """What the services usually hear instead.

    Kept because it lets reconciliation recognise a wrong candidate as a
    known mistake rather than as an ordinary rival spelling, which is a much
    stronger reason to prefer the right one.
    """

    confirmation_count: int = 0
    """How many times a person has confirmed this term in a real transcript.

    A term nobody has ever confirmed is a guess the user typed in once. A
    term confirmed twenty times is known to occur in their recordings, and
    that is what decides which terms are kept when a service will not take
    them all.
    """

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "category": self.category.value if self.category else None,
            "language": self.language.value if self.language else None,
            "pronunciation_hints": list(self.pronunciation_hints),
            "common_misrecognitions": list(self.common_misrecognitions),
            "confirmation_count": self.confirmation_count,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "VocabularyTerm | None":
        """Build a term from loaded JSON, or return ``None`` if there is no term there.

        A term with no text is not a term the user has half-filled in; it is
        nothing at all, and keeping it would put an empty string in front of
        every service. Everything else falls back to its default rather than
        throwing the whole term away.
        """
        if not isinstance(data, dict):
            return None
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        return cls(
            text=text.strip(),
            category=TermCategory.from_value(data.get("category")),
            language=_language_from_value(data.get("language")),
            pronunciation_hints=_clean_text_list(data.get("pronunciation_hints")),
            common_misrecognitions=_clean_text_list(data.get("common_misrecognitions")),
            confirmation_count=_clean_count(data.get("confirmation_count")),
        )


@dataclass
class VocabularyProfile:
    """One named list of terms, at one level.

    There is one class rather than four because a global list and a speaker
    list differ in nothing but how narrowly they apply. Four near-identical
    classes would have to be kept in step by hand for ever, and every piece
    of code that walked over them would have to name all four.
    """

    id: str
    level: VocabularyLevel = VocabularyLevel.GLOBAL
    name: str = ""
    terms: list[VocabularyTerm] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        """What to show a person, falling back to the identifier."""
        return self.name or self.id

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "level": self.level.value,
            "name": self.name,
            "terms": [term.to_dict() for term in self.terms],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "VocabularyProfile | None":
        """Build a profile from loaded JSON, or ``None`` if it has no identifier.

        Without an identifier a profile can never be chosen for a recording,
        so there is nothing useful to keep.
        """
        if not isinstance(data, dict):
            return None
        identifier = data.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            return None
        raw_terms = data.get("terms")
        terms: list[VocabularyTerm] = []
        if isinstance(raw_terms, list):
            for item in raw_terms:
                term = VocabularyTerm.from_dict(item)
                if term is not None:
                    terms.append(term)
        name = data.get("name")
        return cls(
            id=identifier.strip(),
            level=VocabularyLevel.from_value(data.get("level")) or VocabularyLevel.GLOBAL,
            name=name.strip() if isinstance(name, str) else "",
            terms=terms,
        )


@dataclass(frozen=True)
class LearnedCorrection:
    """One change a person made, and how often they have had to make it."""

    wrong_text: str
    """What the transcript said before the person changed it."""

    right_text: str
    """What they changed it to."""

    occurrences: int = 1
    last_seen: str = ""
    """When it last happened, as an ISO 8601 string in local time."""

    provider: Provider | None = None
    """Which service produced the wrong text, where that is known.

    This is the part section 23 asks for. One service repeatedly mishearing
    the same word is evidence about that service, not only about the word,
    and provider weighting cannot learn anything from corrections that do not
    say who made the mistake.
    """

    def to_dict(self) -> dict[str, Any]:
        return {
            "wrong_text": self.wrong_text,
            "right_text": self.right_text,
            "occurrences": self.occurrences,
            "last_seen": self.last_seen,
            "provider": self.provider.value if self.provider else None,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "LearnedCorrection | None":
        if not isinstance(data, dict):
            return None
        wrong = data.get("wrong_text")
        right = data.get("right_text")
        if not isinstance(wrong, str) or not isinstance(right, str):
            return None
        if not wrong.strip() or not right.strip():
            return None
        provider_value = data.get("provider")
        provider: Provider | None = None
        if isinstance(provider_value, str):
            try:
                provider = Provider(provider_value.strip().lower())
            except ValueError:
                provider = None
        last_seen = data.get("last_seen")
        return cls(
            wrong_text=wrong.strip(),
            right_text=right.strip(),
            occurrences=_clean_count(data.get("occurrences"), default=1) or 1,
            last_seen=last_seen if isinstance(last_seen, str) else "",
            provider=provider,
        )


@dataclass
class LearnedCorrections:
    """Every correction a person has made, counted rather than listed twice.

    Counting matters more than listing here. The same correction made once is
    a slip; the same correction made every week is a word the services cannot
    hear, and only the count tells the two apart.
    """

    corrections: list[LearnedCorrection] = field(default_factory=list)

    def record(
        self,
        wrong_text: str,
        right_text: str,
        provider: Provider | None = None,
        when: str | None = None,
    ) -> LearnedCorrection | None:
        """Note that a person changed ``wrong_text`` to ``right_text``.

        A correction that has been seen before has its count raised and its
        date moved on, rather than being added again. Which service produced
        the wrong text is part of what makes two corrections the same, so
        that one word misheard by two services stays two separate pieces of
        evidence about two separate services.

        A change of spelling alone still counts. "Mueller" corrected to
        "Müller" is the same sound and a different word on the page, and the
        spelling is the whole reason the vocabulary exists. Only text that is
        identical once trimmed is nothing to record.

        Both halves are compared on their plain form rather than on their
        spoken equivalence, and in their own direction. Judging them by
        equivalence would file a correction of "Müller" to "Mueller" as
        another sighting of the correction that runs the other way, and the
        count would then say the opposite of what happened.

        Returns the correction as it now stands, or ``None`` when there was
        nothing to record because the text did not really change.
        """
        wrong = wrong_text.strip()
        right = right_text.strip()
        if not wrong or not right or wrong == right:
            return None
        stamp = when or datetime.now().isoformat(timespec="seconds")
        for position, existing in enumerate(self.corrections):
            if existing.provider is not provider:
                continue
            if _plain_form(existing.wrong_text) != _plain_form(wrong):
                continue
            if _plain_form(existing.right_text) != _plain_form(right):
                continue
            updated = LearnedCorrection(
                wrong_text=existing.wrong_text,
                right_text=existing.right_text,
                occurrences=existing.occurrences + 1,
                last_seen=stamp,
                provider=existing.provider,
            )
            self.corrections[position] = updated
            return updated
        added = LearnedCorrection(
            wrong_text=wrong,
            right_text=right,
            occurrences=1,
            last_seen=stamp,
            provider=provider,
        )
        self.corrections.append(added)
        return added

    def corrections_for(self, wrong_text: str) -> list[LearnedCorrection]:
        """Every correction recorded against ``wrong_text``, whichever service made it."""
        return [
            correction
            for correction in self.corrections
            if are_equivalent(correction.wrong_text, wrong_text)
        ]

    def mistakes_by_provider(self) -> dict[Provider, int]:
        """How many corrections each service has caused, most recently counted.

        This is the raw material for provider weighting. It is deliberately
        only a count: turning it into a judgement about a service belongs
        with the weighting rules, not here.
        """
        counts: dict[Provider, int] = {}
        for correction in self.corrections:
            if correction.provider is None:
                continue
            seen = counts.get(correction.provider, 0)
            counts[correction.provider] = seen + correction.occurrences
        return counts

    def as_terms(
        self, minimum_occurrences: int = MINIMUM_CORRECTION_OCCURRENCES
    ) -> list[VocabularyTerm]:
        """The recurring corrections, as terms to send to the services.

        The right text becomes the term and the wrong text becomes a known
        misrecognition of it, which is exactly the shape the rest of this
        module already understands. Corrections seen fewer than
        ``minimum_occurrences`` times are left out, because a service told to
        listen for a word that was only ever corrected once is being pushed
        towards a mistake rather than away from one.
        """
        merged: dict[str, VocabularyTerm] = {}
        for correction in self.corrections:
            if correction.occurrences < minimum_occurrences:
                continue
            key = normalise(correction.right_text)
            existing = merged.get(key)
            if existing is None:
                merged[key] = VocabularyTerm(
                    text=correction.right_text,
                    common_misrecognitions=(correction.wrong_text,),
                    confirmation_count=correction.occurrences,
                )
                continue
            misrecognitions = list(existing.common_misrecognitions)
            if correction.wrong_text not in misrecognitions:
                misrecognitions.append(correction.wrong_text)
            merged[key] = VocabularyTerm(
                text=existing.text,
                category=existing.category,
                language=existing.language,
                pronunciation_hints=existing.pronunciation_hints,
                common_misrecognitions=tuple(misrecognitions),
                confirmation_count=existing.confirmation_count + correction.occurrences,
            )
        return list(merged.values())

    def to_dict(self) -> dict[str, Any]:
        return {"corrections": [correction.to_dict() for correction in self.corrections]}

    @classmethod
    def from_dict(cls, data: Any) -> "LearnedCorrections":
        if not isinstance(data, dict):
            return cls()
        raw = data.get("corrections")
        if not isinstance(raw, list):
            return cls()
        found: list[LearnedCorrection] = []
        for item in raw:
            correction = LearnedCorrection.from_dict(item)
            if correction is not None:
                found.append(correction)
        return cls(corrections=found)


@dataclass
class Vocabulary:
    """Everything the user knows about the words in their recordings."""

    profiles: list[VocabularyProfile] = field(default_factory=list)
    corrections: LearnedCorrections = field(default_factory=LearnedCorrections)

    def profile(self, profile_id: str) -> VocabularyProfile | None:
        for profile in self.profiles:
            if profile.id == profile_id:
                return profile
        return None

    def profiles_at(self, level: VocabularyLevel) -> list[VocabularyProfile]:
        return [profile for profile in self.profiles if profile.level is level]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": VOCABULARY_FORMAT_VERSION,
            "profiles": [profile.to_dict() for profile in self.profiles],
            "learned": self.corrections.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Vocabulary":
        """Build a vocabulary from loaded JSON, ignoring anything unusable.

        A profile that cannot be understood is dropped on its own. Losing one
        client's list of names is a small loss; refusing to load the file
        because of it would lose all of them.
        """
        raw_profiles = data.get("profiles")
        profiles: list[VocabularyProfile] = []
        if isinstance(raw_profiles, list):
            for item in raw_profiles:
                profile = VocabularyProfile.from_dict(item)
                if profile is not None:
                    profiles.append(profile)
        return cls(
            profiles=profiles,
            corrections=LearnedCorrections.from_dict(data.get("learned")),
        )


class VocabularyStore:
    """Reads and writes the vocabulary file at a fixed location."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> Vocabulary:
        """Return the saved vocabulary, or an empty one if there is none usable."""
        data = read_json_object(self._path)
        if data is None:
            return Vocabulary()
        return Vocabulary.from_dict(data)

    def save(self, vocabulary: Vocabulary) -> bool:
        return write_json_object(self._path, vocabulary.to_dict())


#: Terms are ordered on this scale, where a smaller number comes first. A
#: level's place on it is the negative of how narrow it is, so the speaker's
#: own list beats the global one.
def _priority_of(level: VocabularyLevel) -> int:
    return -level.specificity


#: Where the corrections a person has made sit against the four levels: ahead
#: of all of them, because they are the only entries here that were produced
#: while somebody was listening to their own audio, rather than typed into a
#: list beforehand in case they came up.
_CORRECTION_PRIORITY = -len(VocabularyLevel)


def resolve_terms(
    vocabulary: Vocabulary,
    profile_ids: Sequence[str],
    include_corrections: bool = True,
    minimum_correction_occurrences: int = MINIMUM_CORRECTION_OCCURRENCES,
) -> list[VocabularyTerm]:
    """Combine the chosen profiles into one ordered, de-duplicated list of terms.

    The order is the point of this function. Several services cap how many
    terms they will accept, and a few cap how many characters the whole list
    may run to, so some of what the user typed will not reach them. What gets
    cut should therefore be what matters least. Two things decide that.

    The first is how narrowly a term applies. A name on a speaker's own list
    is far more likely in a recording of that speaker than a name on the
    global list, so the narrow lists come first and the global list last.
    Corrections a person actually made come before all of them, because they
    are the only evidence here that came from real audio.

    The second is how often a person has confirmed the term. Within one level,
    a term seen twenty times in real transcripts beats one that has never been
    seen at all.

    A term on more than one list appears once, at its narrowest position, and
    its confirmations are added together, so that a term the user thought
    worth writing down several times is treated as the strong signal it is.
    """
    gathered: list[tuple[int, int, VocabularyTerm]] = []
    arrival = 0

    if include_corrections:
        for term in vocabulary.corrections.as_terms(minimum_correction_occurrences):
            gathered.append((_CORRECTION_PRIORITY, arrival, term))
            arrival += 1

    # The profiles are visited narrowest first so that the entry kept for a
    # repeated term is the one from the most specific list, which is the one
    # carrying the category and language the user chose for that context.
    chosen = [profile for profile in (vocabulary.profile(pid) for pid in profile_ids) if profile]
    chosen.sort(key=lambda profile: -profile.level.specificity)
    for profile in chosen:
        priority = _priority_of(profile.level)
        for term in profile.terms:
            gathered.append((priority, arrival, term))
            arrival += 1

    merged: dict[str, tuple[int, int, VocabularyTerm]] = {}
    for priority, position, term in gathered:
        key = _key_for(term.text, merged)
        existing = merged.get(key)
        if existing is None:
            merged[key] = (priority, position, term)
            continue
        merged[key] = (existing[0], existing[1], _combine(existing[2], term))

    ordered = sorted(
        merged.values(),
        key=lambda entry: (entry[0], -entry[2].confirmation_count, entry[1]),
    )
    return [entry[2] for entry in ordered]


def _key_for(text: str, merged: dict[str, tuple[int, int, VocabularyTerm]]) -> str:
    """Find which gathered term ``text`` belongs with, or give it a key of its own.

    The comparison form settles almost every case in one lookup. It does not
    settle all of them: a name written "Müller" on one list and "Muller" on
    another is the same name, but only the full comparison sees that, because
    recognising a dropped umlaut needs to know that one of the two texts has
    a German letter in it. Merging matters here rather than being a tidiness
    concern, since ElevenLabs is sent fewer than a hundred terms and spending
    two of those places on one name wastes one of them.

    So a miss falls back to comparing against the terms gathered so far. That
    is a walk over the list for every genuinely new term, which sounds worse
    than it is: five hundred terms cost about a sixth of a second in total,
    once per recording, against a transcription measured in minutes.
    """
    key = normalise(text)
    if key in merged:
        return key
    for other, entry in merged.items():
        if are_equivalent(entry[2].text, text):
            return other
    return key


def _combine(kept: VocabularyTerm, other: VocabularyTerm) -> VocabularyTerm:
    """Fold a repeat of a term into the copy already being kept.

    The kept copy wins every disagreement, because it came from the narrower
    list. What the other copy adds is the things it knows and the kept copy
    does not: more hints, more misrecognitions, and its own confirmations.
    """
    hints = list(kept.pronunciation_hints)
    for hint in other.pronunciation_hints:
        if hint not in hints:
            hints.append(hint)
    misrecognitions = list(kept.common_misrecognitions)
    for wrong in other.common_misrecognitions:
        if wrong not in misrecognitions:
            misrecognitions.append(wrong)
    return VocabularyTerm(
        text=kept.text,
        category=kept.category or other.category,
        language=kept.language or other.language,
        pronunciation_hints=tuple(hints),
        common_misrecognitions=tuple(misrecognitions),
        confirmation_count=kept.confirmation_count + other.confirmation_count,
    )


def terms_from_learned_names(names: Iterable["LearnedName"]) -> list[VocabularyTerm]:
    """Turn the names a folder learned in its reviews into terms for the services.

    Each name was typed by a person listening to a recording in that folder,
    so it counts as confirmed once. The spellings the services wrote instead
    go with it, so reconciliation can recognise them as known mistakes. A
    name whose language was never established belongs to every language,
    which is what a term with no language means.
    """
    terms: list[VocabularyTerm] = []
    for name in names:
        text = name.text.strip()
        if not text:
            continue
        terms.append(
            VocabularyTerm(
                text=text,
                language=_language_from_value(name.language),
                common_misrecognitions=_clean_text_list(list(name.wrong_forms)),
                confirmation_count=1,
            )
        )
    return terms


def folder_terms_first(
    folder_terms: Sequence[VocabularyTerm],
    profile_terms: Sequence[VocabularyTerm],
) -> list[VocabularyTerm]:
    """Put a folder's own names ahead of the terms from the chosen profiles.

    A service that will not take every term cuts from the end of the list,
    so whatever comes first survives. The folder's names come first because
    they were heard in this folder's recordings, where a profile term was
    typed into a list in case it came up. A profile term that is the same
    word as a folder name is folded into the folder's entry rather than
    sent twice, which would spend one of the few places a service offers.

    Two terms are folded together only when their languages agree. A profile
    term with no language belongs to every language, and folding it into an
    Afrikaans folder name would give it that name's language, so a run with
    Afrikaans off would drop a term it sends today.
    """
    merged: dict[str, tuple[int, int, VocabularyTerm]] = {}
    for position, term in enumerate([*folder_terms, *profile_terms]):
        key = _key_for(term.text, merged)
        existing = merged.get(key)
        if existing is not None and existing[2].language != term.language:
            key = f"{key}|{term.language}"
            existing = merged.get(key)
        if existing is None:
            merged[key] = (0, position, term)
        else:
            merged[key] = (existing[0], existing[1], _combine(existing[2], term))
    # A dictionary keeps the order its keys arrived in, which is the order wanted.
    return [entry[2] for entry in merged.values()]


@dataclass(frozen=True)
class VocabularyMatch:
    """A term the user knows about, and how closely the text matched it.

    Reconciliation needs both halves. A candidate that is the user's term
    character for character is strong evidence for that candidate. A
    candidate that only reaches the term once numbers and hyphens have been
    set aside is evidence that the *word* is right, and says nothing about
    which of the rival spellings of it to print.
    """

    term: VocabularyTerm
    kind: EquivalenceKind

    @property
    def is_exact(self) -> bool:
        """Whether the text is the term as the user wrote it, give or take capitals."""
        return self.kind in (EquivalenceKind.IDENTICAL, EquivalenceKind.CASE_ONLY)


class VocabularyIndex:
    """Answers "is this a word the user told us about?".

    Reconciliation asks this of every candidate spelling it is weighing up.
    A candidate that matches a known term has a real reason to be preferred
    over one that does not, and a candidate that matches a known
    misrecognition has a real reason to be rejected.

    Matching a candidate uses the full spoken-equivalence comparison, because
    a person who wrote "Van Der Merwe" in their list means the same word a
    service returned as "van der merwe", and one who wrote "Müller" means the
    word a service returned as "Muller".

    The known mistakes are held apart from that, and matched only on their
    plain form. This is not fussiness. A user recording that services write
    "Mueller" where they should write "Müller" has recorded two texts that
    the equivalence comparison considers identical, so an index built on that
    comparison would answer "yes, a known term" and "yes, a known mistake" to
    the same question, and the distinction the user was drawing would vanish
    without anything appearing to go wrong.
    """

    def __init__(self, terms: Iterable[VocabularyTerm]) -> None:
        self._terms: list[VocabularyTerm] = list(terms)
        self._by_text: dict[str, VocabularyTerm] = {}
        self._by_dropped: dict[str, VocabularyTerm] = {}
        self._by_dropped_german: dict[str, VocabularyTerm] = {}
        self._exact_texts: dict[str, VocabularyTerm] = {}
        self._misrecognitions: dict[str, VocabularyTerm] = {}
        for term in self._terms:
            self._by_text.setdefault(normalise(term.text), term)
            dropped = _compound_form(term.text, True)
            self._by_dropped.setdefault(dropped, term)
            if _has_german_letter(term.text):
                self._by_dropped_german.setdefault(dropped, term)
            self._exact_texts.setdefault(_plain_form(term.text), term)
            for wrong in term.common_misrecognitions:
                self._misrecognitions.setdefault(_plain_form(wrong), term)

    @property
    def terms(self) -> list[VocabularyTerm]:
        return list(self._terms)

    def __len__(self) -> int:
        return len(self._terms)

    def __contains__(self, text: object) -> bool:
        return isinstance(text, str) and self.knows(text)

    def knows(self, text: str) -> bool:
        """Whether ``text`` is one of the user's terms."""
        return self.match(text) is not None

    def match(self, text: str) -> VocabularyMatch | None:
        """The term ``text`` names and how it matched it, or ``None``.

        Three questions are asked in turn, and the order is what keeps them
        from contradicting each other.

        A text that is one of the user's terms as they wrote it wins outright.
        That comes first so that a name which happens to be listed elsewhere
        as somebody else's misspelling is still found.

        Failing that, a text that is exactly a mistake the user recorded loses
        outright, because handing vocabulary support to the spelling they have
        already rejected is worse than giving none at all.

        Only then does the spoken-equivalence comparison decide, and it is
        answered entirely by lookup. Reconciliation asks this question several
        times for every word of a transcript, and almost every answer is "no",
        so a miss has to be cheap: walking three hundred terms with a
        comparison each made a ten-thousand-word reconcile ten times slower
        than the same reconcile with no vocabulary at all.
        """
        cleaned = text.strip() if text else ""
        if not cleaned:
            return None

        exact = self._exact_texts.get(_plain_form(cleaned))
        if exact is not None:
            return VocabularyMatch(exact, equivalence_kind(exact.text, cleaned))
        if _plain_form(cleaned) in self._misrecognitions:
            return None

        found = self._equivalent_term(cleaned)
        if found is None:
            return None
        return VocabularyMatch(found, equivalence_kind(found.text, cleaned))

    def _equivalent_term(self, text: str) -> VocabularyTerm | None:
        """The first term that is the same spoken evidence as ``text``.

        This gives the answer walking the terms with
        :func:`~vox_verbatim.transcription.normalise.are_equivalent`
        would give, without the walk. That function says two texts are the
        same when their comparison forms agree, or -- only when one of the
        two has a German letter in it -- when their forms with the umlauts
        dropped agree. Both are questions about a key worked out from each
        text alone, so each term is filed under its two keys when the index
        is built and a text is answered by looking its own two keys up.

        The German condition is what the two dropped-form tables are for.
        A text with a German letter may meet any term through the dropped
        form; a text without one may only meet a term that has one. There is
        nothing the comparison can say yes to that these lookups cannot,
        which is why no walk remains for the cases the keys miss.

        Within each table the earliest term wins, as it did in the walk.
        """
        found = self._by_text.get(normalise(text))
        if found is not None:
            return found
        dropped = _compound_form(text, True)
        if _has_german_letter(text):
            return self._by_dropped.get(dropped)
        return self._by_dropped_german.get(dropped)
    def misrecognition_of(self, text: str) -> VocabularyTerm | None:
        """The term ``text`` is a known mistake for, or ``None``.

        Matched on the plain form only, for the reason given on the class: a
        looser comparison folds a recorded mistake into the very term it is a
        mistake for. Capitals and surrounding space are still set aside,
        because a service writing "bosh" rather than "Bosh" has made the same
        mistake, and casefolding cannot fold a German letter into anything.
        """
        cleaned = text.strip() if text else ""
        if not cleaned:
            return None
        return self._misrecognitions.get(_plain_form(cleaned))
