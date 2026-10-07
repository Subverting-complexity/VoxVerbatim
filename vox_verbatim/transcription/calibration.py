"""How well each service has actually done, and what that is worth as a weight.

When two services disagree about a word, something has to decide which of
them to believe. The honest answer is not a number a vendor publishes and
not an opinion written into the source code: it is what happened the last
few hundred times that service was believed and a person then had to change
the word. This module is where that record is kept, and where it is turned
into a weight the reconciliation rules can use.

Section 24 of the specification lists many dimensions a service's
performance might vary over. Only some of them can be observed by this
application, and only those are recorded here: the language of the word,
whether it was a proper noun, whether it was numeric, and which vocabulary
category it belongs to. Audio quality,
recording environment, microphone and speaker accent are deliberately left
out. Nothing in the application measures any of them today, so a column for
them would hold nothing but zeros while looking exactly like knowledge. If
a way to observe one of them arrives later, it becomes another
:class:`Dimension` and the file grows a key; nothing else has to change.

The speaker is left out too, for a different reason. A speaker label such
as ``speaker_0`` means a different person in every recording, so a
breakdown by label measures nothing, and a label a person has replaced with
a name would put a client's name into a file that every folder shares.
Speaker mistakes are therefore kept as one total for each service.

The same section is explicit about the order of work in version 1: start
with explicit rules, collect real correction data, and only then calibrate.
That is why every weight here begins at a documented default and moves only
as evidence arrives, rather than being learned from nothing by a model.

Three decisions in the weighting are worth explaining, because each of them
protects against a way this could go wrong and look authoritative while
doing it.

The first is the prior. A weight is not the raw success rate. It is the
success rate seen through a prior of :data:`PRIOR_STRENGTH` imagined
observations at :data:`DEFAULT_WEIGHT`, so a service needs a real body of
evidence before its weight moves far. A service that gets its first three
words wrong is far more likely to have met three hard words than to be
three times worse than the rest, and a weighting system that marks it down
on that basis is worse than no weighting at all: it is confidently wrong,
and the confidence is what does the damage.

The second is that only outcomes a person confirmed move a weight. A
candidate that reconciliation rejected is counted and reported, but it does
not count against the service, because nobody has checked it. Letting
rejections carry weight would close a loop in which the rules mark down
whichever service they already disagree with, and then disagree with it
more.

The third is that the four confidence categories are measured and reported,
never quietly retuned. Section 16 asks for the categories to be calibrated
against real corrections. Measuring how often each category turned out to
need changing is what makes that possible; moving the thresholds
automatically would mean a transcript produced today could not be compared
with one produced last month, and nobody would know why.

The statistics live in one small JSON file, read and written through the
shared store, and a damaged file is ignored exactly as a damaged settings
file is: it comes back as an empty record rather than stopping the
application. Losing the accumulated statistics costs accuracy that will
rebuild itself; refusing to start costs the user their work.

Nothing here depends on Qt or on any provider library, so it can be tested
on its own.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from vox_verbatim.json_store import read_json_object, write_json_object
from vox_verbatim.transcription.model import Confidence, Language, Provider
from vox_verbatim.transcription.reconcile import (
    DEFAULT_PROVIDER_RELIABILITY,
    DEFAULT_RELIABILITY,
    ReliabilityWeights,
)
from vox_verbatim.transcription.vocabulary import TermCategory

CALIBRATION_FILE_NAME = "calibration.json"

#: Bumped only if the shape on disk changes in a way that needs migrating.
CALIBRATION_FORMAT_VERSION = 1

#: What a service is assumed to be worth before anything is known about it.
#: It is deliberately high and deliberately the same for every service.
#: Version 1 has no grounds to rank them, and a default that differed
#: between them would be exactly the permanently assumed weighting section
#: 23 says not to build.
DEFAULT_WEIGHT = 0.9

#: How much evidence it takes to argue with the default, measured in words.
#: The weight is the success rate of ``PRIOR_STRENGTH`` imagined
#: observations at :data:`DEFAULT_WEIGHT` plus the real ones, so fifty real
#: words carry as much say as the assumption does, and a handful cannot
#: carry much at all: five corrections in five chosen words move a weight
#: from 0.90 to about 0.82, which is a nudge rather than a verdict.
PRIOR_STRENGTH = 50.0

#: How much evidence a single dimension needs before its own weight is used
#: in place of the service's overall one. Below this the two answers differ
#: by almost nothing anyway, and preferring the thinner one would only add
#: noise.
MINIMUM_DIMENSION_EVIDENCE = 30

#: How many times something has to have happened before it is called
#: recurring. Once is an accident; twice is the beginning of a pattern, and
#: it is the same threshold the vocabulary uses before it teaches a
#: correction to the services.
REPEATED_MISTAKE_THRESHOLD = 2


class Dimension(str, Enum):
    """The ways a service's performance is broken down.

    These are the dimensions from section 24 that this application can
    actually observe. They are listed here narrowest first, and
    :meth:`ProviderStatistics.weight_for_word` relies on that order when it
    looks for the most specific answer that has evidence behind it.
    """

    VOCABULARY = "vocabulary"
    NUMERIC = "numeric"
    PROPER_NOUN = "proper_noun"
    LANGUAGE = "language"

    @property
    def display_name(self) -> str:
        return _DIMENSION_DISPLAY_NAMES[self]


_DIMENSION_DISPLAY_NAMES: dict[Dimension, str] = {
    Dimension.VOCABULARY: "Vocabulary category",
    Dimension.NUMERIC: "Numbers",
    Dimension.PROPER_NOUN: "Names",
    Dimension.LANGUAGE: "Language",
}

#: The value stored for the two dimensions that are simply true or false.
#: Only the true case is recorded. The false case is almost every word in
#: every recording, so a bucket for it would double the file to say the
#: same thing the overall record already says.
_YES = "yes"

#: The key prefix an older file used for its speaker breakdown, which is
#: dropped on load. See the module docstring for why.
_SPEAKER_PREFIX = "speaker:"


def dimension_key(dimension: Dimension, value: str) -> str:
    """The key one bucket of evidence is stored under, such as ``language:en``."""
    return f"{dimension.value}:{value}"


@dataclass(frozen=True)
class Observation:
    """What was known about one word, for the purpose of counting it.

    It is deliberately a small flat record rather than the word itself.
    Statistics are kept for years and read back long after the transcript
    they came from has been forgotten, so what is counted has to be the
    handful of facts that will still mean something then.
    """

    provider: Provider
    language: Language = Language.UNKNOWN
    is_proper_noun: bool = False
    is_numeric: bool = False
    vocabulary_category: TermCategory | None = None

    def dimension_keys(self) -> tuple[str, ...]:
        """The buckets this word belongs to, narrowest first.

        A dimension that says nothing is left out entirely. An unknown
        language is not a language, so it gets no bucket.
        """
        keys: list[str] = []
        if self.vocabulary_category is not None:
            keys.append(dimension_key(Dimension.VOCABULARY, self.vocabulary_category.value))
        if self.is_numeric:
            keys.append(dimension_key(Dimension.NUMERIC, _YES))
        if self.is_proper_noun:
            keys.append(dimension_key(Dimension.PROPER_NOUN, _YES))
        if self.language is not Language.UNKNOWN:
            keys.append(dimension_key(Dimension.LANGUAGE, self.language.value))
        return tuple(keys)


@dataclass
class OutcomeCounts:
    """How one service fared in one bucket, in whole words.

    Three numbers rather than a rate, because a rate on its own is the one
    thing nobody should be shown: three words out of three is not a hundred
    per cent, it is three words.
    """

    chosen: int = 0
    """How many times this service's candidate became the final word."""

    corrected: int = 0
    """How many of those a person then changed."""

    rejected: int = 0
    """How many times its candidate lost to another service's.

    Counted and reported, but never weighed. See the module docstring for
    why a rejection is not treated as a mistake.
    """

    @property
    def confirmed(self) -> int:
        """Words this service was believed for and nobody had to change."""
        return max(0, self.chosen - self.corrected)

    @property
    def correction_rate(self) -> float | None:
        """The share of chosen words a person changed, or ``None`` if none were chosen."""
        if self.chosen <= 0:
            return None
        return self.corrected / self.chosen

    @property
    def weight(self) -> float:
        """How far this service should be believed here, between zero and one.

        With nothing recorded this is exactly :data:`DEFAULT_WEIGHT`, which
        is what makes the default the honest starting point rather than a
        special case in the code.
        """
        prior = DEFAULT_WEIGHT * PRIOR_STRENGTH
        return (self.confirmed + prior) / (self.chosen + PRIOR_STRENGTH)

    @property
    def has_evidence(self) -> bool:
        return self.chosen > 0 or self.rejected > 0

    def to_dict(self) -> dict[str, Any]:
        return {"chosen": self.chosen, "corrected": self.corrected, "rejected": self.rejected}

    @classmethod
    def from_dict(cls, data: Any) -> "OutcomeCounts":
        if not isinstance(data, dict):
            return cls()
        counts = cls(
            chosen=_clean_count(data.get("chosen")),
            corrected=_clean_count(data.get("corrected")),
            rejected=_clean_count(data.get("rejected")),
        )
        # A file claiming more corrections than chosen words describes
        # something that cannot have happened, so the impossible half is
        # dropped rather than allowed to produce a negative weight.
        counts.corrected = min(counts.corrected, counts.chosen)
        return counts


@dataclass
class ConfidenceCounts:
    """How one confidence category turned out, among the words a person saw."""

    reviewed: int = 0
    corrected: int = 0

    @property
    def correction_rate(self) -> float | None:
        if self.reviewed <= 0:
            return None
        return self.corrected / self.reviewed

    def to_dict(self) -> dict[str, Any]:
        return {"reviewed": self.reviewed, "corrected": self.corrected}

    @classmethod
    def from_dict(cls, data: Any) -> "ConfidenceCounts":
        if not isinstance(data, dict):
            return cls()
        counts = cls(
            reviewed=_clean_count(data.get("reviewed")),
            corrected=_clean_count(data.get("corrected")),
        )
        counts.corrected = min(counts.corrected, counts.reviewed)
        return counts


@dataclass
class ProviderRecord:
    """Everything known about one service."""

    provider: Provider
    overall: OutcomeCounts = field(default_factory=OutcomeCounts)
    dimensions: dict[str, OutcomeCounts] = field(default_factory=dict)
    speaker_mistakes: int = 0
    """How often a person changed the speaker this service gave a word.

    One total, with no labels; the module docstring says why. Kept apart from the word counts because a diarisation mistake is a
    different kind of mistake. Section 23 names repeated speaker mistakes as
    their own category, and folding them in with misheard words would make
    both numbers mean less than they do now.
    """

    def counts_for(self, key: str) -> OutcomeCounts:
        """The bucket under ``key``, creating it if this is the first word in it."""
        found = self.dimensions.get(key)
        if found is None:
            found = OutcomeCounts()
            self.dimensions[key] = found
        return found

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall.to_dict(),
            "dimensions": {key: counts.to_dict() for key, counts in self.dimensions.items()},
            "speaker_mistakes": self.speaker_mistakes,
        }

    @classmethod
    def from_dict(cls, provider: Provider, data: Any) -> "ProviderRecord":
        record = cls(provider=provider)
        if not isinstance(data, dict):
            return record
        record.overall = OutcomeCounts.from_dict(data.get("overall"))
        raw_dimensions = data.get("dimensions")
        if isinstance(raw_dimensions, dict):
            for key, counts in raw_dimensions.items():
                if isinstance(key, str) and key.strip() and not key.startswith(_SPEAKER_PREFIX):
                    record.dimensions[key] = OutcomeCounts.from_dict(counts)
        raw_speakers = data.get("speaker_mistakes")
        if isinstance(raw_speakers, dict):
            # An older file kept the mistakes by speaker label. The total is
            # kept and the labels are dropped, so the next save writes none.
            record.speaker_mistakes = sum(_clean_count(count) for count in raw_speakers.values())
        else:
            record.speaker_mistakes = _clean_count(raw_speakers)
        return record


@dataclass(frozen=True)
class ConfidenceReliability:
    """What one confidence category turned out to be worth, in plain numbers."""

    category: Confidence
    reviewed: int
    corrected: int

    @property
    def correction_rate(self) -> float | None:
        if self.reviewed <= 0:
            return None
        return self.corrected / self.reviewed

    @property
    def summary(self) -> str:
        """One sentence a person can read, with the sample size in it."""
        if self.reviewed <= 0:
            return (
                f"{self.category.display_name}: no words in this category have been "
                "reviewed yet, so there is nothing to check the category against."
            )
        return (
            f"{self.category.display_name}: {_words(self.corrected)} out of "
            f"{_words(self.reviewed)} looked at needed changing "
            f"({_percentage(self.corrected, self.reviewed)})."
        )


@dataclass
class ProviderStatistics:
    """The whole record: what each service did, and how the categories held up.

    This is the object the pipeline updates, the store writes, and the
    statistics page reads. It counts and it reports. It never decides
    anything on its own, which is why every judgement it can offer comes out
    as a number with the evidence behind it attached.
    """

    providers: dict[Provider, ProviderRecord] = field(default_factory=dict)
    confidence: dict[Confidence, ConfidenceCounts] = field(default_factory=dict)

    # -- Recording what happened ----------------------------------------

    def record_for(self, provider: Provider) -> ProviderRecord:
        """The record for one service, creating it the first time it is used."""
        found = self.providers.get(provider)
        if found is None:
            found = ProviderRecord(provider=provider)
            self.providers[provider] = found
        return found

    # Every method here takes an ``amount``. A negative amount takes back a
    # result counted earlier, which is how a word whose result changed has
    # its old result removed before the new one goes in. No count goes
    # below zero: a file that lost a count in a race between two windows
    # must not end up describing a negative number of words.

    def record_choice(
        self, observation: Observation, corrected: bool = False, amount: int = 1
    ) -> None:
        """Note that this service's candidate became the final word.

        ``corrected`` says whether a person then changed it. Both cases are
        recorded, because a weight needs the words that were right as much
        as the ones that were wrong: without them there is a count of
        mistakes and no idea what to divide it by.
        """
        record = self.record_for(observation.provider)
        _add_choice(record.overall, corrected, amount)
        for key in observation.dimension_keys():
            _add_choice(record.counts_for(key), corrected, amount)

    def record_rejection(self, observation: Observation, amount: int = 1) -> None:
        """Note that this service's candidate lost to another one."""
        record = self.record_for(observation.provider)
        record.overall.rejected = max(0, record.overall.rejected + amount)
        for key in observation.dimension_keys():
            counts = record.counts_for(key)
            counts.rejected = max(0, counts.rejected + amount)

    def record_speaker_correction(self, provider: Provider, amount: int = 1) -> None:
        """Note that a person changed who this service said was talking."""
        record = self.record_for(provider)
        record.speaker_mistakes = max(0, record.speaker_mistakes + amount)

    def record_review(self, category: Confidence, corrected: bool, amount: int = 1) -> None:
        """Note that a person looked at a word in this confidence category."""
        counts = self.confidence.get(category)
        if counts is None:
            counts = ConfidenceCounts()
            self.confidence[category] = counts
        counts.reviewed = max(0, counts.reviewed + amount)
        if corrected:
            counts.corrected = max(0, counts.corrected + amount)
        counts.corrected = min(counts.corrected, counts.reviewed)

    # -- Reading it back ------------------------------------------------

    def counts_for(
        self,
        provider: Provider,
        dimension: Dimension | None = None,
        value: str | None = None,
    ) -> OutcomeCounts:
        """The counts for one service, overall or in one bucket.

        A bucket nothing has ever landed in comes back as an empty set of
        counts rather than as ``None``, so a caller never has to decide what
        an absent bucket means: it means no evidence, and empty counts say
        exactly that.
        """
        record = self.providers.get(provider)
        if record is None:
            return OutcomeCounts()
        if dimension is None:
            return record.overall
        key = dimension_key(dimension, value if value is not None else _YES)
        return record.dimensions.get(key, OutcomeCounts())

    def weight_for(
        self,
        provider: Provider,
        dimension: Dimension | None = None,
        value: str | None = None,
    ) -> float:
        """How far to believe one service, overall or in one bucket."""
        return self.counts_for(provider, dimension, value).weight

    def weight_for_word(self, observation: Observation) -> float:
        """One weight for one word, from the most specific evidence there is.

        The dimensions are tried narrowest first, and the first that has
        :data:`MINIMUM_DIMENSION_EVIDENCE` words behind it is used. The
        alternative, multiplying the weights of every dimension a word
        belongs to, would be arithmetic rather than evidence: the dimensions
        overlap heavily, so a German name that is also one of the user's
        terms would be marked down three times for what is one observation.
        """
        record = self.providers.get(observation.provider)
        if record is None:
            return DEFAULT_WEIGHT
        for key in observation.dimension_keys():
            counts = record.dimensions.get(key)
            if counts is not None and counts.chosen >= MINIMUM_DIMENSION_EVIDENCE:
                return counts.weight
        return record.overall.weight

    def repeated_speaker_mistakes(
        self, minimum: int = REPEATED_MISTAKE_THRESHOLD
    ) -> dict[Provider, int]:
        """How many speaker mistakes each service has made, where it keeps happening."""
        return {
            provider: record.speaker_mistakes
            for provider, record in self.providers.items()
            if record.speaker_mistakes >= minimum
        }

    def repeated_numeric_mistakes(
        self, minimum: int = REPEATED_MISTAKE_THRESHOLD
    ) -> dict[Provider, int]:
        """How many numbers each service has had corrected, where it keeps happening."""
        found: dict[Provider, int] = {}
        for provider in self.providers:
            counts = self.counts_for(provider, Dimension.NUMERIC)
            if counts.corrected >= minimum:
                found[provider] = counts.corrected
        return found

    def confidence_reliability(self) -> list[ConfidenceReliability]:
        """How each confidence category turned out, in the order they are declared."""
        return [
            ConfidenceReliability(
                category=category,
                reviewed=self.confidence.get(category, ConfidenceCounts()).reviewed,
                corrected=self.confidence.get(category, ConfidenceCounts()).corrected,
            )
            for category in Confidence
        ]

    @property
    def total_words(self) -> int:
        """Every word any service has been believed for."""
        return sum(record.overall.chosen for record in self.providers.values())

    def summary_sentences(self) -> list[str]:
        """The whole record as sentences, each carrying the evidence behind it.

        Every rate here is followed by the number of words it was worked out
        from. That is not politeness. A service that was right about three
        words out of three has a hundred per cent success rate and has told
        us nothing, and a reader shown only the rate has no way to know
        that.
        """
        sentences: list[str] = []
        if self.total_words <= 0:
            sentences.append(
                "Nothing has been learned yet. These figures fill in as you correct "
                "transcripts, and until then every service is treated as equally "
                f"reliable, at the starting weight of {DEFAULT_WEIGHT:.2f}."
            )
        else:
            sentences.append(
                "These figures come from the corrections you have made. A service's "
                f"weight starts at {DEFAULT_WEIGHT:.2f} and moves as evidence arrives, "
                f"and it takes about {int(PRIOR_STRENGTH)} words of evidence before it "
                "moves far, so a few corrections will not change it much."
            )
            for provider in Provider:
                record = self.providers.get(provider)
                if record is None or not record.overall.has_evidence:
                    continue
                sentences.append(describe_provider(record))
        for reliability in self.confidence_reliability():
            if reliability.reviewed > 0:
                sentences.append(reliability.summary)
        for provider, count in self.repeated_speaker_mistakes().items():
            sentences.append(
                f"{provider.display_name} has given the wrong speaker to a word "
                f"{_times(count)}."
            )
        for provider, count in self.repeated_numeric_mistakes().items():
            sentences.append(
                f"{provider.display_name} has had {_words(count)} containing numbers "
                "corrected, so its numbers are worth checking."
            )
        return sentences

    # -- On disk --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": CALIBRATION_FORMAT_VERSION,
            "providers": {
                provider.value: record.to_dict() for provider, record in self.providers.items()
            },
            "confidence": {
                category.value: counts.to_dict() for category, counts in self.confidence.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ProviderStatistics":
        """Build statistics from loaded JSON, ignoring anything unusable.

        A service or a category that cannot be understood is dropped on its
        own. Statistics are an accumulation of small facts, so losing one of
        them costs almost nothing, while refusing to load the file would
        throw away every correction the user has ever made.
        """
        statistics = cls()
        if not isinstance(data, dict):
            return statistics
        raw_providers = data.get("providers")
        if isinstance(raw_providers, dict):
            for name, record in raw_providers.items():
                provider = _provider_from_value(name)
                if provider is not None:
                    statistics.providers[provider] = ProviderRecord.from_dict(provider, record)
        raw_confidence = data.get("confidence")
        if isinstance(raw_confidence, dict):
            for name, counts in raw_confidence.items():
                category = _confidence_from_value(name)
                if category is not None:
                    statistics.confidence[category] = ConfidenceCounts.from_dict(counts)
        return statistics


class CalibrationStore:
    """Reads and writes the statistics file at a fixed location."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> ProviderStatistics:
        """Return the saved statistics, or an empty record if there are none usable."""
        data = read_json_object(self._path)
        if data is None:
            return ProviderStatistics()
        return ProviderStatistics.from_dict(data)

    def save(self, statistics: ProviderStatistics) -> bool:
        return write_json_object(self._path, statistics.to_dict())

    def apply(self, change: Callable[[ProviderStatistics], None]) -> bool:
        """Read the file fresh, make one change to it, and write it back.

        Every review window saves through this rather than through a copy it
        loaded earlier. Two windows that each loaded the file, changed their
        copy and saved it would each wipe out the other's figures; reading
        immediately before writing narrows that to the moment between the
        two, where losing one count is acceptable.
        """
        statistics = self.load()
        change(statistics)
        return self.save(statistics)


def reliability_weights(statistics: ProviderStatistics) -> ReliabilityWeights:
    """The reliability table reconcile uses, moved by what the statistics show.

    Reconcile starts each service at its own default reliability, which
    differ between services, while every learned weight here starts at the
    same :data:`DEFAULT_WEIGHT`. So the learned weight is used as a ratio to
    that starting point rather than in place of the default: a service whose
    weight has fallen ten per cent below where it started is believed ten
    per cent less than its default. Used directly, a learned 0.9 would mark
    OpenAI down from 1.0 before a single word of evidence had been seen.

    A service with no evidence keeps its default exactly, and the prior in
    :attr:`OutcomeCounts.weight` keeps a few corrections from moving any
    service far. Only each service's overall weight is used; the finer
    breakdown by language, names, numbers and vocabulary category is not.
    """
    by_provider = dict(DEFAULT_PROVIDER_RELIABILITY)
    for provider, record in statistics.providers.items():
        if record.overall.chosen <= 0:
            continue
        default = by_provider.get(provider, DEFAULT_RELIABILITY.unknown_provider)
        by_provider[provider] = default * record.overall.weight / DEFAULT_WEIGHT
    return ReliabilityWeights(by_provider=by_provider)


# -- Saying it in words --------------------------------------------------


def describe_provider(record: ProviderRecord) -> str:
    """One sentence about a service, with its sample size in it."""
    counts = record.overall
    name = record.provider.display_name
    if counts.chosen <= 0:
        if counts.rejected > 0:
            return (
                f"{name} has not been believed for any word yet, and {_words(counts.rejected)} "
                "of its suggestions were passed over, so there is nothing to judge it on."
            )
        return f"{name} has no record yet."
    return (
        f"{name} was believed for {_words(counts.chosen)}, and {_words(counts.corrected)} "
        f"of those needed correcting ({_percentage(counts.corrected, counts.chosen)}). "
        f"Its weight is {counts.weight:.2f}, from {evidence_phrase(counts.chosen)}."
    )


def evidence_phrase(chosen: int) -> str:
    """How much evidence a number rests on, said in words rather than shown as a bar.

    The wording matters more than it looks. A reader who is given a rate and
    no sense of how thin it is will believe the rate, so the strength of the
    evidence is stated as a phrase they cannot mistake for a decoration.
    """
    if chosen <= 0:
        return "no evidence at all"
    if chosen < MINIMUM_DIMENSION_EVIDENCE:
        return f"very little evidence, only {_words(chosen)}"
    if chosen < PRIOR_STRENGTH:
        return f"a little evidence, {_words(chosen)}"
    if chosen < PRIOR_STRENGTH * 10:
        return f"some evidence, {_words(chosen)}"
    return f"a good body of evidence, {_words(chosen)}"


def _words(count: int) -> str:
    return f"{count:,} word" if count == 1 else f"{count:,} words"


def _times(count: int) -> str:
    return "once" if count == 1 else f"{count:,} times"


def _percentage(part: int, whole: int) -> str:
    """A share, said as a share and never on its own.

    Every caller puts the counts beside this, which is the point: "3 per
    cent" is a claim, and "3 per cent of 400 words" is a measurement.
    """
    if whole <= 0:
        return "no words to work it out from"
    share = 100.0 * part / whole
    if 0 < share < 1:
        return "under 1 per cent"
    return f"{share:.0f} per cent"


# -- Reading loaded values -----------------------------------------------


def _add_choice(counts: OutcomeCounts, corrected: bool, amount: int = 1) -> None:
    counts.chosen = max(0, counts.chosen + amount)
    if corrected:
        counts.corrected = max(0, counts.corrected + amount)
    counts.corrected = min(counts.corrected, counts.chosen)


def _clean_count(value: Any, default: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return default
    return value


def _provider_from_value(value: Any) -> Provider | None:
    if not isinstance(value, str):
        return None
    try:
        return Provider(value.strip().lower())
    except ValueError:
        return None


def _confidence_from_value(value: Any) -> Confidence | None:
    if not isinstance(value, str):
        return None
    try:
        return Confidence(value.strip().lower())
    except ValueError:
        return None
