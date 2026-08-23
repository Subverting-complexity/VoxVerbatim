"""What language a stretch of the recording is in, when almost nothing says.

The specification asks for span-level language evidence. The services do not
supply it. What they actually offer, checked against what each of them
returns rather than against what would be convenient, is this:

* **ElevenLabs Scribe** reports one language for the whole file. There is no
  per-word or per-segment language field.
* **OpenAI** reports the languages it detected for the whole file, as a list.
* **Microsoft** reports a locale on *every phrase*. This is the only genuine
  per-segment language signal anywhere in the stack, and it is therefore
  worth far more than its share of the providers would suggest.
* **AssemblyAI** detects the language per request, so an escalation window
  cut around a disputed word gives a reading local to that window.

None of those is a per-word answer, so this module does not pretend to have
one. It combines weak signals instead. The whole-file readings become a
prior, which is what the evidence looks like where nothing local is known.
Microsoft's per-phrase locale is the strongest local signal there is.
Lexical evidence, from the small set of words that exist in only one of the
three languages, is the third. An escalation reading, where one has been
taken, sits alongside them.

The lexical lists are a heuristic and are written to look like one: short,
explicit, and made of function words rather than of vocabulary, because
function words are frequent, are rarely proper nouns, and are what a
code-switch actually turns over. Anything whose comparison form is not a
plain word is dropped, which quietly removes the number words: "een" is
Afrikaans for one and normalises to "1", and a lexicon that matched every
"1" in the transcript as Afrikaans would be worse than no lexicon at all.
A word that appears in more than one of the lists in play is dropped from
all of them, because by definition it distinguishes nothing.

**Afrikaans is not merely filtered out when it is disabled; it is never
considered.** Section 3 of the specification asks for that, and the reason
is worth stating because it is easy to implement the weaker version by
accident. Computing an Afrikaans score and then discarding it would leave
the lexical lists competing over words that German and Afrikaans share, and
a German span would go on looking uncertain because of a language the user
has already said is not in the recording. Removing the language removes that
whole class of wrong guesses: with Afrikaans out of the picture, a word the
two languages share becomes a perfectly good German marker.

The eligibility rule the specification asks for lives here as well.
Microsoft must not contribute evidence to a confidently Afrikaans span. At a
code-switch boundary, where the language is genuinely uncertain, it is
reduced rather than excluded, because a hard cut-off at an arbitrary
threshold would swing a service's whole opinion on a hundredth of a point of
language score. Where Afrikaans is disabled, Microsoft is simply valid
everywhere, which is what section 3 says.

Nothing here depends on Qt or on any provider library, so it can be tested
on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

from vox_verbatim.transcription.alignment import AlignedRow, AlignedTable
from vox_verbatim.transcription.model import (
    AudioSpan,
    Language,
    LanguageEvidence,
    Provider,
    ProviderResult,
    RecordingConfiguration,
)
from vox_verbatim.transcription.normalise import normalise

#: How sure the evidence must be before a span counts as confidently one
#: language. Above this, Microsoft is excluded from an Afrikaans span.
CONFIDENT_LANGUAGE_THRESHOLD = 0.75

#: Below this, the language of a span is genuinely in doubt, which is a
#: review reason when Afrikaans is enabled.
UNCERTAIN_LANGUAGE_THRESHOLD = 0.6

#: How much Afrikaans a span may carry before Microsoft starts to lose
#: weight. A trace of Afrikaans in the evidence is not a reason to discount
#: a service on an otherwise plainly English sentence.
MICROSOFT_FULL_WEIGHT_BELOW = 0.15

#: How far Microsoft can fall on an uncertain span without being excluded.
#: Not zero, because the span is not known to be Afrikaans; well under one,
#: because it might be.
MICROSOFT_MINIMUM_WEIGHT = 0.2


@dataclass(frozen=True)
class LanguageWeights:
    """How much each signal counts, and how far a signal reaches.

    These are the numbers to move when the calibration work reaches
    language. They are ratios rather than probabilities: what matters is
    that a Microsoft locale outweighs a lexical hit, and that both outweigh
    the whole-file prior, since the prior is the same everywhere and cannot
    tell one part of a recording from another.
    """

    whole_file: float = 1.0
    """What one service's whole-file reading contributes to the prior."""

    prior_share: float = 1.0
    """How far the prior counts against local evidence at a position."""

    microsoft_locale: float = 3.0
    """The only genuine per-segment signal, and weighted as such."""

    escalation: float = 2.5
    """A reading taken from a window cut around this part of the audio."""

    lexical: float = 2.0
    """A word that exists in only one of the languages in play."""

    neighbourhood: int = 4
    """How many words either side a local signal reaches."""

    neighbour_decay: float = 0.6
    """How much a signal fades with each word of distance.

    Language does not change from one word to the next, so evidence about a
    word is evidence about its neighbours. Without this, a sentence would
    look English at the three words that happen to be distinctively English
    and unknown everywhere else, and every gap between them would be
    reported as uncertain.
    """


DEFAULT_LANGUAGE_WEIGHTS = LanguageWeights()


# -- The lexicons --------------------------------------------------------
#
# Function words that belong to one of the three languages and not to the
# others. They are short lists on purpose. A long list would be a dictionary,
# and a dictionary of a language is not something anybody can check by eye or
# keep honest as it grows.

_ENGLISH_MARKERS = (
    "the", "this", "these", "those", "with", "which", "would", "could",
    "should", "about", "because", "people", "something", "anything",
    "always", "never", "right", "thank", "thanks", "please", "sorry",
    "understand", "question", "going", "doing", "having", "through",
    "though", "enough", "really", "actually", "maybe", "perhaps",
    "between", "before", "after", "there", "where", "when", "what",
)

_GERMAN_MARKERS = (
    "ich", "nicht", "auch", "aber", "sehr", "schon", "noch", "wenn",
    "dann", "weil", "immer", "etwas", "jetzt", "heute", "und", "oder",
    "machen", "sagen", "haben", "werden", "wollen", "können", "müssen",
    "über", "zwischen", "deutsch", "danke", "bitte", "gut", "vielleicht",
    "natürlich", "wirklich", "verstehen", "frage", "arbeit", "zeit",
    "welt", "möchte", "kein", "mein", "dein", "sein",
)

_AFRIKAANS_MARKERS = (
    "ek", "nie", "baie", "maar", "moet", "julle", "hulle", "ons", "jou",
    "jy", "dankie", "asseblief", "lekker", "sommer", "nogal", "hierdie",
    "daardie", "sodat", "want", "omdat", "gaan", "goed", "nou", "altyd",
    "miskien", "verstaan", "vraag", "tyd", "mense", "iets", "niks",
    "nog", "wees", "kry", "sien", "praat", "weet", "dink", "self",
)

_MARKERS: dict[Language, tuple[str, ...]] = {
    Language.ENGLISH: _ENGLISH_MARKERS,
    Language.GERMAN: _GERMAN_MARKERS,
    Language.AFRIKAANS: _AFRIKAANS_MARKERS,
}


@lru_cache(maxsize=8)
def _lexicon(allowed: tuple[Language, ...]) -> dict[str, Language]:
    """Build the word-to-language map for exactly the languages in play.

    Two things happen here that matter. Every entry goes through the same
    comparison form the transcript words will be in, so that "über" and
    "ueber" are one entry and not two misses. And an entry whose form is not
    a plain word is thrown away, which is what keeps the number words out:
    a lexicon containing "1" would report every numeral in the recording as
    evidence of a language.

    The map is built only from the allowed languages, so with Afrikaans
    disabled no Afrikaans word is ever looked at, and a word the two
    languages share becomes a usable German marker rather than a clash.
    """
    counted: dict[str, list[Language]] = {}
    for language in allowed:
        for word in _MARKERS.get(language, ()):
            form = normalise(word)
            if not form or not form.isalpha():
                continue
            counted.setdefault(form, []).append(language)
    return {form: languages[0] for form, languages in counted.items() if len(languages) == 1}


def allowed_languages(configuration: RecordingConfiguration) -> tuple[Language, ...]:
    """The languages this recording may contain, in a fixed order.

    Afrikaans appears only when the user has said it may occur. Everything
    else in this module works from this tuple, so switching it off here
    switches off every Afrikaans code path at once rather than leaving each
    of them to remember.
    """
    if configuration.afrikaans_enabled:
        return (Language.ENGLISH, Language.GERMAN, Language.AFRIKAANS)
    return (Language.ENGLISH, Language.GERMAN)


@dataclass(frozen=True)
class SpanLanguage:
    """A language reading taken from one region of the audio.

    This is how a localised opinion gets in from outside: the escalation
    stage sends a padded window to AssemblyAI, which detects the language of
    what it was sent, and that answer is about that window rather than about
    the whole recording.
    """

    span: AudioSpan
    language: Language
    provider: Provider | None = None
    strength: float = 1.0


@dataclass(frozen=True)
class LanguageReading:
    """The language evidence for every backbone word of one recording."""

    allowed: tuple[Language, ...]
    prior: LanguageEvidence
    by_position: tuple[LanguageEvidence, ...] = ()
    afrikaans_enabled: bool = False

    def for_position(self, position: int) -> LanguageEvidence:
        """The evidence at one backbone word, or the prior where there is none.

        Positions outside the recording fall back to the prior rather than
        raising, because an inserted word sits between two positions and a
        caller should not have to think about which side of the gap it is
        on to ask a question about language.
        """
        if not self.by_position:
            return self.prior
        if position < 0:
            return self.by_position[0]
        if position >= len(self.by_position):
            return self.by_position[-1]
        return self.by_position[position]

    def language_at(self, position: int) -> Language:
        """The most likely language at one word, or unknown if nothing is known."""
        evidence = self.for_position(position)
        return evidence.best if evidence.scores else Language.UNKNOWN

    def is_uncertain(self, position: int) -> bool:
        """Whether the language here is genuinely in doubt.

        Only ever true when Afrikaans is enabled. With two languages allowed
        and Microsoft valid on both of them, an even split between English
        and German changes nothing about who may speak or what may be
        believed, so putting it in front of a person would be asking them to
        confirm something the application was not going to act on. With
        Afrikaans enabled it does change something, which is why the
        specification makes it a review reason there and only there.
        """
        if not self.afrikaans_enabled:
            return False
        evidence = self.for_position(position)
        if not evidence.scores:
            return True
        return evidence.best_score < UNCERTAIN_LANGUAGE_THRESHOLD


def read_languages(
    table: AlignedTable,
    results: Iterable[ProviderResult],
    configuration: RecordingConfiguration,
    escalation_readings: Sequence[SpanLanguage] = (),
    weights: LanguageWeights = DEFAULT_LANGUAGE_WEIGHTS,
) -> LanguageReading:
    """Work out the language evidence at every backbone word.

    The prior is built once from the whole-file readings. The local evidence
    is gathered per word, smoothed across its neighbours because language
    does not change word by word, and then added to the prior. Where a word
    has no local evidence of its own and none near it, what comes back is
    the prior, which is the honest answer: we know what the recording is in
    and nothing more about this particular word.
    """
    allowed = allowed_languages(configuration)
    lexicon = _lexicon(allowed)
    gathered = list(results)
    prior = _whole_file_prior(gathered, allowed, weights)

    count = len(table.backbone_tokens)
    local = [dict.fromkeys(allowed, 0.0) for _ in range(count)]
    for row in table.rows:
        if row.is_insertion_row or not 0 <= row.position < count:
            continue
        _add_microsoft_locale(local[row.position], row, allowed, weights)
        _add_lexical(local[row.position], row, lexicon, weights)
    _add_escalations(local, table, escalation_readings, allowed, weights)

    smoothed = _smooth(local, allowed, weights)
    by_position = tuple(
        _as_evidence(scores, prior, allowed, weights) for scores in smoothed
    )
    return LanguageReading(
        allowed=allowed,
        prior=_normalised(prior, allowed),
        by_position=by_position,
        afrikaans_enabled=configuration.afrikaans_enabled,
    )


def _whole_file_prior(
    results: Sequence[ProviderResult],
    allowed: tuple[Language, ...],
    weights: LanguageWeights,
) -> dict[Language, float]:
    """Turn every service's reading of the whole file into one starting point.

    A service that reports a language outside the allowed set is ignored
    rather than argued with. When Afrikaans is disabled that is the rule
    doing real work: a service guessing Afrikaans on a German recording is
    exactly the wrong answer the user has already ruled out, and the way to
    honour that is to give the guess nowhere to go.
    """
    scores = dict.fromkeys(allowed, 0.0)
    for result in results:
        if not result.succeeded:
            continue
        language = result.detected_language
        if language in scores:
            scores[language] += weights.whole_file
    return scores


def _add_microsoft_locale(
    scores: dict[Language, float],
    row: AlignedRow,
    allowed: tuple[Language, ...],
    weights: LanguageWeights,
) -> None:
    """Add the locale Microsoft put on the phrase this word came from.

    Only Microsoft's per-token language is read. The other services set the
    same value on every word they return, because it is the file-level
    answer copied down, and counting it here would be counting the prior a
    second time under a different name at every word in the recording.
    """
    column = row.column_for(Provider.MICROSOFT)
    if column is None:
        return
    for token in column.aligned_tokens:
        if token.language in scores and token.language in allowed:
            scores[token.language] += weights.microsoft_locale


def _add_lexical(
    scores: dict[Language, float],
    row: AlignedRow,
    lexicon: dict[str, Language],
    weights: LanguageWeights,
) -> None:
    """Add what the word itself says, if it is a word only one language has.

    Every service's reading of this word is looked at, but each language is
    credited once. Three services agreeing that the word was "nicht" is one
    piece of evidence about the language, not three: they were all listening
    to the same sound.
    """
    texts: list[str] = []
    if row.backbone_token is not None:
        texts.append(row.backbone_token.text)
    for column in row.columns:
        texts.extend(token.text for token in column.aligned_tokens)

    seen: set[Language] = set()
    for text in texts:
        language = lexicon.get(normalise(text))
        if language is not None and language in scores and language not in seen:
            scores[language] += weights.lexical
            seen.add(language)


def _add_escalations(
    local: Sequence[dict[Language, float]],
    table: AlignedTable,
    readings: Sequence[SpanLanguage],
    allowed: tuple[Language, ...],
    weights: LanguageWeights,
) -> None:
    """Apply readings taken from windows of audio to the words inside them."""
    if not readings:
        return
    for index, token in enumerate(table.backbone_tokens):
        if token.start is None or token.end is None:
            continue
        span = AudioSpan(token.start, token.end)
        for reading in readings:
            if reading.language not in allowed or not span.overlaps(reading.span):
                continue
            local[index][reading.language] += weights.escalation * reading.strength


def _smooth(
    local: Sequence[dict[Language, float]],
    allowed: tuple[Language, ...],
    weights: LanguageWeights,
) -> list[dict[Language, float]]:
    """Let each word's evidence reach the words around it, fading with distance."""
    reach = max(0, weights.neighbourhood)
    smoothed: list[dict[Language, float]] = []
    for index in range(len(local)):
        totals = dict.fromkeys(allowed, 0.0)
        start = max(0, index - reach)
        end = min(len(local), index + reach + 1)
        for other in range(start, end):
            factor = weights.neighbour_decay ** abs(other - index)
            for language, value in local[other].items():
                if value:
                    totals[language] += value * factor
        smoothed.append(totals)
    return smoothed


def _as_evidence(
    scores: dict[Language, float],
    prior: dict[Language, float],
    allowed: tuple[Language, ...],
    weights: LanguageWeights,
) -> LanguageEvidence:
    """Mix the local evidence with the prior and turn it into shares of one."""
    prior_scores = _normalised(prior, allowed).scores
    combined = {
        language: scores.get(language, 0.0)
        + prior_scores.get(language, 0.0) * weights.prior_share
        for language in allowed
    }
    return _normalised(combined, allowed)


def _normalised(scores: dict[Language, float], allowed: tuple[Language, ...]) -> LanguageEvidence:
    """Turn raw counts into shares of one, over the allowed languages only.

    With nothing to go on, the languages share the total equally. That is
    the honest reading of no evidence, and it falls below the uncertainty
    threshold, so a recording nothing could be said about is put in front of
    a person rather than assumed to be English.
    """
    total = sum(max(0.0, scores.get(language, 0.0)) for language in allowed)
    if total <= 0.0:
        share = 1.0 / len(allowed) if allowed else 0.0
        return LanguageEvidence({language: share for language in allowed})
    return LanguageEvidence(
        {language: max(0.0, scores.get(language, 0.0)) / total for language in allowed}
    )


def provider_language_weight(
    provider: Provider,
    evidence: LanguageEvidence,
    afrikaans_enabled: bool,
) -> float:
    """How far this service may be believed about a word in this language.

    Only one service is affected, and only in one direction. Microsoft
    MAI-Transcribe does not handle Afrikaans, so it must not contribute
    evidence to a span confidently identified as Afrikaans; the
    specification says excluded or heavily down-weighted, and excluded is
    the honest reading when the span is confidently Afrikaans.

    The interesting case is the boundary. A word halfway through a switch
    into Afrikaans is not confidently anything, and a hard cut-off would
    make a service's entire opinion depend on which side of an arbitrary
    number the score landed. So the weight falls off gradually as Afrikaans
    becomes more likely, from full weight while Afrikaans is only a trace to
    :data:`MICROSOFT_MINIMUM_WEIGHT` just short of confidence. That is what
    section 10 means by eligibility depending on language confidence rather
    than on a hard decision.

    With Afrikaans disabled this returns full weight always, because section
    3 says to treat Microsoft as valid wherever English or German evidence
    is relevant, and with Afrikaans disabled that is everywhere.
    """
    if provider is not Provider.MICROSOFT or not afrikaans_enabled:
        return 1.0
    afrikaans = evidence.score_for(Language.AFRIKAANS)
    if evidence.is_confidently(Language.AFRIKAANS, CONFIDENT_LANGUAGE_THRESHOLD):
        return 0.0
    if afrikaans <= MICROSOFT_FULL_WEIGHT_BELOW:
        return 1.0
    spread = CONFIDENT_LANGUAGE_THRESHOLD - MICROSOFT_FULL_WEIGHT_BELOW
    travelled = min(1.0, (afrikaans - MICROSOFT_FULL_WEIGHT_BELOW) / spread)
    return 1.0 - (1.0 - MICROSOFT_MINIMUM_WEIGHT) * travelled


def lexical_language(text: str, allowed: Sequence[Language]) -> Language | None:
    """The language this single word belongs to, if it belongs to only one.

    Exposed so that the lexical heuristic can be inspected and tested
    directly rather than only through its effect on a whole recording.
    """
    return _lexicon(tuple(allowed)).get(normalise(text))
