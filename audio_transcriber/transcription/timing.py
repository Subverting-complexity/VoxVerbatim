"""When a reconciled word may keep the time of the word it replaced.

Only one service in the ensemble measures where each word falls in the
audio. Reconciliation then changes some of those words: a spelling is
corrected, two words become one, one word becomes two, a word nobody timed
is added. Every one of those changes raises the same question, and this
module is the one place that answers it. May this word keep the span the
service measured, and if not, what may it honestly claim instead?

The answers are the timing states in the specification, and they exist
because the easy alternative is a lie. "Jurgen" corrected to "Jürgen" is the
same sound in the same place, so it keeps the measured span and the
transcript is none the worse. "data base" joined into "database" honestly
covers both spans, because the outer edges of the pair were both measured.
But "cannot" split into "can not" is different in kind: the two words
together cover a span that was measured, and where inside it the join falls
was never measured by anybody. Splitting the span down the middle would
produce two precise-looking numbers that nothing in the recording supports,
and every part of the application that trusts a timestamp would then be
trusting an invention. So a split shares one span, says that it is shared,
and refuses to guess.

That refusal is the rule this module exists to enforce, and it is the rule
most likely to be broken by well-meaning code. The specification states it
three times. **When forced alignment cannot be run, or is run and fails, the
words keep the best span already known and are marked approximate or
uncertain. Precise word boundaries are never invented.** Any change here
that produces a number nobody measured is a bug, however reasonable it looks
while it is being written.

Measuring again is what forced alignment is for, and it runs selectively
rather than over everything. Running it on words that did not change would
cost a request per phrase to confirm times that were already right. It runs
where the text materially changed, which is exactly where the inherited span
has stopped describing what the words now say.

It runs through the :class:`ForcedAligner` interface and never against one
service directly, because the service that does this today is not the one
that will do it forever. Today's cannot hear Afrikaans at all. An Afrikaans
span therefore keeps the backbone timing it inherited, falls back to a
phrase-level span where the word boundaries are not defensible, and says
that it is uncertain. That is not a limitation being worked around; it is
the correct answer, and it leaves the door open for an Afrikaans-capable
aligner to be dropped in later with nothing else changing.

One number in the aligner's answer is worth knowing about before reading the
code. Transcription reports a log probability per word, where higher is
better. Forced alignment reports a **loss**, where **lower** is better. They
are both small floating-point numbers, they sit in the same place in the
response, and reading one as the other would quietly mark the worst
alignments as the best. :func:`status_for_loss` is where the polarity lives,
it has a test of its own, and it is not a line to tidy up.

Nothing here depends on Qt or on any provider library, so it can be tested
on its own.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from audio_transcriber.transcription.model import (
    AlignmentStatus,
    AudioSpan,
    Confidence,
    FinalToken,
    Language,
    Provider,
    ProviderToken,
    ReviewReason,
    TimingStatus,
    TokenReference,
)
from audio_transcriber.transcription.normalise import EquivalenceKind, equivalence_kind
from audio_transcriber.transcription.providers.base import ForcedAligner, ProviderError

_log = logging.getLogger(__name__)

#: How large a per-word alignment loss may be and still count as a
#: measurement. Lower is better, so this is a ceiling and not a floor. The
#: figure is a starting point rather than a finding: it should be calibrated
#: against spans a person has confirmed, and until it has been it errs
#: towards calling a span approximate.
DEFAULT_MAXIMUM_ALIGNMENT_LOSS = 0.35

#: The timing states that mean the text has moved away from what was
#: measured, and so are worth measuring again where that is possible.
_WORTH_REALIGNING = frozenset(
    {
        TimingStatus.MAPPED_SUBSTITUTION,
        TimingStatus.SHARED_PHRASE_SPAN,
        TimingStatus.APPROXIMATE_SPAN,
        TimingStatus.UNALIGNED,
        TimingStatus.UNCERTAIN,
    }
)


@dataclass(frozen=True)
class TimingOptions:
    """What a person can change about how timing is decided."""

    forced_alignment_enabled: bool = True
    maximum_alignment_loss: float = DEFAULT_MAXIMUM_ALIGNMENT_LOSS

    @classmethod
    def from_settings(cls, processing: Any = None) -> "TimingOptions":
        """Read the options off the processing settings, without importing them.

        Attributes are read by name, so this module stays testable on its own
        and a change to the shape of the settings does not reach in here.
        """
        enabled = getattr(processing, "forced_alignment_enabled", None)
        if not isinstance(enabled, bool):
            enabled = cls.forced_alignment_enabled
        return cls(forced_alignment_enabled=enabled)


@dataclass(frozen=True)
class AlignedWord:
    """One word as the forced aligner measured it, in canonical time.

    ``loss`` is how badly the word fitted the audio, so a small number is a
    good one. It is optional because the aligner interface does not require
    it: a service that reports no per-word figure has still measured the
    word, and the absence of a complaint is not a complaint.
    """

    text: str
    start: float
    end: float
    loss: float | None = None

    @property
    def span(self) -> AudioSpan:
        return AudioSpan(self.start, max(self.start, self.end))


@dataclass(frozen=True)
class TimingOutcome:
    """What one final word is entitled to say about when it was spoken.

    ``span`` is the word's own start and end, and is ``None`` when nobody
    measured them. ``containing_span`` is the wider region the word is known
    to sit somewhere inside, which is what lets a person hear an unaligned
    word without the application pretending to know where it begins.
    """

    status: TimingStatus
    span: AudioSpan | None = None
    containing_span: AudioSpan | None = None
    source: Provider | None = None
    confidence: Confidence = Confidence.UNRESOLVED
    review_reasons: tuple[ReviewReason, ...] = ()
    shares_span: bool = False
    """Whether this span is shared with the words beside it, boundary unknown."""

    @property
    def is_exact(self) -> bool:
        """Whether this span may be used for word-level navigation."""
        return self.status.is_exact

    @property
    def may_enter_transcript(self) -> bool:
        """Whether this word may appear in the verbatim transcript as it stands.

        A word with no acoustic evidence at all may not. The specification is
        explicit: it has to be verified as actually spoken first, and a
        person doing that is what turns it from an invention into a word.
        """
        return self.status is not TimingStatus.UNALIGNED


# -- Mapping the timing of words that changed ----------------------------


def map_timing(
    final_texts: Sequence[str],
    sources: Sequence[ProviderToken],
    containing_span: AudioSpan | None = None,
    confident: bool = True,
) -> list[TimingOutcome]:
    """Decide what the reconciled words may claim, given what was measured.

    One call covers every case in the specification, because they are all the
    same question asked of different counts. One word against one measured
    word is an exact match, a formatting change or a substitution, depending
    on how far the text moved. Several measured words against one final word
    is a merge, and the combined span is honest because both of its outer
    edges were measured. One measured word against several final words is a
    split, and every one of them gets the same span, marked as shared,
    because the join between them was never measured and this module will not
    invent it. No measured words at all is an added word, which is unaligned
    and must be verified before it can enter the transcript. No final words
    at all is a removed word, which returns nothing at all: it belongs in
    provenance, and :func:`note_removed_word` is what puts it there.

    ``confident`` says whether reconciliation was sure of a substitution. An
    unsure one still inherits the span, because there is nothing better to
    give it, but it is flagged so that a person sees it.
    """
    texts = [text for text in final_texts]
    timed = [source for source in sources if source.has_timing]
    region = _combined_span(timed) or containing_span

    if not texts:
        # A removed word has no timing to decide, because it has no word in
        # the transcript to carry one.
        return []
    if not timed:
        return [_unaligned(text, region) for text in texts]

    source_provider = timed[0].provider
    if len(texts) == 1 and len(timed) == 1:
        return [_one_to_one(texts[0], timed[0], confident)]
    if len(texts) == 1:
        return [_merged(texts[0], timed, source_provider)]
    return _shared(texts, region, source_provider)


def _one_to_one(text: str, source: ProviderToken, confident: bool) -> TimingOutcome:
    """One final word standing where one measured word stood.

    How far the text moved decides everything. Identical text keeps the
    measured time outright. A difference only in how the word is written
    leaves the span fitting exactly as well as it did, because the sound did
    not change. A genuinely different word inherits the span, which is the
    best available answer and is honestly not a measurement of this word, so
    it is marked as inherited rather than exact.
    """
    span = AudioSpan(source.start or 0.0, source.end or 0.0)
    kind = equivalence_kind(text, source.text)
    if kind is EquivalenceKind.IDENTICAL:
        return TimingOutcome(
            status=TimingStatus.EXACT_PROVIDER_TIME,
            span=span,
            containing_span=span,
            source=source.provider,
            confidence=Confidence.HIGH,
        )
    if kind.is_equivalent:
        return TimingOutcome(
            status=TimingStatus.MAPPED_FORMAT_EQUIVALENT,
            span=span,
            containing_span=span,
            source=source.provider,
            confidence=Confidence.HIGH,
        )
    return TimingOutcome(
        status=TimingStatus.MAPPED_SUBSTITUTION,
        span=span,
        containing_span=span,
        source=source.provider,
        confidence=Confidence.REVIEW_SUGGESTED,
        review_reasons=() if confident else (ReviewReason.WEAK_ALIGNMENT,),
    )


def _merged(text: str, timed: Sequence[ProviderToken], provider: Provider) -> TimingOutcome:
    """Several measured words that became one word.

    The combined span is defensible in a way a split's boundary is not: the
    beginning of the first word and the end of the last were both measured,
    and the new word covers exactly that stretch of audio. Where the joined
    text says the same thing as the words it replaced, that is a formatting
    change and the span is as good as an exact one. Where it does not, it is
    a substitution over a wider span.
    """
    span = _combined_span(timed)
    assert span is not None  # every token here has timing, so there is a span
    joined = " ".join(source.text for source in timed)
    kind = equivalence_kind(text, joined)
    if kind.is_equivalent:
        return TimingOutcome(
            status=TimingStatus.MAPPED_FORMAT_EQUIVALENT,
            span=span,
            containing_span=span,
            source=provider,
            confidence=Confidence.HIGH,
        )
    return TimingOutcome(
        status=TimingStatus.MAPPED_SUBSTITUTION,
        span=span,
        containing_span=span,
        source=provider,
        confidence=Confidence.REVIEW_SUGGESTED,
    )


def _shared(
    texts: Sequence[str], region: AudioSpan | None, provider: Provider
) -> list[TimingOutcome]:
    """Several final words covering one measured span, boundaries unknown.

    Every word gets the whole span, identically. That is the point: nobody
    measured where one word ends and the next begins, so any two different
    numbers here would be a guess presented as a measurement. The words are
    marked as sharing their span, which is what tells the review window to
    play the phrase rather than pretending it can play one word of it.
    """
    if region is None:
        return [_unaligned(text, None) for text in texts]
    return [
        TimingOutcome(
            status=TimingStatus.SHARED_PHRASE_SPAN,
            span=region,
            containing_span=region,
            source=provider,
            confidence=Confidence.REVIEW_SUGGESTED,
            shares_span=True,
        )
        for _ in texts
    ]


def _unaligned(text: str, region: AudioSpan | None) -> TimingOutcome:
    """A word with no acoustic evidence behind it at all.

    It keeps the region it is believed to sit inside, so that it can be
    listened to, and it claims no span of its own. It is also flagged,
    because a word that no service heard has to be confirmed as spoken before
    it is allowed to stand in a verbatim transcript.
    """
    _log.debug("The word %r has no acoustic evidence and is left unaligned.", text)
    return TimingOutcome(
        status=TimingStatus.UNALIGNED,
        span=None,
        containing_span=region,
        source=None,
        confidence=Confidence.UNRESOLVED,
        review_reasons=(ReviewReason.UNALIGNED_WORD,),
    )


def _combined_span(sources: Sequence[ProviderToken]) -> AudioSpan | None:
    timed = [source for source in sources if source.has_timing]
    if not timed:
        return None
    start = min(source.start for source in timed if source.start is not None)
    end = max(source.end for source in timed if source.end is not None)
    return AudioSpan(start, max(start, end))


def apply_timing(token: FinalToken, outcome: TimingOutcome) -> None:
    """Write a timing decision onto a word, and nothing else.

    It touches the timing fields and the review flags and leaves the text and
    the speaker exactly as they were. The three answers have separate sources
    and separate confidences, and a function that quietly adjusted one while
    settling another would undo the whole design.
    """
    span = outcome.span
    token.start = None if span is None else span.start
    token.end = None if span is None else span.end
    token.timing_status = outcome.status
    token.timing_confidence = outcome.confidence
    token.timing_source = outcome.source
    if outcome.containing_span is not None:
        token.source_audio_span = outcome.containing_span
    for reason in outcome.review_reasons:
        token.flag(reason)


def note_removed_word(token: FinalToken, source: ProviderToken | TokenReference) -> None:
    """Keep a word reconciliation threw out, without putting it in the text.

    A service that hallucinated a word leaves its evidence here, on the word
    beside where it would have gone. Nothing shows it to a reader, and a
    pattern of one service inventing words can still be found afterwards,
    which is the whole reason provenance is never destroyed.
    """
    reference = (
        source
        if isinstance(source, TokenReference)
        else TokenReference(provider=source.provider, index=source.index)
    )
    if reference not in token.rejected_tokens:
        token.rejected_tokens.append(reference)


def may_enter_transcript(token: FinalToken) -> bool:
    """Whether this word may stand in the verbatim transcript as it is.

    An unaligned word may not, until a person has confirmed that it was
    spoken. Everything else may: an approximate span is a real region of real
    audio, and only a word with nothing behind it at all is an invention.
    """
    if token.timing_status is not TimingStatus.UNALIGNED:
        return True
    return token.human_corrected


# -- Deciding when to measure again --------------------------------------


def needs_forced_alignment(
    token: FinalToken,
    settled_dispute: bool = False,
    options: TimingOptions | None = None,
) -> bool:
    """Whether this word's timing is worth measuring against the audio again.

    Not every word, and deliberately so. A word that came back the same from
    every service, at a time one of them measured, has nothing to gain from
    being measured a second time and would cost a request to confirm what is
    already known. The words worth the trouble are the ones whose text has
    materially moved away from what was measured: a substitution, a merge or
    split sharing one span, a span already known to be approximate, and a
    word with no evidence at all that a measurement might yet find.

    ``settled_dispute`` covers the remaining case from the specification: a
    high-value phrase that escalation or adjudication has just settled, where
    precise timing is wanted because somebody is going to look at it.
    """
    options = options or TimingOptions()
    if not options.forced_alignment_enabled:
        return False
    if settled_dispute:
        return True
    if token.timing_status in _WORTH_REALIGNING:
        return True
    return token.alignment_status in (AlignmentStatus.MERGE, AlignmentStatus.SPLIT)


def status_for_loss(loss: float | None, options: TimingOptions | None = None) -> TimingStatus:
    """Turn an alignment loss into a verdict on the span it came with.

    **Lower is better.** This is the opposite polarity to the log probability
    a transcription reports, and the two numbers look alike enough that
    reading one as the other is easy and silent: the worst alignments would
    be marked as the best, every span would look measured, and nothing would
    ever fail visibly. A small loss earns ``forced_aligned``; a large one
    earns ``approximate_span``, which says that the region is right and the
    edges are not to be trusted.

    A missing loss is not a bad one. A service that reports no per-word
    figure has still measured the word, so the measurement stands.
    """
    options = options or TimingOptions()
    if loss is None:
        return TimingStatus.FORCED_ALIGNED
    if loss <= options.maximum_alignment_loss:
        return TimingStatus.FORCED_ALIGNED
    return TimingStatus.APPROXIMATE_SPAN


def read_aligned_words(rows: Iterable[Any]) -> list[AlignedWord]:
    """Read whatever an aligner returned into words with an optional loss.

    The interface every aligner implements returns a text, a start and an
    end. An aligner that also reports how well each word fitted may return a
    fourth value, or an object or mapping with a ``loss`` on it, and that
    figure is worth having because it is what separates a measured span from
    an approximate one. Reading all of those shapes here means a better
    aligner can be dropped in without the interface changing under everyone
    who already implements it.
    """
    words: list[AlignedWord] = []
    for row in rows:
        if isinstance(row, AlignedWord):
            words.append(row)
            continue
        if isinstance(row, (tuple, list)):
            text, start, end = row[0], row[1], row[2]
            loss = row[3] if len(row) > 3 else None
        elif isinstance(row, dict):
            text, start, end = row.get("text", ""), row.get("start"), row.get("end")
            loss = row.get("loss")
        else:
            text = getattr(row, "text", "")
            start, end = getattr(row, "start", None), getattr(row, "end", None)
            loss = getattr(row, "loss", None)
        if start is None or end is None:
            # A word the aligner did not place is exactly what the call was
            # made to obtain, so an unplaced one is dropped rather than given
            # a position it never had.
            continue
        words.append(
            AlignedWord(
                text=str(text),
                start=float(start),
                end=float(end),
                loss=None if loss is None else float(loss),
            )
        )
    return words


def realign(
    tokens: Sequence[FinalToken],
    aligner: ForcedAligner | None,
    audio_path: Path,
    canonical_offset: float = 0.0,
    language: Language = Language.UNKNOWN,
    options: TimingOptions | None = None,
) -> list[TimingOutcome]:
    """Measure a phrase against its audio again, or say honestly why not.

    The words are handed over as plain running text, in order, because that
    is what an aligner takes and because the verbatim words are the ones the
    recording actually contains. The times that come back are already
    canonical, since the aligner is told the offset of the audio it is given.

    Four things can happen, and only one of them produces new numbers.

    The alignment can run and succeed, in which case each word gets the span
    that was measured for it, and the loss decides whether that counts as a
    measurement or only as the right region.

    The aligner may not handle the language, which today means Afrikaans.
    Then the words keep the backbone timing they already had, fall back to
    the phrase span where their own boundaries are not defensible, and are
    marked uncertain. Nothing is invented and nothing is thrown away.

    Forced alignment may be switched off or unconfigured, in which case the
    words keep their spans and are marked approximate, because a span that
    was inherited rather than measured is exactly that.

    Or the alignment can fail. Then the words keep the best span already
    known, are marked approximate or uncertain, and are flagged so a person
    sees them. A failed measurement is never a reason to divide a span into
    plausible-looking pieces.
    """
    options = options or TimingOptions()
    if not tokens:
        return []
    phrase = _phrase_span(tokens)

    if not options.forced_alignment_enabled:
        return _keep_spans(tokens, phrase, TimingStatus.APPROXIMATE_SPAN, ())
    if aligner is None or not aligner.is_configured():
        _log.debug("No forced aligner is available, so the mapped spans are kept.")
        return _keep_spans(tokens, phrase, TimingStatus.APPROXIMATE_SPAN, ())
    if not aligner.supports(language):
        _log.info(
            "%s cannot be aligned, so the phrase keeps its mapped timing and is "
            "marked uncertain.",
            language.display_name,
        )
        return _keep_spans(tokens, phrase, TimingStatus.UNCERTAIN, ())

    text = " ".join(token.text for token in tokens if token.text.strip())
    try:
        measured = read_aligned_words(
            aligner.align(audio_path, text, language, canonical_offset)
        )
    except ProviderError as error:
        _log.warning("Forced alignment did not answer: %s", error)
        return _keep_spans(
            tokens, phrase, TimingStatus.APPROXIMATE_SPAN, (ReviewReason.FORCED_ALIGNMENT_FAILED,)
        )
    except Exception as error:  # noqa: BLE001 - a broken aligner must not stop the run
        _log.exception("Forced alignment failed unexpectedly.")
        _log.debug("The phrase keeps its inherited timing: %s", error)
        return _keep_spans(
            tokens, phrase, TimingStatus.APPROXIMATE_SPAN, (ReviewReason.FORCED_ALIGNMENT_FAILED,)
        )

    if len(measured) != len(tokens) or not _rises(measured):
        # The aligner answered with a different number of words, or with
        # words out of order, so which measurement belongs to which word is
        # not known. Guessing the correspondence would put real numbers on
        # the wrong words, which is worse than having no new numbers at all.
        _log.warning(
            "Forced alignment returned %d words for a phrase of %d, so its measurements "
            "cannot be matched to the words and the mapped spans are kept.",
            len(measured),
            len(tokens),
        )
        return _keep_spans(
            tokens, phrase, TimingStatus.APPROXIMATE_SPAN, (ReviewReason.FORCED_ALIGNMENT_FAILED,)
        )

    provider = getattr(aligner, "provider", None)
    outcomes: list[TimingOutcome] = []
    for word in measured:
        status = status_for_loss(word.loss, options)
        outcomes.append(
            TimingOutcome(
                status=status,
                span=word.span,
                containing_span=phrase or word.span,
                source=provider if isinstance(provider, Provider) else None,
                confidence=(
                    Confidence.HIGH
                    if status is TimingStatus.FORCED_ALIGNED
                    else Confidence.REVIEW_SUGGESTED
                ),
            )
        )
    return outcomes


def _keep_spans(
    tokens: Sequence[FinalToken],
    phrase: AudioSpan | None,
    status: TimingStatus,
    reasons: tuple[ReviewReason, ...],
) -> list[TimingOutcome]:
    """Keep the best span each word already has, and say what it is worth.

    A word that inherited its own start and end keeps them and is marked at
    the given status, which says the region is right and the edges were not
    measured for these words. A word that never had its own boundaries falls
    back to the span of the whole phrase and says that it shares it. Neither
    branch produces a number that nobody measured, which is the entire point
    of this function.
    """
    outcomes: list[TimingOutcome] = []
    for token in tokens:
        own = token.span
        if own is not None:
            outcomes.append(
                TimingOutcome(
                    status=status,
                    span=own,
                    containing_span=phrase or token.source_audio_span or own,
                    source=token.timing_source,
                    confidence=Confidence.REVIEW_SUGGESTED,
                    review_reasons=reasons,
                )
            )
            continue
        region = phrase or token.source_audio_span
        if region is None:
            outcomes.append(
                TimingOutcome(
                    status=TimingStatus.UNALIGNED,
                    span=None,
                    containing_span=None,
                    source=token.timing_source,
                    confidence=Confidence.UNRESOLVED,
                    review_reasons=reasons + (ReviewReason.UNALIGNED_WORD,),
                )
            )
            continue
        outcomes.append(
            TimingOutcome(
                status=TimingStatus.SHARED_PHRASE_SPAN,
                span=region,
                containing_span=region,
                source=token.timing_source,
                confidence=Confidence.REVIEW_SUGGESTED,
                review_reasons=reasons,
                shares_span=True,
            )
        )
    return outcomes


def _phrase_span(tokens: Sequence[FinalToken]) -> AudioSpan | None:
    """The whole region a phrase is known to sit in.

    Built from the words' own spans where they have them and from their
    containing spans where they do not, so a phrase whose middle word is
    unaligned still has a region that covers all of it.
    """
    starts: list[float] = []
    ends: list[float] = []
    for token in tokens:
        span = token.span or token.source_audio_span
        if span is None:
            continue
        starts.append(span.start)
        ends.append(span.end)
    if not starts:
        return None
    return AudioSpan(min(starts), max(max(ends), min(starts)))


def _rises(words: Sequence[AlignedWord]) -> bool:
    """Whether the measured words run forwards and each one ends after it starts.

    An answer that goes backwards is not a measurement of this phrase,
    whatever it is, and treating it as one would scatter the words across the
    recording.
    """
    previous = float("-inf")
    for word in words:
        if word.end < word.start or word.start < previous:
            return False
        previous = word.start
    return True
