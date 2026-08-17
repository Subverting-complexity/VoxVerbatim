"""Turning one recording into one transcript, from beginning to end.

Everything else in this package does one job well and knows nothing about
the others. This is the module that knows the order they go in, and it is
deliberately the only one that does. A reader who wants to understand what
the application actually does to a recording should be able to read the one
function below and follow it, rather than tracing calls through nine files.

The order is not arbitrary, and two parts of it are worth stating plainly
because they look like they could be swapped and cannot.

Reconciliation runs twice. The first pass settles everything the evidence
already covers and, just as importantly, works out what it cannot settle.
Only then is there a list of genuine disputes worth paying a second service
to look at. Escalating first would mean escalating everything, which costs
money on the ninety-odd per cent of words that were never in doubt.

Timing is corrected last, after the text is final. Measuring where words
fall in the audio and then changing the words would leave the measurements
describing something that is no longer there. That is the whole reason
section 20 puts forced alignment where it does.

Nothing here raises because a service failed. The pipeline's contract is
that it returns a transcript, and a service that did not answer becomes a
sentence in that transcript's warnings which the user reads at the end. It
raises only when the recording itself cannot be read, because then there is
genuinely nothing to return.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from audio_transcriber.settings import TranscriptionSettings
from audio_transcriber.transcription import (
    canonical as canonical_audio,
    diarisation,
    escalation,
    exports,
    language as language_rules,
    passes,
    reconcile as reconcile_rules,
    timing as timing_rules,
)
from audio_transcriber.transcription.alignment import build_aligned_table, choose_backbone
from audio_transcriber.transcription.context import build_context_package
from audio_transcriber.transcription.model import (
    CanonicalAudio,
    FinalToken,
    Language,
    Provider,
    ProviderResult,
    RecordingConfiguration,
    Speaker,
    Transcript,
)
from audio_transcriber.transcription.providers import registry
from audio_transcriber.transcription.store import TranscriptStore
from audio_transcriber.transcription.vocabulary import (
    Vocabulary,
    VocabularyIndex,
    resolve_terms,
)

_log = logging.getLogger(__name__)


@dataclass
class PipelineOptions:
    """Everything a run needs that is not the recording itself."""

    configuration: RecordingConfiguration = field(default_factory=RecordingConfiguration)
    settings: TranscriptionSettings = field(default_factory=TranscriptionSettings)
    vocabulary: Vocabulary = field(default_factory=Vocabulary)


@dataclass(frozen=True)
class PipelineProgress:
    """How far a run has come, and what it is doing, in words.

    The sentence is shown on screen and read out by a screen reader, so it
    is written for somebody listening rather than as a log line. "Sending
    the recording to ElevenLabs, OpenAI and Microsoft, all at once" tells a
    waiting user something; "stage 2 of 9" does not.
    """

    fraction: float
    stage: str
    provider: Provider | None = None


ProgressCallback = Callable[[PipelineProgress], None]
CancelCheck = Callable[[], bool]

#: Roughly how much of the wall clock each stage takes, so the progress bar
#: moves at a believable rate. The services dominate by a wide margin: they
#: are minutes of network waiting, where everything after them is seconds of
#: arithmetic. These are honest proportions rather than equal slices, which
#: would leave the bar apparently stuck for most of the run and then sprint.
_STAGE_SHARES: tuple[tuple[str, float], ...] = (
    ("preparing", 0.04),
    ("services", 0.62),
    ("aligning", 0.06),
    ("reconciling", 0.06),
    ("escalating", 0.12),
    ("adjudicating", 0.06),
    ("timing", 0.03),
    ("saving", 0.01),
)


def transcribe_recording(
    recording: Path,
    options: PipelineOptions,
    progress: ProgressCallback | None = None,
    cancelled: CancelCheck | None = None,
) -> Transcript:
    """Transcribe one recording and return the finished transcript.

    Raises OSError only if the recording cannot be read at all. Every other
    failure, including every service failing, comes back as a transcript
    carrying warnings.
    """
    recording = Path(recording)
    started = time.monotonic()
    reporter = _Reporter(progress)
    settings = options.settings
    configuration = options.configuration

    store = TranscriptStore(recording, settings.processing.transcript_folder_suffix)
    transcript = Transcript(
        recording_name=recording.name,
        configuration=configuration,
        created_at=_now(),
        effective_settings=settings.for_provenance(),
    )

    # -- Prepare the one file and the one clock everything measures against
    reporter.stage("preparing", 0.0, f"Preparing {recording.name}.")
    canonical = canonical_audio.prepare_canonical_audio(recording, store.folder)
    transcript.canonical_audio = canonical
    reporter.stage("preparing", 1.0, "The recording is ready.")

    if _stopped(cancelled):
        return _finish(transcript, store, started, stopped=True)

    # -- Work out what to tell the services about this recording
    terms = resolve_terms(options.vocabulary, configuration.vocabulary_profile_ids)
    context = build_context_package(configuration, terms)
    index = VocabularyIndex(terms)

    # -- Ask every service at once
    providers = registry.build_providers(settings)
    outcome = passes.run_passes(
        canonical=canonical,
        providers=providers,
        context=context,
        configuration=configuration,
        store=store,
        chunk_folder=store.folder / "chunks",
        progress=lambda fraction, message: reporter.stage("services", fraction, message),
        cancelled=cancelled,
        chunk_overlap_seconds=settings.processing.provider_chunk_overlap_seconds,
    )
    transcript.provider_results = outcome.results
    transcript.requests.extend(outcome.requests)
    transcript.warnings.extend(outcome.warnings)

    if not outcome.any_succeeded:
        transcript.warnings.append(
            "No transcription service produced a result, so there is no transcript. "
            "Check the API keys in Settings and that this computer is online."
        )
        return _finish(transcript, store, started, stopped=outcome.cancelled)

    if _stopped(cancelled):
        return _finish(transcript, store, started, stopped=True)

    # -- Put every service's words beside each other on one timeline
    reporter.stage("aligning", 0.0, "Comparing what each service heard.")
    succeeded = [result for result in outcome.results.values() if result.succeeded]
    backbone_provider = choose_backbone(succeeded)
    if backbone_provider is None:
        transcript.warnings.append(
            "No service returned word timings, so the words cannot be placed in the "
            "recording. The text is still available but nothing can be played back."
        )
        return _finish(transcript, store, started, stopped=False)
    if backbone_provider is not Provider.ELEVENLABS:
        # Worth saying plainly rather than leaving in a log. The backbone
        # decides every timestamp and every initial speaker label, so a
        # different one is a different transcript, not a detail.
        transcript.warnings.append(
            f"{Provider.ELEVENLABS.display_name} did not answer, so timings and "
            f"speakers come from {backbone_provider.display_name} instead. They are "
            "usually less precise."
        )
    backbone = outcome.results[backbone_provider]
    others = [result for result in succeeded if result.provider is not backbone_provider]
    table = build_aligned_table(backbone, others)
    reporter.stage("aligning", 1.0, "The services have been compared.")

    # -- Decide what was said
    reporter.stage("reconciling", 0.0, "Working out what was said.")
    languages = language_rules.read_languages(table, succeeded, configuration)
    tokens = reconcile_rules.reconcile(
        table,
        configuration=configuration,
        vocabulary=index,
        languages=languages,
        results=succeeded,
    )
    reporter.stage("reconciling", 1.0, _reconciled_sentence(tokens))

    # -- Ask a second service about the places that are still in doubt
    tokens = _escalate(
        tokens,
        transcript,
        outcome.results,
        canonical,
        settings,
        store,
        reporter,
        cancelled,
    )

    # -- Let a language model decide the few that are still unsettled
    tokens = _adjudicate(tokens, transcript, settings, context, reporter, cancelled)

    # -- Check the speakers, then correct the timing of what changed
    tokens = _check_speakers(tokens, transcript, outcome.results, configuration)
    tokens = _correct_timing(tokens, transcript, canonical, settings, reporter)

    transcript.tokens = tokens
    transcript.speakers = _speakers_in(tokens, configuration)
    return _finish(transcript, store, started, stopped=_stopped(cancelled))


# -- The stages that need more than a line ------------------------------


def _escalate(
    tokens: list[FinalToken],
    transcript: Transcript,
    results: dict[Provider, ProviderResult],
    canonical: CanonicalAudio,
    settings: TranscriptionSettings,
    store: TranscriptStore,
    reporter: "_Reporter",
    cancelled: CancelCheck | None,
) -> list[FinalToken]:
    """Get a second opinion on the disputes worth paying for."""
    if not settings.processing.escalation_enabled:
        return tokens
    windows = escalation.plan_escalation(tokens, results, canonical.duration)
    if not windows:
        return tokens

    limit = settings.processing.maximum_escalations_per_recording
    if len(windows) > limit:
        # Said out loud rather than silently trimmed. A user who sees ten
        # unresolved words and does not know that forty more were never
        # looked at has been misled about how much review is left.
        transcript.warnings.append(
            f"{len(windows)} places needed a second opinion but the limit in Settings "
            f"is {limit}, so {len(windows) - limit} of them were left for you to "
            "check by hand instead."
        )
        windows = windows[:limit]

    provider = registry.build_escalation_provider(settings)
    if provider is None:
        transcript.warnings.append(
            f"{len(windows)} places needed a second opinion, but no escalation "
            "service is set up, so they are waiting for you instead."
        )
        return tokens

    reporter.stage("escalating", 0.0, _escalation_sentence(len(windows)))
    result = escalation.escalate(
        windows,
        provider,
        canonical,
        folder=store.folder / "escalation",
        cancelled=cancelled,
    )
    escalation.apply_outcome(transcript, result)
    transcript.requests.extend(getattr(result, "requests", ()))
    transcript.warnings.extend(getattr(result, "warnings", ()))
    reporter.stage("escalating", 1.0, "The second opinions are in.")
    return tokens


def _adjudicate(
    tokens: list[FinalToken],
    transcript: Transcript,
    settings: TranscriptionSettings,
    context,
    reporter: "_Reporter",
    cancelled: CancelCheck | None,
) -> list[FinalToken]:
    """Let the configured OpenAI model decide the few still unsettled.

    Deliberately last among the deciding stages. A language model is the
    most expensive opinion available and the one least anchored to the
    audio, so it should only ever see what cheaper, better-grounded evidence
    could not settle.
    """
    if not settings.processing.adjudication_enabled:
        return tokens
    if _stopped(cancelled):
        return tokens

    from audio_transcriber.transcription import adjudication

    adjudicator = adjudication.Adjudicator(
        api_key=settings.openai_adjudication.api_key,
        model=settings.openai_adjudication.model,
        reasoning_effort=settings.openai_adjudication.reasoning_effort,
        parameters=dict(settings.openai_adjudication.parameters),
        timeout_seconds=settings.processing.provider_timeout_seconds,
    )
    if not adjudicator.is_configured():
        return tokens

    disputes = _build_disputes(tokens, transcript)
    if not disputes:
        return tokens

    reporter.stage("adjudicating", 0.0, _adjudication_sentence(len(disputes)))
    result = adjudicator.adjudicate(
        disputes,
        recording_context=context.recording_context,
        vocabulary_terms=context.term_texts,
        languages=context.languages,
    )
    if result.request is not None:
        transcript.requests.append(result.request)
    if result.error:
        transcript.warnings.append(
            f"The language model could not be consulted, so {len(disputes)} places "
            f"are waiting for you instead. {result.error}"
        )
    reporter.stage("adjudicating", 1.0, adjudication.summarise(result))
    return tokens


#: How many settled words either side of a dispute go with it as context.
#: Enough for the model to see the sentence the disputed word sits in,
#: without turning a one-word question into a paragraph it might range
#: across.
_ADJUDICATION_CONTEXT_WORDS = 12


def _build_disputes(tokens: list[FinalToken], transcript: Transcript) -> list:
    """Gather the words still in doubt into the regions to ask about.

    Adjacent unsettled words become one dispute rather than several. They
    are usually one misheard phrase rather than three unrelated problems,
    and asking about them together gives the model the phrase instead of
    three words with holes between them.

    A high-risk value is deliberately included even though the model is
    forbidden from settling it. Its neighbours may still be decidable, and
    the adjudicator's own validation refuses the risky word itself, which is
    a guarantee rather than a request.
    """
    from audio_transcriber.transcription.adjudication import Dispute

    disputes: list[Dispute] = []
    position = 0
    while position < len(tokens):
        if not tokens[position].needs_review:
            position += 1
            continue
        end = position
        while end + 1 < len(tokens) and tokens[end + 1].needs_review:
            end += 1
        group = tokens[position : end + 1]
        backbone = [
            source
            for token in group
            for source in _backbone_tokens_for(token, transcript)
        ]
        reasons = sorted(
            {reason.display_name for token in group for reason in token.review_reasons}
        )
        disputes.append(
            Dispute(
                tokens=tuple(group),
                backbone_tokens=tuple(backbone),
                escalation_tokens=(),
                preceding_text=_words_before(tokens, position),
                following_text=_words_after(tokens, end),
                note="; ".join(reasons),
            )
        )
        position = end + 1
    return disputes


def _backbone_tokens_for(token: FinalToken, transcript: Transcript):
    """The service words behind one final word, for the model to weigh."""
    found = []
    for reference in token.source_tokens:
        result = transcript.provider_results.get(reference.provider)
        if result is None:
            continue
        source = result.token_at(reference.index)
        if source is not None:
            found.append(source)
    return found


def _words_before(tokens: list[FinalToken], position: int) -> str:
    start = max(0, position - _ADJUDICATION_CONTEXT_WORDS)
    return " ".join(token.text for token in tokens[start:position] if token.text)


def _words_after(tokens: list[FinalToken], end: int) -> str:
    stop = min(len(tokens), end + 1 + _ADJUDICATION_CONTEXT_WORDS)
    return " ".join(token.text for token in tokens[end + 1 : stop] if token.text)


def _check_speakers(
    tokens: list[FinalToken],
    transcript: Transcript,
    results: dict[Provider, ProviderResult],
    configuration: RecordingConfiguration,
) -> list[FinalToken]:
    """Compare the speaker labels against a second service where there is one.

    A speaker correction never touches the text or the timing. They are
    three separate answers with three separate sources, which is the whole
    idea the design rests on.
    """
    second = results.get(Provider.ASSEMBLYAI)
    if second is None or not second.succeeded:
        return tokens
    backbone_turns = diarisation.speaker_turns(tokens)
    other_turns = diarisation.turns_from_result(second)
    if not other_turns:
        return tokens
    mapping = diarisation.map_speaker_labels(backbone_turns, other_turns)
    decisions = diarisation.reconcile_speakers(tokens, other_turns, mapping)
    changed = diarisation.apply_speakers(tokens, decisions)
    if changed:
        transcript.warnings.append(
            f"{changed} words were given a different speaker after a second service "
            "was consulted. Their text and timing were not changed."
        )
    return tokens


def _correct_timing(
    tokens: list[FinalToken],
    transcript: Transcript,
    canonical: CanonicalAudio,
    settings: TranscriptionSettings,
    reporter: "_Reporter",
) -> list[FinalToken]:
    """Measure the words again where reconciliation moved them.

    Only where it moved them. Running this over every word would cost a
    great deal for no gain, since a word nobody disputed already has the
    timing the service measured for it.
    """
    if not settings.processing.forced_alignment_enabled:
        return tokens
    wanted = [token for token in tokens if timing_rules.needs_forced_alignment(token)]
    if not wanted:
        return tokens

    aligner = registry.build_forced_aligner(settings)
    if aligner is None:
        transcript.warnings.append(
            f"{len(wanted)} words changed enough to need their timing measured again, "
            "but no forced aligner is set up, so their timing is approximate."
        )
        return tokens

    reporter.stage("timing", 0.0, f"Measuring {len(wanted)} corrected words again.")
    outcomes = timing_rules.realign(
        wanted,
        aligner,
        Path(canonical.path),
        language=_dominant_language(tokens),
    )
    for token, outcome in zip(wanted, outcomes):
        timing_rules.apply_timing(token, outcome)
    reporter.stage("timing", 1.0, "The corrected words have been timed again.")
    return tokens


# -- Small helpers -------------------------------------------------------


class _Reporter:
    """Turns each stage's own progress into one honest overall figure."""

    def __init__(self, progress: ProgressCallback | None) -> None:
        self._progress = progress
        self._before: dict[str, float] = {}
        running = 0.0
        for name, share in _STAGE_SHARES:
            self._before[name] = running
            running += share
        self._shares = dict(_STAGE_SHARES)

    def stage(self, name: str, fraction: float, message: str) -> None:
        if self._progress is None:
            return
        share = self._shares.get(name, 0.0)
        start = self._before.get(name, 0.0)
        overall = start + share * max(0.0, min(1.0, fraction))
        self._progress(PipelineProgress(fraction=min(1.0, overall), stage=message))


def _finish(
    transcript: Transcript,
    store: TranscriptStore,
    started: float,
    stopped: bool,
) -> Transcript:
    """Save the transcript and its exports, and say how it went."""
    transcript.completed_at = _now()
    if stopped:
        transcript.warnings.append(
            "The run was stopped before it finished, so this transcript is "
            "incomplete."
        )
    if not store.save(transcript):
        transcript.warnings.append(
            f"The transcript could not be saved to {store.transcript_path}."
        )
    _write_exports(transcript, store)
    _log.info(
        "Transcribed %s in %.1f seconds: %d words, %d needing review.",
        transcript.recording_name,
        time.monotonic() - started,
        len(transcript.tokens),
        len(transcript.review_tokens),
    )
    return transcript


def _write_exports(transcript: Transcript, store: TranscriptStore) -> None:
    """Write the two readable documents, without letting either stop the run.

    An export that fails has cost the user a convenience, not their
    transcript, which is safely saved by this point. So it is reported and
    stepped over rather than raised.
    """
    for name, render in (
        (exports.TEXT_EXPORT_NAME, exports.render_plain_text),
        (exports.REPORT_EXPORT_NAME, exports.render_review_report),
    ):
        try:
            store.write_export(name, render(transcript))
        except Exception:  # an export is a convenience, never the transcript
            _log.exception("Could not write the %s export.", name)
            transcript.warnings.append(f"The {name} export could not be written.")


def _speakers_in(
    tokens: list[FinalToken],
    configuration: RecordingConfiguration,
) -> list[Speaker]:
    """The speakers actually found, named from the user's list where possible.

    The names are matched to labels in the order each speaker first talks,
    which is a guess and is treated as one: it saves the user renaming
    everybody by hand, and the review window lets them correct it.
    """
    order: list[str] = []
    for token in tokens:
        if token.speaker and token.speaker not in order:
            order.append(token.speaker)
    known = list(configuration.known_speakers)
    return [
        Speaker(id=label, name=known[position] if position < len(known) else None)
        for position, label in enumerate(order)
    ]


def _dominant_language(tokens: list[FinalToken]) -> Language:
    counts: dict[Language, int] = {}
    for token in tokens:
        if token.language is not Language.UNKNOWN:
            counts[token.language] = counts.get(token.language, 0) + 1
    if not counts:
        return Language.ENGLISH
    return max(counts.items(), key=lambda item: item[1])[0]


def _reconciled_sentence(tokens: list[FinalToken]) -> str:
    needing = sum(1 for token in tokens if token.needs_review)
    if not needing:
        return f"{len(tokens)} words settled, with nothing left in doubt."
    places = "place" if needing == 1 else "places"
    return f"{len(tokens)} words settled, with {needing} {places} still in doubt."


def _escalation_sentence(count: int) -> str:
    places = "place" if count == 1 else "places"
    return f"Asking a second service about {count} {places}."


def _adjudication_sentence(count: int) -> str:
    places = "place" if count == 1 else "places"
    return f"Asking the language model to decide {count} {places}."


def _stopped(cancelled: CancelCheck | None) -> bool:
    return cancelled is not None and cancelled()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
