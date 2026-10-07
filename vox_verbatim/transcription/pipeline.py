"""Turning one recording into one transcript, from beginning to end.

Everything else in this package does one job well and knows nothing about
the others. This is the module that knows the order they go in, and it is
deliberately the only one that does. A reader who wants to understand what
the application actually does to a recording should be able to read the one
function below and follow it, rather than tracing calls through nine files.

The order is not arbitrary, and two parts of it are worth stating plainly
because they look like they could be swapped and cannot.

Reconciliation runs before escalation. It settles everything the evidence
already covers and, just as importantly, works out what it cannot settle.
Only then is there a list of genuine disputes worth paying a second service
to look at. Escalating first would mean escalating everything, which costs
money on the ninety-odd per cent of words that were never in doubt. The
second opinions are then weighed against the words they were asked about,
in :func:`escalation.apply_answers`, rather than by running the whole
reconciliation again: the answer covers a few seconds, and a service that
heard a few seconds is allowed to choose between the readings the others
offered, never to rewrite words nobody asked about.

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
from typing import Callable, Sequence

from vox_verbatim.settings import TranscriptionSettings
from vox_verbatim.transcription import (
    canonical as canonical_audio,
    diarisation,
    escalation,
    exports,
    language as language_rules,
    passes,
    reconcile as reconcile_rules,
    timing as timing_rules,
)
from vox_verbatim.transcription.alignment import build_aligned_table, choose_backbone
from vox_verbatim.transcription.calibration import CalibrationStore, reliability_weights
from vox_verbatim.transcription.context import build_context_package
from vox_verbatim.transcription.model import (
    AudioSpan,
    CanonicalAudio,
    Confidence,
    FinalToken,
    Language,
    Provider,
    ProviderResult,
    RecordingConfiguration,
    ReviewReason,
    Speaker,
    Transcript,
)
from vox_verbatim.transcription.providers import registry
from vox_verbatim.transcription.store import TranscriptStore
from vox_verbatim.transcription.vocabulary import (
    Vocabulary,
    VocabularyIndex,
    VocabularyTerm,
    folder_terms_first,
    resolve_terms,
)

_log = logging.getLogger(__name__)


@dataclass
class PipelineOptions:
    """Everything a run needs that is not the recording itself."""

    configuration: RecordingConfiguration = field(default_factory=RecordingConfiguration)
    settings: TranscriptionSettings = field(default_factory=TranscriptionSettings)
    vocabulary: Vocabulary = field(default_factory=Vocabulary)
    calibration_path: Path | None = None
    """Where the service statistics are kept, or ``None`` to use the defaults.

    The file is read when each recording starts, so a run picks up what the
    reviews before it taught. Reconcile then believes each service by what
    it has earned rather than by the fixed defaults alone.
    """
    folder_terms: tuple[VocabularyTerm, ...] = ()
    """The names the recordings' folder learned in its reviews, as terms.

    They are sent ahead of the profile terms, so a service that will not
    take every term drops profile terms first. Read from the folder's
    project file once, when the run starts.
    """


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
    terms = folder_terms_first(
        options.folder_terms,
        resolve_terms(options.vocabulary, configuration.vocabulary_profile_ids),
    )
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
            "No service returned any words, so there is nothing to build a transcript "
            "from. The raw answers are kept in the transcript folder."
        )
        return _finish(transcript, store, started, stopped=False)
    if backbone_provider is not Provider.ELEVENLABS:
        # Worth saying plainly rather than leaving in a log. The backbone
        # decides every timestamp and every initial speaker label, so a
        # different one is a different transcript, not a detail. And the
        # services that can stand in for ElevenLabs do not time their
        # words, so the honest sentence is that nothing can be played.
        timed = any(token.has_timing for token in outcome.results[backbone_provider].tokens)
        if timed:
            detail = "They are usually less precise."
        else:
            detail = (
                f"{backbone_provider.display_name} does not time its words, so the words "
                "cannot be placed in the recording and nothing can be played back from "
                "the review window."
            )
        transcript.warnings.append(
            f"{Provider.ELEVENLABS.display_name} did not answer, so timings and "
            f"speakers come from {backbone_provider.display_name} instead. {detail}"
        )
    backbone = outcome.results[backbone_provider]
    others = [result for result in succeeded if result.provider is not backbone_provider]
    try:
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
            weights=_learned_weights(options.calibration_path),
        )
    except Exception as error:  # noqa: BLE001 - the paid answers must still be kept
        # The services have answered and been paid by now. A fault here used
        # to end the recording as "failed", with nothing to show for the
        # money. What every service said is still on the transcript, and is
        # saved, so the run can be explained and the words recovered.
        _log.exception("Comparing the services' answers failed for %s.", recording.name)
        transcript.warnings.append(
            "The services answered, but their answers could not be compared and "
            f"reconciled: {error}. Each service's words are kept in the raw-responses "
            "folder, and in the transcript file when this run saves one."
        )
        return _finish(transcript, store, started, stopped=False)
    reporter.stage("reconciling", 1.0, _reconciled_sentence(tokens))

    # From here on the transcript has words. They go onto it now rather
    # than at the end, so that every later stage works on the transcript's
    # own list and so that a stage that fails leaves the words it had
    # rather than none. Each stage below changes the words in place.
    transcript.tokens = tokens

    # -- Ask a second service about the places that are still in doubt
    _guarded(
        transcript,
        "second opinions",
        lambda: _escalate(
            tokens, transcript, outcome.results, canonical, settings, context, store,
            reporter, cancelled,
        ),
    )

    # -- Let a language model decide the few that are still unsettled
    _guarded(
        transcript,
        "adjudication",
        lambda: _adjudicate(tokens, transcript, settings, context, reporter, cancelled),
    )

    # -- Check the speakers, then correct the timing of what changed
    _guarded(
        transcript,
        "the speaker check",
        lambda: _check_speakers(tokens, transcript, outcome.results, configuration),
    )
    _guarded(
        transcript,
        "the timing correction",
        lambda: _correct_timing(
            tokens, transcript, canonical, settings, store, reporter, cancelled
        ),
    )

    transcript.speakers = _speakers_in(tokens, configuration)
    return _finish(transcript, store, started, stopped=_stopped(cancelled))


def _learned_weights(path: Path | None) -> reconcile_rules.ReliabilityWeights:
    """The reliability weights the service statistics support, or the defaults.

    A missing or damaged statistics file gives the defaults. Each candidate
    records the weight it was given under ``provider_reliability`` beside
    the other parts of its score, so a result that changed because of what
    was learned can be explained from the transcript.
    """
    if path is None:
        return reconcile_rules.DEFAULT_RELIABILITY
    try:
        return reliability_weights(CalibrationStore(path).load())
    except Exception:  # noqa: BLE001 - statistics must never stop a run
        _log.warning("The service statistics could not be used; using the defaults.", exc_info=True)
        return reconcile_rules.DEFAULT_RELIABILITY


def _guarded(transcript: Transcript, stage_name: str, run: Callable[[], object]) -> None:
    """Run one of the later stages without letting it cost the transcript.

    By the time these stages run, every service has answered and been paid.
    A bug in one of them used to escape the pipeline, and the runner then
    recorded the whole recording as failed, with the raw answers on disk and
    no transcript. The words already reconciled are worth far more than any
    of these refinements, so a failure here is logged, said in the warnings,
    and stepped over.
    """
    try:
        run()
    except Exception as error:  # noqa: BLE001 - a refinement must not lose the words
        _log.exception("The %s stage failed; the transcript continues without it.", stage_name)
        transcript.warnings.append(
            f"The {stage_name} stage failed part way through, so the transcript was "
            f"finished without it: {error}"
        )


# -- The stages that need more than a line ------------------------------


def _escalate(
    tokens: list[FinalToken],
    transcript: Transcript,
    results: dict[Provider, ProviderResult],
    canonical: CanonicalAudio,
    settings: TranscriptionSettings,
    context,
    store: TranscriptStore,
    reporter: "_Reporter",
    cancelled: CancelCheck | None,
) -> list[FinalToken]:
    """Get a second opinion on the disputes worth paying for, and use it.

    The options come from Settings and from the recording's own
    configuration, through one call, so that the Afrikaans switch, the
    speaker count, the vocabulary and the model names all reach the
    service. Without that the escalation runs with built-in defaults, which
    looks the same from the outside and is a different request.

    The ceiling is applied by :func:`escalation.escalate` itself, to a list
    ordered by how much each window is worth, so that when a long
    recording has more disputes than the limit it is the agreed-upon
    numbers that are left out rather than the real disagreements. Whatever
    it leaves out it records, so every word it did not reach is flagged.
    """
    if not settings.processing.escalation_enabled:
        return tokens
    options = escalation.EscalationOptions.from_settings(
        settings.processing,
        settings.assemblyai,
        transcript.configuration,
        vocabulary_terms=context.term_texts,
    )
    windows = escalation.plan_escalation(tokens, results, canonical.duration, options)
    if not windows:
        return tokens
    windows = escalation.prioritise(windows)

    provider = registry.build_escalation_provider(settings)
    problem = None if provider is None else provider.describe_configuration_problem()
    if provider is None or problem is not None:
        detail = f" {problem}" if problem else ""
        transcript.warnings.append(
            f"{len(windows)} places needed a second opinion, but no escalation "
            f"service is set up, so they are waiting for you instead.{detail}"
        )
        asked_about = {token_id for window in windows for token_id in window.token_ids}
        for token in tokens:
            if token.id in asked_about:
                token.flag(ReviewReason.ESCALATION_UNRESOLVED)
        return tokens

    sending = min(len(windows), max(0, options.maximum_escalations))
    reporter.stage("escalating", 0.0, _escalation_sentence(sending))
    result = escalation.escalate(
        windows,
        provider,
        canonical,
        options,
        folder=store.folder / "escalation",
        cancelled=cancelled,
        progress=lambda done, total: reporter.stage(
            "escalating",
            done / max(1, total),
            f"Second opinion {done} of {total} is in.",
        ),
    )
    transcript.requests.extend(result.requests)
    # The answers first, then the bookkeeping. apply_outcome flags the words
    # no answer reached, and it must see the words that apply_answers has
    # just settled so that it does not flag those.
    applied = escalation.apply_answers(tokens, result)
    if applied.evidence is not None:
        transcript.provider_results[applied.evidence.provider] = applied.evidence
    escalation.apply_outcome(transcript, result)
    reporter.stage("escalating", 1.0, applied.sentence)
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
    if not settings.openai_adjudication.enabled:
        # The user has switched the service off, so nothing is sent. The run
        # still finishes, and the words it would have been asked about wait
        # for review. The warning is only given where there was something to
        # ask about, so a clean recording does not mention it.
        if _build_disputes(tokens, transcript):
            transcript.warnings.append(
                "Adjudication was not used, because OpenAI adjudication is "
                "switched off in Settings. The words it would have settled are "
                "waiting for review."
            )
        return tokens

    from vox_verbatim.transcription import adjudication

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

    # Sent in batches rather than all at once. A three-hour recording can
    # have thousands of disputed places, and one request carrying all of
    # them is far past what any model will read; it fails, and every one of
    # them is then left for a person. A batch is small enough to answer and
    # large enough that the requests do not run into the hundreds.
    reporter.stage("adjudicating", 0.0, _adjudication_sentence(len(disputes)))
    batches = [
        disputes[start : start + _ADJUDICATION_BATCH_SIZE]
        for start in range(0, len(disputes), _ADJUDICATION_BATCH_SIZE)
    ]
    settled = refused = left = 0
    failures: list[str] = []
    for number, batch in enumerate(batches, start=1):
        if _stopped(cancelled):
            # What has not been asked about yet stays as reconciliation
            # left it; a stopped run is reported as incomplete elsewhere.
            break
        result = adjudicator.adjudicate(
            batch,
            recording_context=context.recording_context,
            vocabulary_terms=context.term_texts,
            languages=context.languages,
        )
        if result.request is not None:
            transcript.requests.append(result.request)
        if result.error:
            failures.append(result.error)
        settled += len(result.applied)
        refused += len(result.refused)
        left += len(result.declined) + len(result.unanswered)
        reporter.stage(
            "adjudicating",
            number / len(batches),
            f"The language model has answered {number} of {len(batches)} batches.",
        )
    if failures:
        distinct = list(dict.fromkeys(failures))
        transcript.warnings.append(
            f"The language model could not be consulted about {len(failures)} of "
            f"{len(batches)} batches of disputed places, so those are waiting for you "
            f"instead. {' '.join(distinct[:3])}"
        )
    parts = [f"{settled} settled"]
    if refused:
        parts.append(f"{refused} refused")
    if left:
        parts.append(f"{left} left for review")
    reporter.stage("adjudicating", 1.0, "Adjudication: " + ", ".join(parts) + ".")
    return tokens


#: How many disputed places go into one request to the language model.
_ADJUDICATION_BATCH_SIZE = 30


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
    from vox_verbatim.transcription.adjudication import Dispute

    disputes: list[Dispute] = []
    position = 0
    while position < len(tokens):
        if not _text_in_doubt(tokens[position]):
            position += 1
            continue
        end = position
        while end + 1 < len(tokens) and _text_in_doubt(tokens[end + 1]):
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
                escalation_tokens=tuple(_escalation_tokens_for(group, transcript)),
                preceding_text=_words_before(tokens, position),
                following_text=_words_after(tokens, end),
                note="; ".join(reasons),
            )
        )
        position = end + 1
    return disputes


#: The review reasons that say the text of a word is in doubt. A word that
#: is in the queue for its speaker or its timing has nothing for a language
#: model to decide, and asking would only cost money and, on a failure,
#: leave a perfectly good word marked as uncertain.
_TEXT_DOUBT_REASONS = frozenset(
    {
        ReviewReason.PROVIDER_DISAGREEMENT,
        ReviewReason.PROPER_NAME_DISAGREEMENT,
        ReviewReason.NUMERIC_DISAGREEMENT,
        ReviewReason.HIGH_RISK_ENTITY,
        ReviewReason.LOW_ACOUSTIC_CONFIDENCE,
        ReviewReason.ESCALATION_UNRESOLVED,
    }
)


def _text_in_doubt(token: FinalToken) -> bool:
    """Whether the language model has a question to answer about this word.

    It needs two things: a reason to doubt the text, and more than one
    reading to choose between. The model is only allowed to pick among the
    readings the services offered, so a word with one reading gives it
    nothing to do.
    """
    if token.human_corrected or not token.needs_review:
        return False
    doubted = token.text_confidence is not Confidence.HIGH or any(
        reason in _TEXT_DOUBT_REASONS for reason in token.review_reasons
    )
    if not doubted:
        return False
    readings = {candidate.text for candidate in token.candidates if candidate.text}
    return len(readings) > 1


def _escalation_tokens_for(group: Sequence[FinalToken], transcript: Transcript):
    """What the escalation service heard across these words, where it was asked."""
    result = transcript.provider_results.get(Provider.ASSEMBLYAI)
    if result is None or not result.tokens:
        return []
    spans = [token.span for token in group if token.span is not None]
    if not spans:
        return []
    region = AudioSpan(min(span.start for span in spans), max(span.end for span in spans))
    return [
        word
        for word in result.tokens
        if word.start is not None
        and word.end is not None
        and AudioSpan(word.start, max(word.start, word.end)).overlaps(region)
    ]


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
    store: TranscriptStore,
    reporter: "_Reporter",
    cancelled: CancelCheck | None = None,
) -> list[FinalToken]:
    """Measure the words again where reconciliation moved them.

    Only where it moved them. Running this over every word would cost a
    great deal for no gain, since a word nobody disputed already has the
    timing the service measured for it.

    Each run of moved words is measured as a phrase against a clip of its
    own few seconds of audio, never as one list against the whole
    recording. An aligner is given a text and an audio file and must place
    every word of the text somewhere in the file. Hand it forty scattered
    words and three hours, and it will place them, in order, at forty
    moments that have nothing to do with where they were said; the answer
    passes every sanity check and is wrong everywhere. A phrase against
    the seconds it was spoken in is the question an aligner can answer.
    """
    if not settings.processing.forced_alignment_enabled:
        return tokens
    phrases = _phrases_to_realign(tokens)
    if not phrases:
        return tokens
    wanted = sum(len(phrase) for phrase in phrases)

    aligner = registry.build_forced_aligner(settings)
    if aligner is None:
        transcript.warnings.append(
            f"{wanted} words changed enough to need their timing measured again, "
            "but no forced aligner is set up, so their timing is approximate."
        )
        return tokens

    limit = _MAXIMUM_ALIGNMENT_PHRASES
    if len(phrases) > limit:
        transcript.warnings.append(
            f"{len(phrases)} phrases needed their timing measured again, which is more "
            f"than the {limit} this application will send for one recording, so the "
            f"last {len(phrases) - limit} keep their approximate timing."
        )
        phrases = phrases[:limit]

    language = _dominant_language(tokens)
    clip_folder = store.folder / "alignment"
    reporter.stage("timing", 0.0, f"Measuring {wanted} corrected words again.")
    measured = 0
    for number, phrase in enumerate(phrases, start=1):
        if _stopped(cancelled):
            break
        span = _phrase_audio_span(phrase)
        if span is None:
            continue
        try:
            clip = canonical_audio.cut_window(
                canonical,
                span.padded(_ALIGNMENT_PADDING_SECONDS, _ALIGNMENT_PADDING_SECONDS,
                            canonical.duration),
                clip_folder / f"phrase-{number:04d}.wav",
                sample_rate=16000,
                channels=1,
            )
        except Exception as error:  # noqa: BLE001 - one clip must not stop the rest
            _log.warning("Could not cut the audio for a phrase to realign: %s", error)
            continue
        try:
            outcomes = timing_rules.realign(
                phrase,
                aligner,
                Path(clip.path),
                canonical_offset=clip.canonical_offset,
                language=language,
            )
            for token, outcome in zip(phrase, outcomes):
                timing_rules.apply_timing(token, outcome)
            measured += len(phrase)
        finally:
            try:
                Path(clip.path).unlink(missing_ok=True)
            except OSError:
                pass
        reporter.stage(
            "timing", number / len(phrases), f"Measured {number} of {len(phrases)} phrases."
        )
    try:
        clip_folder.rmdir()
    except OSError:
        pass
    reporter.stage("timing", 1.0, f"{measured} corrected words have been timed again.")
    return tokens


#: How many phrases one recording may send for re-measurement. Each is a
#: request of its own, and past a few hundred the recording is one the
#: services could not agree on, where a person is going to listen anyway.
_MAXIMUM_ALIGNMENT_PHRASES = 300

#: Audio either side of a phrase in the clip sent to the aligner. Enough
#: to catch a word boundary the inherited span clipped; not so much that
#: the clip holds speech the text does not mention, which an aligner has
#: to force onto the words it was given.
_ALIGNMENT_PADDING_SECONDS = 0.35

#: Words this close together, in time, are one phrase for alignment.
_PHRASE_GAP_SECONDS = 0.75


def _phrases_to_realign(tokens: Sequence[FinalToken]) -> list[list[FinalToken]]:
    """Gather the words worth measuring again into phrases of neighbours.

    Adjacent words in the transcript whose known spans nearly touch are one
    phrase. A word with no span at all joins the phrase of its neighbours,
    because a measurement might yet find it; a word with no span and no
    timed neighbour cannot be placed and is left alone.
    """
    phrases: list[list[FinalToken]] = []
    current: list[FinalToken] = []
    last_end: float | None = None
    for token in tokens:
        wanted = timing_rules.needs_forced_alignment(token)
        if not wanted:
            if current:
                phrases.append(current)
            current, last_end = [], None
            continue
        if token.start is not None and last_end is not None and token.start - last_end > (
            _PHRASE_GAP_SECONDS
        ):
            phrases.append(current)
            current = []
        current.append(token)
        if token.end is not None:
            last_end = token.end
    if current:
        phrases.append(current)
    return [phrase for phrase in phrases if _phrase_audio_span(phrase) is not None]


def _phrase_audio_span(phrase: Sequence[FinalToken]) -> AudioSpan | None:
    """Where the phrase is in the recording, or None if that is not known.

    Only a word's own span counts. A word with no span of its own has the
    gap between its neighbours as the region it might be in, and a phrase
    made only of such words, an insertion one service heard and the
    backbone did not, has a region that is often nothing at all. Cutting a
    clip of the neighbours and making an aligner place the inserted words
    in it would give words nobody measured a measured-looking time; that is
    the whole-recording mistake again, at a smaller scale. So a phrase is
    placed by the words in it that were timed, and an inserted word is only
    measured when it sits inside a phrase that has some.
    """
    starts = [token.start for token in phrase if token.start is not None]
    ends = [token.end for token in phrase if token.end is not None]
    if not starts:
        return None
    start = min(starts)
    end = max(ends) if ends else start
    return AudioSpan(start, max(start, end))


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
    """Save the transcript and its exports, and say how it went.

    A run that produced no words never replaces an earlier transcript that
    may hold them. That earlier file is where the user's review corrections
    live, and a run that failed or was stopped has nothing to put in its
    place.
    """
    transcript.completed_at = _now()
    if stopped:
        transcript.stopped = True
        transcript.warnings.append(
            "The run was stopped before it finished, so this transcript is "
            "incomplete."
        )
    if _keeps_earlier_transcript(transcript, store):
        transcript.warnings.append(
            "This recording already had a transcript, and this run produced no "
            "words, so the earlier transcript and its exports were kept unchanged."
        )
        _log.info(
            "Kept the earlier transcript of %s, because this run produced no words.",
            transcript.recording_name,
        )
        return transcript
    if not store.save(transcript):
        transcript.warnings.append(
            f"The transcript could not be saved to {store.transcript_path}. If another "
            "program was holding the old file open, a copy of the new one was kept at "
            f"{store.unsaved_transcript_path}; close that program and move it into place."
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


def _keeps_earlier_transcript(transcript: Transcript, store: TranscriptStore) -> bool:
    """Whether an earlier transcript must be left alone rather than replaced.

    Only a new transcript with no words is held back. An earlier file that
    cannot be read is kept too: it may be locked or damaged rather than
    empty, and replacing it would lose whatever it still holds. An earlier
    transcript that itself has no words has nothing to lose, so it is
    replaced as before.
    """
    if transcript.tokens or not store.has_transcript:
        return False
    earlier = store.load()
    return earlier is None or bool(earlier.tokens)


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
