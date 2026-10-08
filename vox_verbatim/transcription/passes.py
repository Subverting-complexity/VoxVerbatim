"""Running the recording past every service, and gathering what they said.

This is the first half of the work: turning one audio file into several
independent opinions about it, all measured against the same clock. The
second half, deciding which opinion to believe, happens elsewhere.

Three things make this more than a loop over some API calls.

The services run at the same time rather than one after another. A recording
takes each of them minutes, and running three in sequence would take three
times as long for no benefit, since they do not depend on each other. The
specification asks for this explicitly, and it is the difference between
waiting a few minutes and waiting a quarter of an hour.

Each service gets the audio in the shape it can accept. Most of them will
take an hour-long recording whole. OpenAI will not: its limit is 25 MB,
which a long recording passes easily. So OpenAI, and only OpenAI, gets a
planned set of chunks, each carrying the exact offset that puts its words
back on the canonical timeline. Nothing else is cut up merely to keep it
company, because chunking costs accuracy at every seam and there is no
reason to pay that where it is not required.

And a service that fails is not allowed to take the run down with it. The
whole architecture rests on carrying on: losing Microsoft should cost a
little accuracy and nothing else, and even losing the backbone should
produce a transcript rather than an error message. Every failure here ends
up as a sentence in the transcript's warnings, which the user reads at the
end.

The chunk files are temporary. Each service's chunks are deleted as soon as
that service's pass has been merged, and the chunk folder goes with them
once it is empty. Fifteen recordings at three hours each would otherwise
leave several gigabytes of re-encoded audio behind, and nothing reads the
chunks after the merge: the provenance record names the canonical time
range each chunk covered, which is what a later question needs.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from vox_verbatim.transcription import chunking
from vox_verbatim.transcription.context import ContextPackage, adapt_for
from vox_verbatim.transcription.model import (
    CanonicalAudio,
    ChunkRecord,
    Language,
    Provider,
    ProviderRequestRecord,
    ProviderResult,
    ProviderToken,
    RecordingConfiguration,
)
from vox_verbatim.transcription.providers.base import (
    CancelCheck,
    TranscriptionProvider,
    TranscriptionRequest,
)
from vox_verbatim.transcription.store import TranscriptStore

_log = logging.getLogger(__name__)

#: Told how far the passes have come, from 0 to 1, and what is happening in
#: words a person can read. The sentence is shown on screen and read out, so
#: it is written for a listener rather than as a log line.
PassProgress = Callable[[float, str], None]


@dataclass
class PassOutcome:
    """What every service said about one recording, and how it went."""

    results: dict[Provider, ProviderResult] = field(default_factory=dict)
    requests: list[ProviderRequestRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cancelled: bool = False

    @property
    def succeeded_providers(self) -> tuple[Provider, ...]:
        return tuple(
            provider for provider, result in self.results.items() if result.succeeded
        )

    @property
    def failed_providers(self) -> tuple[Provider, ...]:
        return tuple(
            provider for provider, result in self.results.items() if not result.succeeded
        )

    @property
    def any_succeeded(self) -> bool:
        return bool(self.succeeded_providers)


def run_passes(
    canonical: CanonicalAudio,
    providers: dict[Provider, TranscriptionProvider],
    context: ContextPackage,
    configuration: RecordingConfiguration,
    store: TranscriptStore,
    chunk_folder: Path,
    progress: PassProgress | None = None,
    cancelled: CancelCheck | None = None,
    chunk_overlap_seconds: float = 2.0,
) -> PassOutcome:
    """Send the recording to every service at once and collect the answers.

    Returns once every service has answered or failed. Never raises for a
    service failure: each one becomes a failed result and a warning, and the
    services that did answer are still returned.
    """
    outcome = PassOutcome()
    if not providers:
        outcome.warnings.append(
            "No transcription service is switched on, so there was nothing to run. "
            "Open Settings and switch on at least one of ElevenLabs, OpenAI or "
            "Microsoft."
        )
        return outcome

    # Each service reports its own progress as a fraction of its own work.
    # They are averaged into one number, so the bar moves smoothly rather
    # than jumping as each service finishes. The lock is needed because the
    # updates arrive from several threads at once.
    shares: dict[Provider, float] = {provider: 0.0 for provider in providers}
    lock = threading.Lock()

    def report(provider: Provider, fraction: float, message: str) -> None:
        if progress is None:
            return
        with lock:
            shares[provider] = max(0.0, min(1.0, fraction))
            overall = sum(shares.values()) / len(shares)
        progress(overall, message)

    # What each service returned, kept per service until every thread has
    # finished so that the writing can happen on one thread afterwards.
    gathered: dict[Provider, list[_ChunkEntry]] = {}

    _log.info(
        "Sending %s to %d services: %s",
        canonical.path,
        len(providers),
        ", ".join(provider.display_name for provider in providers),
    )
    if progress is not None:
        progress(0.0, _starting_sentence(providers))

    # One thread per service. There are at most five, each spending nearly
    # all its time waiting on a network, so a thread apiece is the simplest
    # thing that works and there is nothing to be gained from a pool that
    # rations them.
    with ThreadPoolExecutor(max_workers=max(1, len(providers))) as pool:
        futures: dict[Provider, Future[_OneProviderOutcome]] = {}
        for provider, adapter in providers.items():
            futures[provider] = pool.submit(
                _run_one_provider,
                provider,
                adapter,
                canonical,
                context,
                configuration,
                chunk_folder,
                lambda fraction, message, p=provider: report(p, fraction, message),
                cancelled,
                chunk_overlap_seconds,
            )
        for provider, future in futures.items():
            try:
                one = future.result()
            except Exception as error:  # noqa: BLE001 - one thread must not end the run
                # _run_one_provider catches what it can, but a failure in
                # the thread's own plumbing still surfaces here, and one
                # service's thread dying must cost that service and nothing
                # else.
                _log.exception("The %s pass failed unexpectedly.", provider.display_name)
                one = _OneProviderOutcome(
                    result=ProviderResult(
                        provider=provider,
                        error=f"{provider.display_name} failed unexpectedly: {error}",
                    ),
                    warnings=[
                        f"{provider.display_name} failed unexpectedly and was not used: {error}"
                    ],
                )
            outcome.results[provider] = one.result
            outcome.warnings.extend(one.warnings)
            gathered[provider] = one.entries

    _remove_chunk_folder_if_empty(chunk_folder)

    # The raw answers are written after every service has finished rather
    # than as each one arrives, so that several threads are never creating
    # files in the same folder at once. The store numbers its files from the
    # highest already present, which is not safe against that.
    for provider, entries in gathered.items():
        outcome.requests.extend(_store_raw_responses(store, provider, entries, outcome))

    if cancelled is not None and cancelled():
        outcome.cancelled = True

    if progress is not None:
        progress(1.0, _finished_sentence(outcome))
    return outcome


@dataclass(frozen=True)
class _ChunkEntry:
    """One request to one service, and the untouched answer it produced.

    The two are kept together because they only mean anything together: the
    provenance record says what was asked and which file holds the answer,
    and the answer is unexplainable without the record. Keeping them in
    separate lists and pairing them by position later is exactly how they
    come apart when a chunk fails and one list gains an entry the other does
    not.
    """

    request: ProviderRequestRecord | None
    payload: object | None
    chunk_index: int


@dataclass
class _OneProviderOutcome:
    result: ProviderResult
    entries: list[_ChunkEntry] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def requests(self) -> list[ProviderRequestRecord]:
        return [entry.request for entry in self.entries if entry.request is not None]


def _run_one_provider(
    provider: Provider,
    adapter: TranscriptionProvider,
    canonical: CanonicalAudio,
    context: ContextPackage,
    configuration: RecordingConfiguration,
    chunk_folder: Path,
    report: Callable[[float, str], None],
    cancelled: CancelCheck | None,
    chunk_overlap_seconds: float,
) -> _OneProviderOutcome:
    """Run one service over the whole recording, chunking only if it must.

    This runs on its own thread, so it must not raise. Anything that goes
    wrong comes back as a failed result with a sentence explaining it.
    """
    name = provider.display_name
    problem = adapter.describe_configuration_problem()
    if problem is not None:
        return _OneProviderOutcome(
            result=ProviderResult(provider=provider, error=problem),
            warnings=[f"{name} was not used: {problem}"],
        )
    plans: list[ChunkRecord] = []
    try:
        plans = chunking.plan_and_write_chunks(
            canonical,
            provider,
            adapter.capabilities,
            chunk_folder,
            options=chunking.ChunkingOptions(overlap_seconds=chunk_overlap_seconds),
        )
    except Exception as error:  # a chunking failure must not escape the thread
        _log.exception("Could not prepare the audio for %s.", name)
        return _OneProviderOutcome(
            result=ProviderResult(
                provider=provider,
                error=f"The audio could not be prepared for {name}: {error}",
            ),
            warnings=[f"{name} was not used because its audio could not be prepared."],
        )

    if len(plans) > 1:
        report(0.0, f"{name} is working through {len(plans)} pieces of the recording.")
    else:
        report(0.0, f"{name} is transcribing the recording.")

    try:
        return _transcribe_chunks(
            provider, adapter, canonical, context, configuration, plans, report, cancelled
        )
    except Exception as error:  # noqa: BLE001 - this thread must not raise
        _log.exception("The %s pass failed unexpectedly.", name)
        return _OneProviderOutcome(
            result=ProviderResult(
                provider=provider,
                error=f"{name} failed unexpectedly: {error}",
            ),
            warnings=[f"{name} failed unexpectedly and was not used: {error}"],
        )
    finally:
        # The chunks have served their purpose once the pass is over, whether
        # it succeeded or not. A failed pass keeps nothing either: the record
        # of what was sent names the time range, and the audio can be cut
        # again from the canonical file if anybody needs it.
        _delete_chunk_files(canonical, plans)


def _transcribe_chunks(
    provider: Provider,
    adapter: TranscriptionProvider,
    canonical: CanonicalAudio,
    context: ContextPackage,
    configuration: RecordingConfiguration,
    plans: list[ChunkRecord],
    report: Callable[[float, str], None],
    cancelled: CancelCheck | None,
) -> _OneProviderOutcome:
    """Send each chunk, then join the answers. May raise; the caller catches."""
    name = provider.display_name
    entries: list[_ChunkEntry] = []
    started = time.monotonic()
    prepared = adapt_for(context, provider, adapter.capabilities)
    tokens: list[ProviderToken] = []
    warnings: list[str] = []
    speakers: list[str] = []
    detected = Language.UNKNOWN
    failures = 0
    first_failure: str | None = None
    succeeded_plans: list[ChunkRecord] = []

    for number, chunk in enumerate(plans, start=1):
        if cancelled is not None and cancelled():
            warnings.append(f"{name} was stopped part way through.")
            break
        request = TranscriptionRequest(
            audio_path=Path(chunk.path or canonical.path),
            canonical_offset=chunk.canonical_offset,
            duration=chunk.canonical_end - chunk.canonical_start,
            languages=_languages_for(configuration),
            expected_speaker_count=configuration.expected_speaker_count,
            diarise=adapter.capabilities.diarisation,
            vocabulary_terms=prepared.terms,
            context_prompt=prepared.prompt,
            verbatim=True,
            extra_parameters=dict(prepared.parameters),
            chunk_index=chunk.chunk_index,
        )
        result = adapter.transcribe(request, cancelled)
        # The record and the untouched answer are kept together, and kept
        # even when the chunk failed: a response explaining a refusal is
        # often the most useful thing in the folder afterwards.
        if result.request is not None or result.raw_response is not None:
            entries.append(
                _ChunkEntry(
                    request=result.request,
                    payload=result.raw_response,
                    chunk_index=chunk.chunk_index,
                )
            )
        if not result.succeeded:
            failures += 1
            if first_failure is None:
                first_failure = result.error
            # The time range is named so a reader knows which stretch of the
            # recording this service has no words for, without having to
            # work it out from the chunk number.
            warnings.append(
                f"{name} could not transcribe part {number} of {len(plans)}, covering "
                f"{_clock(chunk.canonical_start)}\u2013{_clock(chunk.canonical_end)}: "
                f"{result.error}"
            )
        else:
            succeeded_plans.append(chunk)
            tokens.extend(result.tokens)
            for speaker in result.speakers:
                if speaker not in speakers:
                    speakers.append(speaker)
            if detected is Language.UNKNOWN:
                detected = result.detected_language
        report(number / len(plans), f"{name} has finished part {number} of {len(plans)}.")

    # Every chunk failing is the same thing as the service failing, and
    # should be reported that way rather than as an empty success.
    if failures and failures == len(plans):
        first = next((entry.request for entry in entries if entry.request), None)
        # The reason the first chunk gave is carried onto the result rather
        # than being flattened into "it did not work". Every chunk failing
        # almost always has one cause, and it is usually something the user
        # can act on: a refused key, a wrong endpoint, an exhausted quota.
        # A caller reading only ``error`` should get that sentence, not a
        # summary that sends them looking through the warnings for it.
        detail = f" {first_failure}" if first_failure else ""
        return _OneProviderOutcome(
            result=ProviderResult(
                provider=provider,
                error=f"{name} did not transcribe any part of the recording.{detail}",
                request=first,
            ),
            entries=entries,
            warnings=warnings,
        )

    if len(succeeded_plans) > 1:
        # Only the chunks that answered are merged. The merge joins each
        # chunk to the one before it on the overlap they share, and a failed
        # chunk in the list would have no words to match against: the join
        # would fall back to cutting a nominal number of words off the start
        # of the chunk after it, and real words would be lost at a place
        # where words are already missing.
        merged = chunking.merge_overlapping_tokens(tokens, succeeded_plans)
        tokens = list(merged.tokens)
        for join in getattr(merged, "uncertain_joins", ()):
            # A seam that could not be matched confidently is not a failure,
            # but it is exactly the sort of place a word goes missing or gets
            # said twice, so the user is told rather than left to find it.
            warnings.append(
                f"{name}: the join between two pieces of the recording could not be "
                f"matched confidently. {join.description}"
            )

    tokens = _renumber(tokens)
    elapsed = time.monotonic() - started
    _log.info("%s returned %d words in %.1f seconds.", name, len(tokens), elapsed)
    return _OneProviderOutcome(
        result=ProviderResult(
            provider=provider,
            tokens=tokens,
            request=next((entry.request for entry in entries if entry.request), None),
            detected_language=detected,
            speakers=speakers,
        ),
        entries=entries,
        warnings=warnings,
    )


def _clock(seconds: float) -> str:
    """Seconds as mm:ss, or h:mm:ss past an hour, for a warning a person reads."""
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _delete_chunk_files(canonical: CanonicalAudio, plans: list[ChunkRecord]) -> None:
    """Remove one service's chunk files now that its pass is over.

    Only files that are not the canonical recording are touched: a service
    that took the recording whole has a plan whose path is the canonical
    path, and that file belongs to the user. Failure to delete is logged and
    nothing more, because a leftover file is untidy and a crashed pass is
    not.

    The folder is left for :func:`_remove_chunk_folder_if_empty`, which runs
    once every thread has finished. Removing it from here would race another
    service's thread, which creates the folder immediately before writing
    each chunk into it.
    """
    canonical_path = Path(canonical.path).resolve()
    for chunk in plans:
        if not chunk.path:
            continue
        path = Path(chunk.path)
        try:
            if path.resolve() == canonical_path:
                continue
            path.unlink(missing_ok=True)
        except OSError:
            _log.warning("The chunk file %s could not be deleted.", path, exc_info=True)


def _remove_chunk_folder_if_empty(chunk_folder: Path) -> None:
    """Remove the chunk folder once every service has deleted its chunks.

    A folder that still holds something, whether a chunk that could not be
    deleted or a file somebody else put there, is left alone.
    """
    try:
        if chunk_folder.is_dir() and not any(chunk_folder.iterdir()):
            chunk_folder.rmdir()
    except OSError:
        _log.debug("The chunk folder %s was left in place.", chunk_folder, exc_info=True)


def _languages_for(configuration: RecordingConfiguration) -> tuple[Language, ...]:
    """The languages a service may consider for this recording.

    Afrikaans appears only when the user said it might. That is the whole
    point of the setting: not to ignore Afrikaans results, but never to ask
    for them, so that a recording containing none cannot be given any.
    """
    if configuration.afrikaans_enabled:
        return (Language.ENGLISH, Language.GERMAN, Language.AFRIKAANS)
    return (Language.ENGLISH, Language.GERMAN)


def _renumber(tokens: list[ProviderToken]) -> list[ProviderToken]:
    """Give the gathered words one running sequence.

    Each chunk numbers its own words from zero, so a recording sent in four
    pieces comes back with four words numbered zero. Alignment and
    provenance both refer to words by their number, so the numbering has to
    be made continuous once the pieces are joined. The chunk each word came
    from is kept on the word itself, so nothing is lost by renumbering.
    """
    from dataclasses import replace

    return [replace(token, index=number) for number, token in enumerate(tokens)]


def _store_raw_responses(
    store: TranscriptStore,
    provider: Provider,
    entries: list[_ChunkEntry],
    outcome: PassOutcome,
) -> list[ProviderRequestRecord]:
    """Write what a service returned, exactly as it returned it.

    The adapters do not write anything themselves, deliberately: an adapter
    that knew where transcripts are kept would be much harder to test and
    would have to be given a folder it has no other use for. So they carry
    the payload back on the result, and this writes it and puts the file
    name into the provenance, so that afterwards the record of what was
    asked and the answer that came back can be found from each other.

    Every chunk gets its own file, because a service sent a recording in
    four pieces answered four times and there is no single response to keep.

    A response that cannot be written is worth a warning rather than a
    failure. The transcript is still good; what has been lost is the ability
    to explain it later, and the user should be told that plainly rather
    than discovering it on the day they come looking.
    """
    from dataclasses import replace

    records: list[ProviderRequestRecord] = []
    for entry in entries:
        record = entry.request
        if entry.payload is not None:
            name = store.write_raw_response(
                provider,
                entry.payload,
                purpose="transcription",
                chunk_index=entry.chunk_index if len(entries) > 1 else None,
            )
            if name is None:
                outcome.warnings.append(
                    f"The answer from {provider.display_name} could not be saved, so "
                    "this transcription cannot be explained or reprocessed later."
                )
            elif record is not None:
                record = replace(record, raw_response_file=name)
        if record is not None:
            records.append(record)

    # The result keeps the updated first record, so that anything reading
    # the result rather than the outcome still finds the file name.
    result = outcome.results.get(provider)
    if result is not None:
        if records:
            result.request = records[0]
        # The payloads have homes of their own now, so they are dropped
        # rather than carried in memory for the rest of the run.
        result.raw_response = None
    return records


def _starting_sentence(providers: dict[Provider, TranscriptionProvider]) -> str:
    names = [provider.display_name for provider in providers]
    if len(names) == 1:
        return f"Sending the recording to {names[0]}."
    listed = ", ".join(names[:-1]) + f" and {names[-1]}"
    return f"Sending the recording to {listed}, all at once."


def _finished_sentence(outcome: PassOutcome) -> str:
    succeeded = len(outcome.succeeded_providers)
    failed = len(outcome.failed_providers)
    if failed == 0:
        return f"All {succeeded} services answered."
    if succeeded == 0:
        return "No service answered."
    services = "service" if failed == 1 else "services"
    return f"{succeeded} answered; {failed} {services} did not."


def stamp() -> str:
    """The current local time, as the provenance records write it."""
    return datetime.now().isoformat(timespec="seconds")
