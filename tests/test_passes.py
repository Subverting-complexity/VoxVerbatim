"""Tests for sending one recording to every service and gathering the answers.

Nothing here touches a network or a provider library. Every service is a
fake adapter that implements the same contract the real ones do and hands
back words that were written by the test, so what is being tested is the
gathering rather than anybody's speech recognition.

The audio is real, because the chunk planner measures the file before it
decides what to send, and a file that cannot be opened would take a
different path through the planner than the one a user's recording takes.
It is two seconds of a tone written by PyAV, which is the same way the
enhancement tests make their recordings.

The failure paths get most of the attention here, because they are what the
whole design rests on. A service that does not answer must cost a little
accuracy and nothing else, and there is no way to be sure of that except by
breaking each service in turn and looking at what came back.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

import pytest

from vox_verbatim.transcription import passes
from vox_verbatim.transcription.canonical import CanonicalAudioError, prepare_canonical_audio
from vox_verbatim.transcription.context import build_context_package
from vox_verbatim.transcription.model import (
    CanonicalAudio,
    Language,
    Provider,
    ProviderRequestRecord,
    ProviderResult,
    ProviderToken,
    RecordingConfiguration,
)
from vox_verbatim.transcription.providers.base import (
    ProviderCapabilities,
    ProviderError,
    TranscriptionProvider,
    TranscriptionRequest,
)
from vox_verbatim.transcription.store import TranscriptStore

from tests.conftest import write_real_audio


# -- The fakes -----------------------------------------------------------
#
# These stand in for the five real adapters. They live in this file rather
# than in a helper module because this is the file that tests the contract
# they implement; the pipeline tests import them from here so that the two
# suites cannot drift apart into two different ideas of what a service does.


#: What ElevenLabs can do, as far as anything here needs to know: it times
#: every word and it tells the speakers apart, which is what makes it the
#: backbone everything else is measured against.
BACKBONE_CAPABILITIES = ProviderCapabilities(
    word_timings=True,
    diarisation=True,
    speaker_count_hint=True,
    word_confidence=True,
    vocabulary_biasing=True,
)

#: What OpenAI and Microsoft can do: hear words, and nothing else. Neither
#: times a word, which is exactly why neither can be the backbone.
TEXT_ONLY_CAPABILITIES = ProviderCapabilities(
    word_timings=False,
    diarisation=False,
    vocabulary_biasing=True,
    context_prompt=True,
    language_detection=True,
)


def spoken_words(
    provider: Provider,
    sentence: str,
    timed: bool = False,
    speaker: str | None = None,
    seconds_each: float = 0.2,
) -> list[ProviderToken]:
    """Turn a sentence into the words one service would return for it.

    ``timed`` gives each word a slot of its own on the canonical clock,
    which is what separates a service that can be the backbone from one that
    cannot.
    """
    tokens: list[ProviderToken] = []
    for index, word in enumerate(sentence.split()):
        start = index * seconds_each
        tokens.append(
            ProviderToken(
                provider=provider,
                index=index,
                text=word,
                start=start if timed else None,
                end=start + seconds_each * 0.8 if timed else None,
                speaker=speaker if timed else None,
                confidence=0.97,
                language=Language.ENGLISH,
            )
        )
    return tokens


class FakeService(TranscriptionProvider):
    """One speech-to-text service, with the network taken out.

    It answers with words the test wrote, remembers every request it was
    given so a test can say what the service was actually asked for, and can
    be told to fail in each of the ways a real service fails: refusing
    because it is not set up, returning an error, or throwing.
    """

    def __init__(
        self,
        provider: Provider,
        tokens: list[ProviderToken] | None = None,
        capabilities: ProviderCapabilities = TEXT_ONLY_CAPABILITIES,
        model: str = "fake-model-1",
        speakers: tuple[str, ...] = (),
        error: str | None = None,
        configuration_problem: str | None = None,
        raises: Exception | None = None,
        raw_response: object | None = None,
        before_answering=None,
    ) -> None:
        self.provider = provider
        self.capabilities = capabilities
        self._model = model
        self._tokens = tokens if tokens is not None else []
        self._speakers = list(speakers)
        self._error = error
        self._configuration_problem = configuration_problem
        self._raises = raises
        self._raw_response = (
            raw_response
            if raw_response is not None
            else {"service": provider.value, "words": [token.text for token in self._tokens]}
        )
        self._before_answering = before_answering
        self.requests_received: list[TranscriptionRequest] = []

    @property
    def model_identifier(self) -> str:
        return self._model

    def is_configured(self) -> bool:
        return self._configuration_problem is None

    def describe_configuration_problem(self) -> str | None:
        return self._configuration_problem

    def _transcribe(self, request, cancelled=None) -> ProviderResult:
        self.requests_received.append(request)
        if self._before_answering is not None:
            self._before_answering()
        if self._raises is not None:
            raise self._raises
        # A real adapter stamps every word with the piece of the recording it
        # came back in, and the merging of the pieces depends on it.
        tokens = [replace(token, chunk_index=request.chunk_index) for token in self._tokens]
        record = ProviderRequestRecord(
            provider=self.provider,
            model_identifier=self._model,
            request_parameters=dict(request.extra_parameters),
            language_configuration=",".join(
                language.value for language in request.languages
            ),
            vocabulary_terms=request.vocabulary_terms,
            started_at=passes.stamp(),
            succeeded=self._error is None,
            error=self._error,
        )
        if self._error is not None:
            return ProviderResult(
                provider=self.provider,
                request=record,
                error=self._error,
                raw_response=self._raw_response,
            )
        return ProviderResult(
            provider=self.provider,
            tokens=tokens,
            request=record,
            detected_language=Language.ENGLISH,
            speakers=list(self._speakers),
            raw_response=self._raw_response,
        )


SENTENCE = "the invoice came to fifteen thousand rand on Tuesday"


def three_services(**overrides) -> dict[Provider, FakeService]:
    """The three services a full pass normally uses, all agreeing.

    Pass ``elevenlabs=...``, ``openai=...`` or ``microsoft=...`` to replace
    one of them with a broken version, which is how every failure test below
    is written.
    """
    built = {
        Provider.ELEVENLABS: FakeService(
            Provider.ELEVENLABS,
            spoken_words(Provider.ELEVENLABS, SENTENCE, timed=True, speaker="speaker_0"),
            capabilities=BACKBONE_CAPABILITIES,
            model="scribe-fake",
            speakers=("speaker_0",),
        ),
        Provider.OPENAI: FakeService(
            Provider.OPENAI,
            spoken_words(Provider.OPENAI, SENTENCE),
            model="openai-fake",
        ),
        Provider.MICROSOFT: FakeService(
            Provider.MICROSOFT,
            spoken_words(Provider.MICROSOFT, SENTENCE),
            model="microsoft-fake",
        ),
    }
    for name, service in overrides.items():
        built[Provider[name.upper()]] = service
    return built


# -- The recording -------------------------------------------------------


@pytest.fixture
def recording(tmp_path):
    """Two seconds of real audio, and the folder its transcript belongs in."""
    path = tmp_path / "meeting.wav"
    write_real_audio(path, codec="pcm_s16le", seconds=2.0, layout="mono", rate=16000)
    return path


@pytest.fixture
def store(recording) -> TranscriptStore:
    return TranscriptStore(recording)


@pytest.fixture
def canonical(recording, store) -> CanonicalAudio:
    return prepare_canonical_audio(recording, store.folder)


def run(
    canonical: CanonicalAudio,
    store: TranscriptStore,
    services: dict[Provider, FakeService],
    configuration: RecordingConfiguration | None = None,
    cancelled=None,
    progress=None,
) -> passes.PassOutcome:
    """Run one pass over the recording with the given fake services."""
    configuration = configuration or RecordingConfiguration()
    return passes.run_passes(
        canonical=canonical,
        providers=services,
        context=build_context_package(configuration),
        configuration=configuration,
        store=store,
        chunk_folder=store.folder / "chunks",
        progress=progress,
        cancelled=cancelled,
    )


# -- Everything working --------------------------------------------------


def test_every_service_that_answers_comes_back_with_its_own_words(canonical, store):
    services = three_services()

    outcome = run(canonical, store, services)

    assert set(outcome.succeeded_providers) == set(services)
    assert not outcome.failed_providers
    assert outcome.any_succeeded
    assert not outcome.warnings
    for provider, result in outcome.results.items():
        assert [token.text for token in result.tokens] == SENTENCE.split()
        assert result.provider is provider


def test_the_words_of_one_service_come_back_in_one_running_sequence(canonical, store):
    """Indexes are what provenance points at, so they must not repeat."""
    outcome = run(canonical, store, three_services())

    for result in outcome.results.values():
        assert [token.index for token in result.tokens] == list(range(len(result.tokens)))


def test_each_service_is_told_what_the_recording_is_about(canonical, store):
    configuration = RecordingConfiguration(
        expected_speaker_count=3,
        recording_context="A meeting about an invoice.",
    )
    services = three_services()

    run(canonical, store, services, configuration=configuration)

    for service in services.values():
        request = service.requests_received[0]
        assert request.expected_speaker_count == 3
        assert request.verbatim is True
        assert request.canonical_offset == 0.0


def test_a_service_is_only_asked_to_diarise_when_it_can(canonical, store):
    """Asking a service for something it cannot do invites a made-up answer."""
    services = three_services()

    run(canonical, store, services)

    assert services[Provider.ELEVENLABS].requests_received[0].diarise is True
    assert services[Provider.OPENAI].requests_received[0].diarise is False
    assert services[Provider.MICROSOFT].requests_received[0].diarise is False


def test_the_services_are_all_worked_on_at_the_same_time(canonical, store):
    """Three services in sequence would take three times as long for nothing.

    Each fake waits at a barrier that only opens once all three have reached
    it, so a run that started them one after another would never get past
    the first and would fail on the barrier's own timeout.
    """
    gate = threading.Barrier(3, timeout=10.0)
    services = three_services()
    for service in services.values():
        service._before_answering = gate.wait

    outcome = run(canonical, store, services)

    assert set(outcome.succeeded_providers) == set(services)


def test_the_progress_is_reported_in_a_sentence_a_person_can_read(canonical, store):
    reported: list[tuple[float, str]] = []

    run(canonical, store, three_services(), progress=lambda f, m: reported.append((f, m)))

    assert reported[0][0] == 0.0
    assert "ElevenLabs Scribe" in reported[0][1]
    assert reported[-1] == (1.0, "All 3 services answered.")


# -- One service failing -------------------------------------------------


def test_a_service_that_returns_an_error_loses_only_itself(canonical, store):
    services = three_services(
        microsoft=FakeService(Provider.MICROSOFT, error="Azure said no.")
    )

    outcome = run(canonical, store, services)

    assert outcome.failed_providers == (Provider.MICROSOFT,)
    assert set(outcome.succeeded_providers) == {Provider.ELEVENLABS, Provider.OPENAI}
    assert not outcome.results[Provider.MICROSOFT].succeeded
    assert any("Microsoft MAI" in warning for warning in outcome.warnings)


def test_a_service_that_throws_becomes_a_failed_result_rather_than_an_exception(
    canonical, store
):
    """A broken adapter must not take the run down with it."""
    services = three_services(
        openai=FakeService(Provider.OPENAI, raises=ProviderError("The key was refused."))
    )

    outcome = run(canonical, store, services)

    assert outcome.failed_providers == (Provider.OPENAI,)
    assert not outcome.results[Provider.OPENAI].succeeded
    # The result says the service produced nothing; the reason it produced
    # nothing is kept in the warning, which is what the user reads.
    assert any("The key was refused." in warning for warning in outcome.warnings)
    assert outcome.any_succeeded


def test_a_service_that_throws_something_nobody_expected_is_caught_as_well(
    canonical, store
):
    services = three_services(
        openai=FakeService(Provider.OPENAI, raises=RuntimeError("A bug in the adapter."))
    )

    outcome = run(canonical, store, services)

    assert not outcome.results[Provider.OPENAI].succeeded
    assert set(outcome.succeeded_providers) == {Provider.ELEVENLABS, Provider.MICROSOFT}


def test_a_service_that_is_not_set_up_is_never_called(canonical, store):
    """A missing key is reported as the plain fact it is, before any request."""
    unconfigured = FakeService(
        Provider.MICROSOFT,
        configuration_problem="Microsoft MAI is not set up: it has no endpoint.",
    )
    services = three_services(microsoft=unconfigured)

    outcome = run(canonical, store, services)

    assert unconfigured.requests_received == []
    assert not outcome.results[Provider.MICROSOFT].succeeded
    assert any("was not used" in warning for warning in outcome.warnings)


# -- Everything failing --------------------------------------------------


def test_every_service_failing_comes_back_as_an_outcome_rather_than_an_exception(
    canonical, store
):
    services = {
        Provider.ELEVENLABS: FakeService(Provider.ELEVENLABS, error="No answer."),
        Provider.OPENAI: FakeService(Provider.OPENAI, error="No answer."),
        Provider.MICROSOFT: FakeService(Provider.MICROSOFT, error="No answer."),
    }

    outcome = run(canonical, store, services)

    assert not outcome.any_succeeded
    assert set(outcome.failed_providers) == set(services)
    assert len(outcome.warnings) >= 3


def test_no_service_switched_on_says_so_rather_than_running_nothing_quietly(
    canonical, store
):
    outcome = run(canonical, store, {})

    assert not outcome.any_succeeded
    assert outcome.warnings
    assert "Settings" in outcome.warnings[0]


# -- The untouched answers -----------------------------------------------


def named_files(outcome: passes.PassOutcome) -> dict[Provider, list[str]]:
    """The raw-response file each request record points at, per service."""
    found: dict[Provider, list[str]] = {}
    for record in outcome.requests:
        found.setdefault(record.provider, []).append(record.raw_response_file)
    return found


def test_the_raw_answers_are_written_beside_the_recording_and_named_in_the_record(
    canonical, store
):
    """A transcript that cannot be explained later is worth much less."""
    outcome = run(canonical, store, three_services())

    written = store.raw_response_names()
    assert len(written) == 3
    named = named_files(outcome)
    for provider in outcome.results:
        [name] = named[provider]
        assert name in written
        assert provider.value in name
        assert "transcription" in name
        body = store.read_raw_response(name)
        assert body is not None and provider.value in body


def test_a_service_sent_the_recording_in_pieces_gets_one_file_for_each_piece(
    canonical, store
):
    """Four pieces means four answers, and there is no single response to keep.

    The service is given a limit small enough that two seconds of audio will
    not fit whole, which is what OpenAI's twenty-five megabytes does to an
    hour-long recording. Each piece is asked for separately, each answer is
    written separately, and each request record names its own file. A test
    that used one chunk would pass over the whole of this.
    """
    chunked = FakeService(
        Provider.OPENAI,
        spoken_words(Provider.OPENAI, SENTENCE),
        capabilities=replace(TEXT_ONLY_CAPABILITIES, maximum_file_bytes=40_000),
    )

    outcome = run(canonical, store, {Provider.OPENAI: chunked})

    pieces = len(chunked.requests_received)
    assert pieces > 1
    # Each piece was asked for with the offset that puts its words back on the
    # canonical clock, so the pieces are genuinely different requests.
    offsets = [request.canonical_offset for request in chunked.requests_received]
    assert offsets == sorted(offsets) and offsets[0] == 0.0 and offsets[-1] > 0.0

    written = store.raw_response_names()
    assert len(written) == pieces
    names = named_files(outcome)[Provider.OPENAI]
    assert len(names) == pieces
    assert len(set(names)) == pieces
    assert set(names) <= set(written)
    for index, name in enumerate(names):
        assert f"chunk{index:02d}" in name


def test_an_answer_that_cannot_be_written_is_a_warning_rather_than_a_failure(
    canonical, store, monkeypatch
):
    monkeypatch.setattr(
        TranscriptStore, "write_raw_response", lambda *args, **kwargs: None
    )

    outcome = run(canonical, store, three_services())

    assert outcome.any_succeeded
    assert any("could not be saved" in warning for warning in outcome.warnings)


def test_the_answer_to_a_failed_request_is_kept_as_well(canonical, store):
    """A response explaining a refusal is often the most useful thing there."""
    outcome = run(
        canonical,
        store,
        three_services(
            microsoft=FakeService(
                Provider.MICROSOFT,
                error="Azure said no.",
                raw_response={"error": "the model was not recognised"},
            )
        ),
    )

    [name] = named_files(outcome)[Provider.MICROSOFT]
    assert name in store.raw_response_names()
    assert "not recognised" in (store.read_raw_response(name) or "")


def test_a_second_run_writes_new_answers_beside_the_old_ones(canonical, store):
    """Provenance is immutable, so nothing already written may be replaced."""
    run(canonical, store, three_services())
    first = store.raw_response_names()

    run(canonical, store, three_services())
    second = store.raw_response_names()

    assert len(second) == 6
    assert set(first) < set(second)


# -- Stopping part way ---------------------------------------------------


def test_a_run_that_was_cancelled_says_so(canonical, store):
    outcome = run(canonical, store, three_services(), cancelled=lambda: True)

    assert outcome.cancelled is True


def test_a_run_that_was_not_cancelled_does_not_claim_to_have_been(canonical, store):
    outcome = run(canonical, store, three_services(), cancelled=lambda: False)

    assert outcome.cancelled is False


# -- The Afrikaans switch ------------------------------------------------


def test_afrikaans_is_never_asked_for_when_the_user_did_not_enable_it(canonical, store):
    """Off means never asking, not asking and then discarding the answer."""
    services = three_services()

    run(canonical, store, services, configuration=RecordingConfiguration())

    for service in services.values():
        request = service.requests_received[0]
        assert request.languages == (Language.ENGLISH, Language.GERMAN)
        assert Language.AFRIKAANS not in request.languages
        assert "af" not in request.extra_parameters.get("languages", [])


def test_afrikaans_reaches_the_services_only_when_the_user_enabled_it(canonical, store):
    services = three_services()
    configuration = RecordingConfiguration(afrikaans_enabled=True)

    run(canonical, store, services, configuration=configuration)

    for service in services.values():
        assert Language.AFRIKAANS in service.requests_received[0].languages
    assert "af" in services[Provider.OPENAI].requests_received[0].extra_parameters["languages"]


# -- A pass that breaks in the middle ---------------------------------------


def test_a_thread_that_dies_costs_only_its_own_service(canonical, store, monkeypatch):
    """Even the plumbing around an adapter must not take the run down."""
    original = passes._run_one_provider

    def run_one(provider, *arguments, **keywords):
        if provider is Provider.MICROSOFT:
            raise RuntimeError("the thread itself broke")
        return original(provider, *arguments, **keywords)

    monkeypatch.setattr(passes, "_run_one_provider", run_one)

    outcome = run(canonical, store, three_services())

    assert outcome.failed_providers == (Provider.MICROSOFT,)
    assert "the thread itself broke" in outcome.results[Provider.MICROSOFT].error
    assert any("the thread itself broke" in warning for warning in outcome.warnings)
    assert set(outcome.succeeded_providers) == {Provider.ELEVENLABS, Provider.OPENAI}


def test_a_context_that_cannot_be_adapted_is_a_failed_result(canonical, store, monkeypatch):
    def broken(*arguments, **keywords):
        raise ValueError("no context for you")

    monkeypatch.setattr(passes, "adapt_for", broken)

    outcome = run(canonical, store, three_services())

    assert outcome.succeeded_providers == ()
    assert all("no context for you" in result.error for result in outcome.results.values())


class FailingOnOnePiece(FakeService):
    """Answers every piece of the recording except the one named."""

    def __init__(self, *arguments, failing_chunk: int, **keywords) -> None:
        super().__init__(*arguments, **keywords)
        self.failing_chunk = failing_chunk

    def _transcribe(self, request, cancelled=None) -> ProviderResult:
        if request.chunk_index == self.failing_chunk:
            self.requests_received.append(request)
            return ProviderResult(provider=self.provider, error="the service hiccupped")
        return super()._transcribe(request, cancelled)


def chunked_service(failing_chunk: int) -> FailingOnOnePiece:
    return FailingOnOnePiece(
        Provider.OPENAI,
        spoken_words(Provider.OPENAI, SENTENCE),
        capabilities=replace(TEXT_ONLY_CAPABILITIES, maximum_file_bytes=40_000),
        failing_chunk=failing_chunk,
    )


def test_one_failed_piece_does_not_cut_words_off_the_piece_after_it(canonical, store):
    """Only the pieces that answered are merged.

    The merge joins each piece to the one before it on their shared overlap.
    A failed piece in that list has no words to match, so the join would have
    fallen back to cutting a nominal number of words off the start of the
    next piece, losing real words at the very place words are already missing.
    """
    service = chunked_service(failing_chunk=1)

    outcome = run(canonical, store, {Provider.OPENAI: service})

    pieces = len(service.requests_received)
    assert pieces >= 3, "the recording must have been cut into at least three pieces"
    result = outcome.results[Provider.OPENAI]
    assert result.succeeded
    words_per_piece = len(SENTENCE.split())
    # Every answered piece's words are there: the piece after the failed one
    # was not trimmed against a neighbour that never answered.
    answered = [token for token in result.tokens if token.chunk_index != 1]
    assert len(answered) == len(result.tokens)
    assert [token.text for token in result.tokens[:words_per_piece]] == SENTENCE.split()
    assert [token.text for token in result.tokens if token.chunk_index == 2][:2] == ["the", "invoice"]


def test_the_warning_for_a_failed_piece_names_the_time_it_covered(canonical, store):
    service = chunked_service(failing_chunk=1)

    outcome = run(canonical, store, {Provider.OPENAI: service})

    failed = [request for request in service.requests_received if request.chunk_index == 1][0]
    warning = next(warning for warning in outcome.warnings if "hiccupped" in warning)
    assert "part 2 of" in warning
    assert passes._clock(failed.canonical_offset) in warning
    assert "–" in warning


def test_clock_times_read_as_minutes_and_seconds():
    assert passes._clock(0) == "00:00"
    assert passes._clock(65.4) == "01:05"
    assert passes._clock(3725) == "1:02:05"


# -- Chunk files are temporary -------------------------------------------------


def test_chunk_files_are_deleted_once_the_pass_is_merged(canonical, store):
    chunk_folder = store.folder / "chunks"
    chunked = FakeService(
        Provider.OPENAI,
        spoken_words(Provider.OPENAI, SENTENCE),
        capabilities=replace(TEXT_ONLY_CAPABILITIES, maximum_file_bytes=40_000),
    )
    seen: list[bool] = []
    chunked._before_answering = lambda: seen.append(any(chunk_folder.glob("*")))

    outcome = run(canonical, store, {Provider.OPENAI: chunked})

    assert outcome.results[Provider.OPENAI].succeeded
    assert seen and all(seen), "the chunks existed while the service was being called"
    assert not chunk_folder.exists(), "nothing should be left once the pass is over"
    assert Path(canonical.path).exists(), "the canonical recording is never touched"


def test_chunk_files_are_deleted_even_when_the_pass_fails(canonical, store):
    chunk_folder = store.folder / "chunks"
    chunked = FakeService(
        Provider.OPENAI,
        capabilities=replace(TEXT_ONLY_CAPABILITIES, maximum_file_bytes=40_000),
        error="refused",
    )

    run(canonical, store, {Provider.OPENAI: chunked})

    assert not chunk_folder.exists()


def test_chunk_files_are_deleted_when_writing_them_fails_part_way(
    monkeypatch, canonical, store
):
    """A chunk that cannot be written must not strand the ones written before it.

    The pass never learns which chunks were written when the writing fails,
    so the clean-up it does once the service has answered cannot cover this.
    """
    from vox_verbatim.transcription import chunking

    chunk_folder = store.folder / "chunks"
    capabilities = replace(TEXT_ONLY_CAPABILITIES, maximum_file_bytes=15_000)
    assert len(chunking.plan_chunks(canonical, Provider.OPENAI, capabilities)) >= 4
    chunked = FakeService(Provider.OPENAI, capabilities=capabilities)
    real_cut_window = chunking.cut_window
    calls: list[Path] = []

    def failing_cut_window(canonical, span, destination, *args, **kwargs):
        calls.append(Path(destination))
        if len(calls) == 4:
            raise CanonicalAudioError("The disc filled up while writing a chunk.")
        return real_cut_window(canonical, span, destination, *args, **kwargs)

    monkeypatch.setattr(chunking, "cut_window", failing_cut_window)

    outcome = run(canonical, store, {Provider.OPENAI: chunked})

    result = outcome.results[Provider.OPENAI]
    assert not result.succeeded
    assert "could not be prepared" in result.error
    assert len(calls) == 4
    assert chunked.requests_received == []
    assert not chunk_folder.exists()
    assert Path(canonical.path).exists(), "the canonical recording is never touched"


def test_a_service_that_took_the_recording_whole_leaves_it_where_it_was(canonical, store):
    outcome = run(canonical, store, three_services())

    assert outcome.any_succeeded
    assert Path(canonical.path).exists()
