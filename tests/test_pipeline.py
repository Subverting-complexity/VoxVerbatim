"""Tests for turning one recording into one finished transcript.

This is the module that knows the order everything goes in, so these tests
run the whole thing rather than any part of it. Nothing here reaches a
network. The services are the same fake adapters the pass tests use,
imported from there so that the two suites cannot drift into two different
ideas of what a service does, and the recording is two seconds of real
audio written by PyAV, because the pipeline opens and measures the file
before it does anything else.

Most of the file is about failure. The specification is emphatic that
losing a service costs a little accuracy and nothing else, and there is no
way to be sure of that except to break each service in turn and look at
what the user is left holding. The rule the tests are written against is
that the pipeline returns a transcript. A service that did not answer
becomes a sentence in that transcript's warnings, never an exception.
"""

from __future__ import annotations

import json
import logging

import pytest

from vox_verbatim.settings import TranscriptionSettings
from vox_verbatim.transcription import exports, pipeline
from vox_verbatim.transcription.calibration import (
    CalibrationStore,
    Observation,
    ProviderStatistics,
)
from vox_verbatim.transcription.model import (
    Candidate,
    Confidence,
    FinalToken,
    Language,
    Provider,
    ProviderResult,
    RecordingConfiguration,
    ReviewReason,
    Transcript,
)
from vox_verbatim.transcription.providers import registry
from vox_verbatim.transcription.providers.assemblyai import ASSEMBLYAI_CAPABILITIES
from vox_verbatim.transcription.store import TranscriptStore

from tests.conftest import write_real_audio
from tests.test_passes import (
    SENTENCE,
    FakeService,
    spoken_words,
    three_services,
)


# -- Setting a run up ----------------------------------------------------


def offline_settings(**processing) -> TranscriptionSettings:
    """Settings for a run that talks to nothing but the fakes.

    Escalation, adjudication and forced alignment are switched off because
    each of them would otherwise reach for a real service through the real
    registry. They have their own tests; what is being tested here is the
    shape of a run and what happens to it when a service fails.
    """
    settings = TranscriptionSettings()
    settings.processing.escalation_enabled = False
    settings.processing.adjudication_enabled = False
    settings.processing.forced_alignment_enabled = False
    for name, value in processing.items():
        setattr(settings.processing, name, value)
    return settings


class Run:
    """One finished run, and everything a test wants to ask about it."""

    def __init__(self, transcript, services, store, registry_calls) -> None:
        self.transcript: Transcript = transcript
        self.services: dict[Provider, FakeService] = services
        self.store: TranscriptStore = store
        self.registry_calls: list[str] = registry_calls

    def warned_about(self, *fragments: str) -> bool:
        """Whether one warning mentions all of these."""
        return any(
            all(fragment in warning for fragment in fragments)
            for warning in self.transcript.warnings
        )

    @property
    def saved(self) -> dict:
        return json.loads(self.store.transcript_path.read_text(encoding="utf-8"))


@pytest.fixture
def recording(tmp_path):
    path = tmp_path / "meeting.wav"
    write_real_audio(path, codec="pcm_s16le", seconds=2.0, layout="mono", rate=16000)
    return path


@pytest.fixture
def transcribe(recording, monkeypatch):
    """Run the whole pipeline over the recording with fake services.

    The registry is replaced rather than the adapters, because the pipeline
    asks the registry for its services and a test that let the real registry
    answer would be a test of whichever provider libraries happen to be
    installed on the machine running it.
    """

    def go(
        services=None,
        settings=None,
        configuration=None,
        cancelled=None,
        progress=None,
        escalation_provider=None,
        aligner=None,
        calibration_path=None,
    ) -> Run:
        services = three_services() if services is None else services
        settings = settings or offline_settings()
        calls: list[str] = []

        def build_providers(_settings, wanted=registry.DEFAULT_FULL_PASS_PROVIDERS):
            calls.append("full pass")
            return dict(services)

        def build_escalation_provider(_settings):
            calls.append("escalation")
            return escalation_provider

        def build_forced_aligner(_settings):
            calls.append("forced aligner")
            return aligner

        monkeypatch.setattr(registry, "build_providers", build_providers)
        monkeypatch.setattr(registry, "build_escalation_provider", build_escalation_provider)
        monkeypatch.setattr(registry, "build_forced_aligner", build_forced_aligner)

        transcript = pipeline.transcribe_recording(
            recording,
            pipeline.PipelineOptions(
                configuration=configuration or RecordingConfiguration(),
                settings=settings,
                calibration_path=calibration_path,
            ),
            progress=progress,
            cancelled=cancelled,
        )
        store = TranscriptStore(recording, settings.processing.transcript_folder_suffix)
        return Run(transcript, services, store, calls)

    return go


# -- A whole run, with everything working --------------------------------


def _reliability_used(transcript: Transcript, provider: Provider) -> set[float]:
    key = f"{provider.value}.provider_reliability"
    return {
        candidate.components[key]
        for token in transcript.tokens
        for candidate in token.candidates
        if key in candidate.components
    }


def test_with_no_statistics_file_a_run_produces_the_same_transcript(transcribe, tmp_path):
    plain = transcribe().transcript
    learned = transcribe(calibration_path=tmp_path / "missing" / "calibration.json").transcript

    assert learned.verbatim_text == plain.verbatim_text
    assert [token.candidates for token in learned.tokens] == [
        token.candidates for token in plain.tokens
    ]
    assert _reliability_used(learned, Provider.OPENAI) == {1.0}


def test_statistics_that_mark_a_service_down_lower_its_weight(transcribe, tmp_path):
    path = tmp_path / "calibration.json"
    statistics = ProviderStatistics()
    for index in range(200):
        statistics.record_choice(Observation(provider=Provider.OPENAI), corrected=index % 2 == 0)
    CalibrationStore(path).save(statistics)

    run = transcribe(calibration_path=path)

    used = _reliability_used(run.transcript, Provider.OPENAI)
    assert len(used) == 1
    assert used.pop() < 1.0
    # A service with no evidence keeps its default.
    assert _reliability_used(run.transcript, Provider.ELEVENLABS) == {0.95}


def test_a_damaged_statistics_file_does_not_stop_a_run(transcribe, tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text("{ this is not json", encoding="utf-8")

    run = transcribe(calibration_path=path)

    assert run.transcript.verbatim_text == SENTENCE
    assert not run.transcript.warnings
    assert _reliability_used(run.transcript, Provider.OPENAI) == {1.0}



def test_a_run_where_every_service_agrees_produces_a_finished_transcript(transcribe):
    run = transcribe()

    assert run.transcript.recording_name == "meeting.wav"
    assert run.transcript.verbatim_text == SENTENCE
    assert run.transcript.created_at and run.transcript.completed_at
    assert not run.transcript.warnings


def test_every_word_carries_a_time_a_speaker_and_the_service_it_came_from(transcribe):
    """The three questions are answered separately, and each names its source."""
    run = transcribe()

    for token in run.transcript.tokens:
        assert token.start is not None and token.end is not None
        assert token.timing_source is Provider.ELEVENLABS
        assert token.speaker == "speaker_0"
        assert token.text_source is not None
    assert [speaker.id for speaker in run.transcript.speakers] == ["speaker_0"]


def test_the_words_are_placed_in_the_recording_in_the_order_they_were_said(transcribe):
    run = transcribe()

    starts = [token.start for token in run.transcript.tokens]
    assert starts == sorted(starts)
    assert run.transcript.canonical_audio is not None
    assert run.transcript.tokens[-1].end <= run.transcript.canonical_audio.duration


def test_what_each_service_said_is_kept_beside_what_was_decided(transcribe):
    """Evidence is added to, never overwritten, so a run can be argued with."""
    run = transcribe()

    assert set(run.transcript.provider_results) == {
        Provider.ELEVENLABS,
        Provider.OPENAI,
        Provider.MICROSOFT,
    }
    for result in run.transcript.provider_results.values():
        assert [token.text for token in result.tokens] == SENTENCE.split()
    assert {request.provider for request in run.transcript.requests} == {
        Provider.ELEVENLABS,
        Provider.OPENAI,
        Provider.MICROSOFT,
    }


def test_the_transcript_and_both_exports_are_written_beside_the_recording(transcribe):
    run = transcribe()

    assert run.store.folder.name == "meeting.wav.transcript"
    assert run.store.folder.parent == run.store.recording_path.parent
    assert run.store.transcript_path.is_file()
    text = run.store.exports_folder / exports.TEXT_EXPORT_NAME
    report = run.store.exports_folder / exports.REPORT_EXPORT_NAME
    assert text.is_file() and report.is_file()
    assert SENTENCE in text.read_text(encoding="utf-8")
    assert "ElevenLabs Scribe" in report.read_text(encoding="utf-8")


def test_the_saved_transcript_can_be_read_back(transcribe):
    run = transcribe()

    reopened = run.store.load()

    assert reopened is not None
    assert reopened.verbatim_text == run.transcript.verbatim_text
    assert set(reopened.provider_results) == set(run.transcript.provider_results)


def test_the_progress_is_reported_as_sentences_rather_than_as_stage_numbers(transcribe):
    reported: list[pipeline.PipelineProgress] = []

    transcribe(progress=reported.append)

    assert reported[0].fraction == 0.0
    assert "meeting.wav" in reported[0].stage
    assert all(0.0 <= item.fraction <= 1.0 for item in reported)
    assert any("what was said" in item.stage for item in reported)


# -- One service failing -------------------------------------------------


def test_microsoft_failing_still_produces_a_transcript_and_says_it_was_lost(transcribe):
    """Microsoft is a third opinion. Losing it costs accuracy, nothing more."""
    run = transcribe(
        three_services(microsoft=FakeService(Provider.MICROSOFT, error="Azure said no."))
    )

    assert run.transcript.verbatim_text == SENTENCE
    assert run.warned_about("Microsoft MAI")
    assert not run.transcript.provider_results[Provider.MICROSOFT].succeeded
    assert run.store.transcript_path.is_file()


def test_openai_failing_still_produces_a_transcript(transcribe):
    run = transcribe(
        three_services(openai=FakeService(Provider.OPENAI, error="The key was refused."))
    )

    assert run.transcript.verbatim_text == SENTENCE
    assert run.warned_about("OpenAI")
    assert run.store.transcript_path.is_file()


def test_openai_failing_does_not_bring_in_an_older_openai_model_instead(transcribe):
    """Timings come from a service that measures them, never from a fallback.

    The temptation the specification rules out is reaching for an older
    OpenAI model that returns timestamps when the configured one fails. So
    the services are asked for exactly once, OpenAI is asked exactly once,
    and no word in the finished transcript takes its spelling from the
    service that did not answer.
    """
    failed = FakeService(Provider.OPENAI, error="The key was refused.")

    run = transcribe(three_services(openai=failed))

    assert run.registry_calls == ["full pass"]
    assert len(failed.requests_received) == 1
    assert not any(
        token.text_source is Provider.OPENAI for token in run.transcript.tokens
    )
    # One request, against the model the user configured, recorded as having
    # failed. A substitution would show up here as a second model name.
    openai_requests = [
        request for request in run.transcript.requests if request.provider is Provider.OPENAI
    ]
    assert [request.model_identifier for request in openai_requests] == ["fake-model-1"]
    assert not openai_requests[0].succeeded


def test_elevenlabs_failing_moves_the_backbone_and_says_so_in_plain_words(transcribe):
    """Losing the backbone changes every timestamp, so it is not a detail.

    OpenAI and Microsoft do not time their words, so the fallback backbone
    here is given timings to make it a realistic stand-in for whichever
    timed service is left when Scribe is missing.
    """
    timed_openai = FakeService(
        Provider.OPENAI,
        spoken_words(Provider.OPENAI, SENTENCE, timed=True, speaker="speaker_1"),
        model="openai-fake",
    )
    services = three_services(
        elevenlabs=FakeService(Provider.ELEVENLABS, error="Scribe did not answer."),
        openai=timed_openai,
    )

    run = transcribe(services)

    assert run.transcript.verbatim_text == SENTENCE
    assert run.warned_about("ElevenLabs Scribe", "timings and speakers", "OpenAI")
    for token in run.transcript.tokens:
        assert token.timing_source is Provider.OPENAI
    assert run.store.transcript_path.is_file()


def test_losing_every_timed_service_leaves_the_words_without_playable_times(transcribe):
    """Nothing left measured a word, so the text stands and the clock does not.

    The words still come out, and they still come out in order, but not one
    of them may be used to move the player, because no service put it
    anywhere. Inventing boundaries here would be the one thing the design
    refuses to do.
    """
    run = transcribe(
        three_services(
            elevenlabs=FakeService(Provider.ELEVENLABS, error="Scribe did not answer.")
        )
    )

    assert run.transcript.verbatim_text == SENTENCE
    assert run.warned_about("ElevenLabs Scribe", "timings and speakers")
    assert not any(token.timing_status.is_exact for token in run.transcript.tokens)
    assert run.store.transcript_path.is_file()


def test_services_that_answer_with_no_words_at_all_say_so(transcribe):
    """Every service answered, and every answer was empty.

    The warning names what happened, no words at all, rather than the old
    sentence about timings, which described a different failure.
    """
    services = {
        provider: FakeService(provider, [])
        for provider in (Provider.ELEVENLABS, Provider.OPENAI, Provider.MICROSOFT)
    }

    run = transcribe(services)

    assert run.transcript.tokens == []
    assert run.warned_about("No service returned any words")
    assert run.store.transcript_path.is_file()


# -- Everything failing --------------------------------------------------


def test_every_service_failing_returns_a_transcript_carrying_a_warning(transcribe):
    """The contract is that a transcript comes back, however badly it went."""
    services = {
        provider: FakeService(provider, error="Nothing came back.")
        for provider in (Provider.ELEVENLABS, Provider.OPENAI, Provider.MICROSOFT)
    }

    run = transcribe(services)

    assert isinstance(run.transcript, Transcript)
    assert run.transcript.tokens == []
    assert run.warned_about("No transcription service produced a result")
    assert run.warned_about("API keys in Settings")
    assert run.store.transcript_path.is_file()


def test_no_service_switched_on_at_all_is_reported_rather_than_raised(transcribe):
    run = transcribe({})

    assert isinstance(run.transcript, Transcript)
    assert run.warned_about("No transcription service is switched on")


# -- An export failing ---------------------------------------------------


def test_an_export_that_cannot_be_written_does_not_cost_the_transcript(
    transcribe, monkeypatch
):
    """An export is a convenience. The transcript is the work."""

    def refuse(_transcript, *args, **kwargs):
        raise OSError("The exports folder is read only.")

    monkeypatch.setattr(exports, "render_plain_text", refuse)

    run = transcribe()

    assert run.store.transcript_path.is_file()
    assert run.saved["tokens"]
    assert run.warned_about(exports.TEXT_EXPORT_NAME, "could not be written")
    # The other export is written even though the first one failed.
    assert (run.store.exports_folder / exports.REPORT_EXPORT_NAME).is_file()


def test_both_exports_failing_still_leaves_the_transcript_on_disk(
    transcribe, monkeypatch
):
    def refuse(_transcript, *args, **kwargs):
        raise OSError("The exports folder is read only.")

    monkeypatch.setattr(exports, "render_plain_text", refuse)
    monkeypatch.setattr(exports, "render_review_report", refuse)

    run = transcribe()

    assert run.store.transcript_path.is_file()
    assert run.warned_about(exports.REPORT_EXPORT_NAME, "could not be written")


# -- Stopping part way ---------------------------------------------------


def test_a_run_stopped_part_way_returns_what_it_had_rather_than_raising(transcribe):
    """Cancelling is an ordinary end to a run, not an error."""
    stopping = {"asked": False}

    def cancelled() -> bool:
        return stopping["asked"]

    def stop_once_asked() -> None:
        stopping["asked"] = True

    services = three_services()
    for service in services.values():
        service._before_answering = stop_once_asked

    run = transcribe(services, cancelled=cancelled)

    assert isinstance(run.transcript, Transcript)
    assert run.warned_about("stopped before it finished")
    assert set(run.transcript.provider_results) == set(services)
    assert run.store.transcript_path.is_file()


def test_a_run_cancelled_before_anything_was_sent_still_saves_a_transcript(transcribe):
    run = transcribe(cancelled=lambda: True)

    assert isinstance(run.transcript, Transcript)
    assert run.transcript.canonical_audio is not None
    assert run.warned_about("stopped before it finished")
    assert run.store.transcript_path.is_file()


# -- Keeping an earlier transcript ---------------------------------------


def every_service_failing() -> dict:
    return {
        provider: FakeService(provider, error="Nothing came back.")
        for provider in (Provider.ELEVENLABS, Provider.OPENAI, Provider.MICROSOFT)
    }


def files_on_disk(store: TranscriptStore) -> dict[str, bytes]:
    """The transcript and both exports, byte for byte."""
    paths = [
        store.transcript_path,
        store.exports_folder / exports.TEXT_EXPORT_NAME,
        store.exports_folder / exports.REPORT_EXPORT_NAME,
    ]
    return {path.name: path.read_bytes() for path in paths}


def test_a_rerun_where_no_service_answers_keeps_the_earlier_transcript(transcribe):
    """The earlier file holds the user's review corrections; nothing replaces it."""
    first = transcribe()
    assert first.transcript.tokens
    before = files_on_disk(first.store)

    run = transcribe(every_service_failing())

    assert run.transcript.tokens == []
    assert files_on_disk(run.store) == before
    assert run.warned_about("already had a transcript", "kept unchanged")


def test_a_rerun_cancelled_before_anything_was_sent_keeps_the_earlier_transcript(
    transcribe,
):
    first = transcribe()
    before = files_on_disk(first.store)

    run = transcribe(cancelled=lambda: True)

    assert files_on_disk(run.store) == before
    assert run.warned_about("stopped before it finished")
    assert run.warned_about("already had a transcript", "kept unchanged")


def test_an_earlier_transcript_that_cannot_be_read_is_kept_rather_than_replaced(
    transcribe,
):
    """A damaged or locked file may still hold work, so it is not overwritten."""
    first = transcribe()
    first.store.transcript_path.write_text("{ not json", encoding="utf-8")
    before = files_on_disk(first.store)

    run = transcribe(every_service_failing())

    assert files_on_disk(run.store) == before
    assert run.warned_about("already had a transcript", "kept unchanged")


def test_an_earlier_transcript_with_no_words_is_replaced_as_before(transcribe):
    first = transcribe(every_service_failing())
    assert first.saved["tokens"] == []

    run = transcribe(every_service_failing())

    assert not run.warned_about("already had a transcript")
    assert run.saved["completed_at"] == run.transcript.completed_at


def test_a_rerun_that_produces_words_replaces_the_earlier_transcript(transcribe):
    first = transcribe(every_service_failing())
    assert first.saved["tokens"] == []

    run = transcribe()

    assert run.saved["tokens"]
    assert not run.warned_about("already had a transcript")


# -- Keeping the evidence ------------------------------------------------


def test_the_untouched_answers_reach_the_folder_and_the_record_names_each_one(
    transcribe,
):
    """A later run that answers differently can only be explained against these."""
    run = transcribe()

    written = run.store.raw_response_names()
    assert len(written) == 3
    named = {
        request.provider: request.raw_response_file for request in run.transcript.requests
    }
    for provider in run.transcript.provider_results:
        assert named[provider] in written
        assert provider.value in named[provider]
    assert set(run.saved["provider_results"]) == {"elevenlabs", "openai", "microsoft"}


def test_the_saved_transcript_carries_the_file_names_rather_than_the_answers(transcribe):
    """The answers live in their own files; the transcript only points at them.

    A long recording's response is large, and holding a second copy of it
    inside the file the review window opens every time would be wasteful.
    """
    run = transcribe()

    saved_requests = run.saved["requests"]
    assert len(saved_requests) == 3
    for request in saved_requests:
        assert request["raw_response_file"]
        assert (run.store.raw_responses_folder / request["raw_response_file"]).is_file()


# -- Keeping the keys off the disk ---------------------------------------


#: Deliberately shaped like nothing in particular. A value beginning ``sk-``
#: or looking like a long hexadecimal string is caught by the transcript
#: store's own second check on the way out, so a canary of that shape would
#: pass this test even if the settings layer stopped redacting altogether.
#: This one is caught only by the rule that is meant to catch it.
CANARY = "canary-value-that-must-never-be-written-down"


def test_no_api_key_reaches_the_transcript_the_exports_or_the_log(
    transcribe, caplog, recording
):
    """A transcript is a document people send each other."""
    settings = offline_settings()
    settings.elevenlabs.api_key = f"{CANARY}-elevenlabs"
    settings.openai_transcription.api_key = f"{CANARY}-openai"
    settings.openai_adjudication.api_key = f"{CANARY}-adjudication"
    settings.microsoft.api_key = f"{CANARY}-microsoft"
    settings.assemblyai.api_key = f"{CANARY}-assemblyai"
    settings.deepgram.api_key = f"{CANARY}-deepgram"
    # A key smuggled in under a free-form parameter is still a key.
    settings.openai_transcription.parameters = {"api_key_override": f"{CANARY}-parameter"}

    with caplog.at_level(logging.DEBUG):
        run = transcribe(settings=settings)

    assert run.transcript.effective_settings
    assert CANARY not in json.dumps(run.transcript.effective_settings)
    for path in sorted(run.store.folder.rglob("*")):
        if path.is_file():
            assert CANARY.encode() not in path.read_bytes(), path
    assert CANARY not in caplog.text
    assert CANARY not in recording.parent.joinpath("meeting.wav").name


def test_the_settings_that_are_not_secret_are_kept_so_a_run_can_be_explained(
    transcribe,
):
    """Redaction has to take the keys out without taking the record with them."""
    settings = offline_settings()
    settings.elevenlabs.api_key = CANARY
    settings.elevenlabs.transcription_model = "scribe-v1-test"

    run = transcribe(settings=settings)

    recorded = json.dumps(run.saved["effective_settings"])
    assert "scribe-v1-test" in recorded
    assert CANARY not in recorded


# -- The Afrikaans switch ------------------------------------------------


def test_afrikaans_disabled_means_no_service_is_ever_asked_about_afrikaans(transcribe):
    """Off removes the Afrikaans path rather than running it and discarding it.

    Asserted on what the adapters were handed, because that is the only
    place the promise can actually be broken.
    """
    services = three_services()

    transcribe(services, configuration=RecordingConfiguration(afrikaans_enabled=False))

    for service in services.values():
        request = service.requests_received[0]
        assert request.languages == (Language.ENGLISH, Language.GERMAN)
        assert Language.AFRIKAANS not in request.languages
        written = json.dumps(request.extra_parameters).lower()
        assert "afrikaans" not in written
        assert '"af"' not in written
        assert "afrikaans" not in request.context_prompt.lower()


def test_afrikaans_enabled_reaches_the_services_that_were_asked(transcribe):
    services = three_services()

    transcribe(services, configuration=RecordingConfiguration(afrikaans_enabled=True))

    for service in services.values():
        assert Language.AFRIKAANS in service.requests_received[0].languages
    openai = services[Provider.OPENAI].requests_received[0]
    assert "af" in openai.extra_parameters["languages"]


def test_the_recording_remembers_whether_afrikaans_was_allowed(transcribe):
    """It is a decision about one recording, so it belongs in that transcript."""
    run = transcribe(configuration=RecordingConfiguration(afrikaans_enabled=True))

    assert run.transcript.configuration.afrikaans_enabled is True
    assert run.saved["configuration"]["afrikaans_enabled"] is True


# -- The second opinion, wired all the way through ----------------------


class SecondOpinion:
    """An escalation service that answers every window with one reading.

    It remembers the requests so a test can see what it was told, and it
    answers with the word it was built with at the time of the window it
    was asked about, which is how a real time-window answer arrives.
    """

    provider = Provider.ASSEMBLYAI
    capabilities = ASSEMBLYAI_CAPABILITIES

    def __init__(
        self,
        hears: str,
        problem: str | None = None,
        at: tuple[float, float] = (0.6, 0.76),
    ) -> None:
        self.hears = hears
        self.problem = problem
        # Where in the recording the word is heard. The fake words are 0.2 s
        # apart, so the default is the fourth word of PLAIN_SENTENCE, with
        # boundaries a little off the backbone's, as a real service's are.
        self.at = at
        self.windows: list = []
        self.requests: list = []

    def describe_configuration_problem(self):
        return self.problem

    def upload(self, audio_path):
        return "https://uploads.example/recording"

    def transcribe_window(self, request, *, audio_url=None, window=None, cancelled=None):
        from vox_verbatim.transcription.model import ProviderToken

        self.windows.append(window)
        self.requests.append(request)
        return ProviderResult(
            provider=self.provider,
            tokens=[
                ProviderToken(
                    provider=self.provider,
                    index=0,
                    text=self.hears,
                    start=self.at[0] - 0.02,
                    end=self.at[1] + 0.02,
                    confidence=0.95,
                )
            ],
        )


#: A sentence with no amount, date or number in it. Those are values the
#: application refuses to settle by any opinion, so a test of settling has
#: to stay away from them, and from the words next to them, which share
#: their risk.
PLAIN_SENTENCE = "we spoke with Peter about the garden this morning"


def disagreeing_services(
    elevenlabs: str = "Peter",
    openai: str = "Pieter",
    microsoft: str = "Petrus",
    target: str = "Peter",
) -> dict[Provider, FakeService]:
    """The three services, each hearing one word its own way."""
    from tests.test_passes import BACKBONE_CAPABILITIES

    def saying(provider, name, **extra):
        return FakeService(
            provider,
            spoken_words(provider, PLAIN_SENTENCE.replace(target, name), **extra),
            model=f"{provider.value}-fake",
            **({"capabilities": BACKBONE_CAPABILITIES, "speakers": ("speaker_0",)}
               if provider is Provider.ELEVENLABS else {}),
        )

    return {
        Provider.ELEVENLABS: saying(
            Provider.ELEVENLABS, elevenlabs, timed=True, speaker="speaker_0"
        ),
        Provider.OPENAI: saying(Provider.OPENAI, openai),
        Provider.MICROSOFT: saying(Provider.MICROSOFT, microsoft),
    }


def test_a_second_opinion_that_sides_with_one_service_corrects_the_word(transcribe):
    """The whole point of escalation, end to end: the answer reaches the word."""
    second = SecondOpinion(hears="Pieter")
    settings = offline_settings(escalation_enabled=True)

    run = transcribe(
        disagreeing_services(), settings=settings, escalation_provider=second
    )

    assert second.windows, "the disputed word was never sent for a second opinion"
    name = run.transcript.tokens[3]
    assert name.text == "Pieter"
    assert name.text_source is Provider.ASSEMBLYAI
    assert name.needs_review is False
    # What the second service said is kept beside the other services, so
    # the review window and the report can show it.
    assert Provider.ASSEMBLYAI in run.transcript.provider_results
    assert run.transcript.provider_results[Provider.ASSEMBLYAI].tokens[0].text == "Pieter"


def test_the_escalation_settings_and_the_recording_reach_the_second_service(transcribe):
    """The speaker count, the context and the vocabulary are not decoration."""
    second = SecondOpinion(hears="Pieter")
    settings = offline_settings(escalation_enabled=True)
    configuration = RecordingConfiguration(
        expected_speaker_count=3, recording_context="A board meeting."
    )

    transcribe(
        disagreeing_services(),
        settings=settings,
        configuration=configuration,
        escalation_provider=second,
    )

    assert second.requests
    assert second.requests[0].expected_speaker_count == 3
    assert "board meeting" in second.requests[0].context_prompt


def test_a_second_opinion_nobody_else_offered_is_evidence_not_an_answer(transcribe):
    second = SecondOpinion(hears="Petra")
    settings = offline_settings(escalation_enabled=True)

    run = transcribe(
        disagreeing_services(), settings=settings, escalation_provider=second
    )

    name = run.transcript.tokens[3]
    assert name.text != "Petra"
    assert name.needs_review is True
    assert "Petra" in [candidate.text for candidate in name.candidates]


def test_an_escalation_service_with_a_problem_is_reported_before_anything_is_sent(
    transcribe,
):
    second = SecondOpinion(hears="Pieter", problem="No AssemblyAI API key is set.")
    settings = offline_settings(escalation_enabled=True)

    run = transcribe(
        disagreeing_services(), settings=settings, escalation_provider=second
    )

    assert second.windows == []
    assert run.warned_about("no escalation service is set up", "No AssemblyAI API key")
    assert run.transcript.tokens[3].needs_review is True


def test_escalation_progress_is_reported_window_by_window(transcribe):
    second = SecondOpinion(hears="Pieter")
    settings = offline_settings(escalation_enabled=True)
    sentences: list[str] = []

    transcribe(
        disagreeing_services(),
        settings=settings,
        escalation_provider=second,
        progress=lambda report: sentences.append(report.stage),
    )

    assert any(sentence.startswith("Second opinion 1 of") for sentence in sentences)


# -- A later stage failing never costs the words ------------------------


def test_a_stage_that_breaks_after_the_services_answered_keeps_the_transcript(
    transcribe, monkeypatch
):
    """Everything after reconciliation is a refinement of words already paid for."""
    from vox_verbatim.transcription import escalation

    def broken(*args, **kwargs):
        raise RuntimeError("a bug in planning")

    monkeypatch.setattr(escalation, "plan_escalation", broken)
    settings = offline_settings(escalation_enabled=True)

    run = transcribe(disagreeing_services(), settings=settings)

    assert run.transcript.verbatim_text.startswith("we spoke with")
    assert run.warned_about("second opinions stage failed", "a bug in planning")
    assert run.store.transcript_path.is_file()


# -- What the language model is asked about, and in what size -----------


class BatchCountingAdjudicator:
    """Answers nothing, remembers how it was asked."""

    def __init__(self, *args, **kwargs) -> None:
        self.batches: list[int] = []

    def is_configured(self) -> bool:
        return True

    def adjudicate(self, disputes, **kwargs):
        from vox_verbatim.transcription.adjudication import AdjudicationOutcome

        self.batches.append(len(disputes))
        for dispute in disputes:
            for token in dispute.tokens:
                token.flag(ReviewReason.ADJUDICATION_DECLINED)
        return AdjudicationOutcome(unanswered=tuple(d.span_id for d in disputes))


def test_only_words_whose_text_is_in_doubt_are_put_to_the_language_model():
    """A word in the queue for its speaker has nothing for the model to decide."""
    disagreement = FinalToken(
        text="fifteen",
        text_confidence=Confidence.REVIEW_REQUIRED,
        candidates=[Candidate("fifteen"), Candidate("fifty")],
    )
    disagreement.flag(ReviewReason.PROVIDER_DISAGREEMENT)
    speaker_only = FinalToken(
        text="yes", text_confidence=Confidence.HIGH, candidates=[Candidate("yes")]
    )
    speaker_only.flag(ReviewReason.SPEAKER_UNCERTAIN)
    one_reading = FinalToken(
        text="um", text_confidence=Confidence.REVIEW_SUGGESTED, candidates=[Candidate("um")]
    )
    one_reading.flag(ReviewReason.LOW_ACOUSTIC_CONFIDENCE)

    disputes = pipeline._build_disputes(
        [disagreement, speaker_only, one_reading], Transcript(recording_name="a")
    )

    assert [dispute.tokens for dispute in disputes] == [(disagreement,)]


def test_disputes_go_to_the_language_model_in_batches(monkeypatch):
    from vox_verbatim.transcription import adjudication

    built: list[BatchCountingAdjudicator] = []

    def build(*args, **kwargs):
        adjudicator = BatchCountingAdjudicator()
        built.append(adjudicator)
        return adjudicator

    monkeypatch.setattr(adjudication, "Adjudicator", build)
    tokens = []
    for number in range(75):
        token = FinalToken(
            text=f"w{number}",
            start=float(number),
            end=number + 0.5,
            text_confidence=Confidence.REVIEW_REQUIRED,
            candidates=[Candidate(f"w{number}"), Candidate(f"v{number}")],
        )
        token.flag(ReviewReason.PROVIDER_DISAGREEMENT)
        tokens.append(token)
    # Settled words between them keep each dispute separate.
    spaced = []
    for token in tokens:
        spaced.append(token)
        spaced.append(FinalToken(text="and", text_confidence=Confidence.HIGH))
    transcript = Transcript(recording_name="a", tokens=spaced)
    settings = TranscriptionSettings()
    settings.openai_adjudication.api_key = "key"

    class Context:
        recording_context = ""
        term_texts = ()
        languages = ()

    pipeline._adjudicate(spaced, transcript, settings, Context(), pipeline._Reporter(None), None)

    assert built[0].batches == [30, 30, 15]


class _NoContext:
    recording_context = ""
    term_texts = ()
    languages = ()


def _adjudication_switched_off(monkeypatch) -> tuple[list, TranscriptionSettings]:
    """Settings with the service off but everything else ready to send."""
    from vox_verbatim.transcription import adjudication

    built: list[BatchCountingAdjudicator] = []

    def build(*args, **kwargs):
        adjudicator = BatchCountingAdjudicator()
        built.append(adjudicator)
        return adjudicator

    monkeypatch.setattr(adjudication, "Adjudicator", build)
    settings = TranscriptionSettings()
    settings.processing.adjudication_enabled = True
    settings.openai_adjudication.api_key = "key"
    settings.openai_adjudication.enabled = False
    return built, settings


def test_switching_openai_adjudication_off_sends_no_request(monkeypatch):
    """The Settings switch must stop the requests, not just look as if it does."""
    built, settings = _adjudication_switched_off(monkeypatch)
    disputed = FinalToken(
        text="w", start=0.0, end=0.5, text_confidence=Confidence.REVIEW_REQUIRED,
        candidates=[Candidate("w"), Candidate("v")],
    )
    disputed.flag(ReviewReason.PROVIDER_DISAGREEMENT)
    tokens = [disputed, FinalToken(text="and", text_confidence=Confidence.HIGH)]
    transcript = Transcript(recording_name="a", tokens=tokens)

    result = pipeline._adjudicate(
        tokens, transcript, settings, _NoContext(), pipeline._Reporter(None), None
    )

    assert built == [], "no adjudicator may be built while the service is off"
    assert result is tokens
    assert transcript.requests == []
    assert any("Adjudication was not used" in warning for warning in transcript.warnings)


def test_a_recording_with_nothing_to_adjudicate_gets_no_warning(monkeypatch):
    built, settings = _adjudication_switched_off(monkeypatch)
    tokens = [FinalToken(text="and", text_confidence=Confidence.HIGH)]
    transcript = Transcript(recording_name="a", tokens=tokens)

    pipeline._adjudicate(tokens, transcript, settings, _NoContext(), pipeline._Reporter(None), None)

    assert built == []
    assert transcript.warnings == []


# -- Timing is measured phrase by phrase, against a clip -----------------


def test_moved_words_are_gathered_into_phrases_by_adjacency_and_time():
    from vox_verbatim.transcription.model import TimingStatus

    def moved(text, start, end):
        return FinalToken(
            text=text, start=start, end=end, timing_status=TimingStatus.MAPPED_SUBSTITUTION
        )

    def fine(text, start, end):
        return FinalToken(
            text=text, start=start, end=end, timing_status=TimingStatus.EXACT_PROVIDER_TIME
        )

    tokens = [
        moved("a", 0.0, 0.3),
        moved("b", 0.35, 0.6),
        fine("c", 0.65, 0.9),
        moved("d", 1.0, 1.2),
        moved("e", 5.0, 5.3),
    ]

    phrases = pipeline._phrases_to_realign(tokens)

    assert [[token.text for token in phrase] for phrase in phrases] == [["a", "b"], ["d"], ["e"]]


class ClipRecordingAligner:
    """A forced aligner that records what audio it was handed."""

    provider = Provider.ELEVENLABS

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float]] = []

    def is_configured(self) -> bool:
        return True

    def supports(self, language) -> bool:
        return True

    def align(self, audio_path, text, language, canonical_offset):
        from vox_verbatim.transcription.canonical import probe_audio

        duration = probe_audio(audio_path).duration
        self.calls.append((text, str(audio_path), duration))
        words = text.split()
        step = duration / max(1, len(words))
        return [
            {
                "text": word,
                "start": canonical_offset + index * step,
                "end": canonical_offset + (index + 1) * step,
                "loss": 0.1,
            }
            for index, word in enumerate(words)
        ]


def test_forced_alignment_is_given_a_clip_of_the_phrase_not_the_whole_recording(transcribe):
    """An aligner must place every word of the text it is given somewhere in
    the audio it is given. Hand it the whole recording and a few scattered
    words, and it places them at moments that have nothing to do with where
    they were said."""
    aligner = ClipRecordingAligner()
    settings = offline_settings(forced_alignment_enabled=True, escalation_enabled=True)

    # The backbone hears an ordinary word differently from the other two,
    # and the second opinion sides with them, so the word is a substitution
    # sitting in a span measured for another word: exactly what is worth
    # measuring again.
    run = transcribe(
        disagreeing_services("carton", "garden", "garden", target="garden"),
        settings=settings,
        escalation_provider=SecondOpinion(hears="garden", at=(1.2, 1.36)),
        aligner=aligner,
    )

    assert run.transcript.tokens[6].text == "garden"
    assert aligner.calls, "nothing was realigned"
    assert aligner.calls[0][0] == "garden"
    assert run.transcript.tokens[6].timing_status.value == "forced_aligned"
    for text, path, duration in aligner.calls:
        assert duration < 2.0, "the aligner was handed the whole recording"
        assert "alignment" in path
    # The clips are working files and do not stay behind.
    assert not (run.store.folder / "alignment").exists()


def test_a_phrase_of_inserted_words_alone_is_not_sent_for_alignment():
    """An inserted word has no span of its own, only the gap between its
    neighbours. A phrase made of nothing else would have an aligner place
    words nobody measured onto the neighbours' audio."""
    from vox_verbatim.transcription.model import AudioSpan as Span, TimingStatus

    inserted = FinalToken(
        text="um", timing_status=TimingStatus.UNALIGNED, source_audio_span=Span(1.0, 1.0)
    )
    timed = FinalToken(
        text="go", start=1.0, end=1.3, timing_status=TimingStatus.MAPPED_SUBSTITUTION
    )

    assert pipeline._phrases_to_realign([inserted]) == []
    # Inside a phrase that has a timed word, the inserted word goes along.
    phrases = pipeline._phrases_to_realign([timed, inserted])
    assert [[token.text for token in phrase] for phrase in phrases] == [["go", "um"]]
    assert pipeline._phrase_audio_span(phrases[0]) == Span(1.0, 1.3)


def test_a_word_with_only_a_start_still_places_its_phrase():
    from vox_verbatim.transcription.model import AudioSpan as Span, TimingStatus

    start_only = FinalToken(text="x", start=1.0, timing_status=TimingStatus.MAPPED_SUBSTITUTION)
    whole = FinalToken(text="y", start=1.4, end=1.8, timing_status=TimingStatus.MAPPED_SUBSTITUTION)

    phrases = pipeline._phrases_to_realign([start_only, whole])

    assert pipeline._phrase_audio_span(phrases[0]) == Span(1.0, 1.8)
