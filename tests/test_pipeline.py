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

from audio_transcriber.settings import TranscriptionSettings
from audio_transcriber.transcription import exports, pipeline
from audio_transcriber.transcription.model import (
    Language,
    Provider,
    RecordingConfiguration,
    Transcript,
)
from audio_transcriber.transcription.providers import registry
from audio_transcriber.transcription.store import TranscriptStore

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
    ) -> Run:
        services = three_services() if services is None else services
        settings = settings or offline_settings()
        calls: list[str] = []

        def build_providers(_settings, wanted=registry.DEFAULT_FULL_PASS_PROVIDERS):
            calls.append("full pass")
            return dict(services)

        def build_escalation_provider(_settings):
            calls.append("escalation")
            return None

        def build_forced_aligner(_settings):
            calls.append("forced aligner")
            return None

        monkeypatch.setattr(registry, "build_providers", build_providers)
        monkeypatch.setattr(registry, "build_escalation_provider", build_escalation_provider)
        monkeypatch.setattr(registry, "build_forced_aligner", build_forced_aligner)

        transcript = pipeline.transcribe_recording(
            recording,
            pipeline.PipelineOptions(
                configuration=configuration or RecordingConfiguration(),
                settings=settings,
            ),
            progress=progress,
            cancelled=cancelled,
        )
        store = TranscriptStore(recording, settings.processing.transcript_folder_suffix)
        return Run(transcript, services, store, calls)

    return go


# -- A whole run, with everything working --------------------------------


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


def test_services_that_answer_with_no_words_at_all_say_nothing_can_be_played(transcribe):
    """Every service answered, and every answer was empty."""
    services = {
        provider: FakeService(provider, [])
        for provider in (Provider.ELEVENLABS, Provider.OPENAI, Provider.MICROSOFT)
    }

    run = transcribe(services)

    assert run.transcript.tokens == []
    assert run.warned_about("No service returned word timings")
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
