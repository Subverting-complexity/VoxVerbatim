"""Tests for the OpenAI adapter.

Nothing here touches the network and nothing needs an API key. A fake client
stands in for the library, which is why the adapter takes one through its
constructor.

Two of these tests are load-bearing in a way the others are not. The first
is that every token comes back without a time, because that is the thing
that looks like a bug and is not. The second is that no older model can be
reached through this adapter, however it is configured: the older models
would hand back the very word timings this service lacks, and taking that
trade would quietly make the transcript worse at the one thing OpenAI is
here to do well.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from audio_transcriber.transcription.model import Language, Provider
from audio_transcriber.transcription.providers import openai as openai_module
from audio_transcriber.transcription.providers.base import (
    ProviderNotConfigured,
    ProviderUnavailable,
    TranscriptionRequest,
)
from audio_transcriber.transcription.providers.openai import (
    MODEL_FAMILY,
    OpenAiTranscriptionProvider,
    is_permitted_model,
)

#: Obviously not real, but distinctive, so a test can prove it appears
#: nowhere in what gets written to provenance.
API_KEY = "sk-secret-key-do-not-record"

#: The models this adapter must never reach, whatever it is told to do. Each
#: of them exists and each of them would answer; that is the problem.
FORBIDDEN_MODELS = (
    "whisper-1",
    "whisper",
    "whisper-large-v3",
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
    "gpt-4o-transcribe-diarize",
    "gpt-4o-audio-preview",
)


# -- The fake client -----------------------------------------------------


class FakeRawResponse:
    """What ``with_raw_response`` hands back.

    It carries the headers, the undecoded body as ``text``, and the parsed
    answer on request. The adapter reads all three, so the double has to
    offer all three.
    """

    def __init__(self, response, headers: dict[str, str], text: str) -> None:
        self._response = response
        self.headers = headers
        self.text = text

    def parse(self):
        return self._response


class FakeWithRawResponse:
    def __init__(self, owner: "FakeTranscriptions") -> None:
        self._owner = owner

    def create(self, **arguments):
        self._owner.calls.append(arguments)
        if self._owner.error is not None:
            raise self._owner.error
        return FakeRawResponse(self._owner.response, self._owner.headers, self._owner.body)


class FakeTranscriptions:
    def __init__(self, response=None, error=None, headers=None, body=None) -> None:
        self.response = response
        self.error = error
        self.headers = headers or {"x-request-id": "req_abc123"}
        if body is None and response is not None:
            body = json.dumps({"text": response.text, "languages": response.languages})
        self.body = body or ""
        self.calls: list[dict] = []
        self.with_raw_response = FakeWithRawResponse(self)


class FakeClient:
    def __init__(self, response=None, error=None, headers=None, body=None) -> None:
        self.transcriptions = FakeTranscriptions(response, error, headers, body)
        self.audio = SimpleNamespace(transcriptions=self.transcriptions)


def transcription(text: str, languages=("en",)):
    return SimpleNamespace(text=text, languages=list(languages), logprobs=None, usage=None)


SAMPLE_TEXT = "Good morning, everyone. Suzanne will present the Q4 numbers."


@pytest.fixture
def audio(tmp_path) -> Path:
    path = tmp_path / "meeting.wav"
    path.write_bytes(b"RIFF----WAVEfmt ")
    return path


@pytest.fixture
def request_for(audio):
    def build(**overrides) -> TranscriptionRequest:
        arguments = {"audio_path": audio, "canonical_offset": 612.5, "duration": 12.0}
        arguments.update(overrides)
        return TranscriptionRequest(**arguments)

    return build


def build_provider(client=None, **overrides) -> OpenAiTranscriptionProvider:
    arguments = {"api_key": API_KEY, "client": client}
    arguments.update(overrides)
    return OpenAiTranscriptionProvider(**arguments)


# -- Words without times -------------------------------------------------


def test_every_word_comes_back_without_a_time_and_that_is_correct(request_for):
    """The model does not time words. Alignment will place them later."""
    client = FakeClient(transcription(SAMPLE_TEXT))
    result = build_provider(client).transcribe(request_for())

    assert result.succeeded
    assert result.tokens
    assert all(token.start is None and token.end is None for token in result.tokens)
    assert all(not token.has_timing for token in result.tokens)
    assert build_provider().capabilities.word_timings is False


def test_the_text_is_split_into_words_exactly_as_the_service_wrote_them(request_for):
    """Punctuation stays attached, because that is how it came back."""
    client = FakeClient(transcription(SAMPLE_TEXT))
    result = build_provider(client).transcribe(request_for())

    assert [token.text for token in result.tokens] == [
        "Good",
        "morning,",
        "everyone.",
        "Suzanne",
        "will",
        "present",
        "the",
        "Q4",
        "numbers.",
    ]
    assert [token.index for token in result.tokens] == list(range(9))


def test_the_words_remember_which_chunk_they_came_from(request_for):
    """Without times, the chunk number is the only place a word can be traced to."""
    client = FakeClient(transcription(SAMPLE_TEXT))
    result = build_provider(client).transcribe(request_for(chunk_index=2))

    assert {token.chunk_index for token in result.tokens} == {2}


def test_a_word_of_bare_punctuation_is_marked_to_be_skipped(request_for):
    client = FakeClient(transcription("Hello -- world"))
    result = build_provider(client).transcribe(request_for())

    assert [token.is_punctuation for token in result.tokens] == [False, True, False]
    assert [token.text for token in result.word_tokens] == ["Hello", "world"]
    # This service returns text and nothing else, so it never reports a
    # sound and no token of its may claim to be one.
    assert all(not token.is_audio_event for token in result.tokens)


def test_the_declared_capabilities_match_what_the_model_actually_does():
    capabilities = build_provider().capabilities

    assert capabilities.word_timings is False
    assert capabilities.diarisation is False
    assert capabilities.word_confidence is False
    assert capabilities.speaker_count_hint is False
    assert capabilities.language_detection is True
    assert capabilities.vocabulary_biasing is True
    assert capabilities.context_prompt is True
    assert capabilities.maximum_file_bytes == 25 * 1024 * 1024


# -- Only one model family -----------------------------------------------


@pytest.mark.parametrize("model", FORBIDDEN_MODELS)
def test_no_older_model_can_be_reached_through_this_adapter(model, request_for):
    """Timestamps from an older model are exactly the trade we refuse to make."""
    client = FakeClient(transcription(SAMPLE_TEXT))
    provider = build_provider(client, model=model)

    assert is_permitted_model(model) is False
    assert provider.is_configured() is False
    assert MODEL_FAMILY in provider.describe_configuration_problem()

    with pytest.raises(ProviderNotConfigured):
        provider._transcribe(request_for())

    result = provider.transcribe(request_for())
    assert not result.succeeded
    assert model in result.error
    # The decisive assertion: the service was never called at all.
    assert client.transcriptions.calls == []


@pytest.mark.parametrize(
    "model", ["gpt-transcribe", "gpt-transcribe-2026-08-01", "GPT-Transcribe"]
)
def test_the_permitted_family_includes_later_members_of_itself(model, request_for):
    """The name stays editable, so a future model works without a code change."""
    client = FakeClient(transcription(SAMPLE_TEXT))
    provider = build_provider(client, model=model)

    assert is_permitted_model(model) is True
    assert provider.is_configured() is True
    assert provider.transcribe(request_for()).succeeded
    assert client.transcriptions.calls[0]["model"] == model


def test_the_default_model_is_the_permitted_one():
    provider = OpenAiTranscriptionProvider(api_key=API_KEY)

    assert provider.model_identifier == MODEL_FAMILY
    assert provider.is_configured()


# -- What gets sent ------------------------------------------------------


def test_context_terms_and_languages_are_sent_as_themselves(request_for):
    """All three are typed parameters now, so none of them goes through extra_body."""
    client = FakeClient(transcription(SAMPLE_TEXT))
    build_provider(client).transcribe(
        request_for(
            context_prompt="A board meeting about the year-end numbers.",
            vocabulary_terms=("Suzanne", "Bosch"),
            languages=(Language.ENGLISH, Language.AFRIKAANS),
        )
    )

    call = client.transcriptions.calls[0]
    assert call["prompt"] == "A board meeting about the year-end numbers."
    assert call["keywords"] == ["Suzanne", "Bosch"]
    assert call["languages"] == ["en", "af"]
    assert call["extra_body"] == {}


def test_free_form_parameters_go_through_the_escape_hatch(request_for):
    """Settings extras and request extras both reach extra_body, request first."""
    client = FakeClient(transcription(SAMPLE_TEXT))
    build_provider(client, parameters={"seed": 1, "some_setting": "a"}).transcribe(
        request_for(extra_parameters={"seed": 42})
    )

    assert client.transcriptions.calls[0]["extra_body"] == {"seed": 42, "some_setting": "a"}


def test_nothing_is_sent_that_the_model_cannot_use(request_for):
    """It has no speakers and no timestamps, so it is never asked for either."""
    client = FakeClient(transcription(SAMPLE_TEXT))
    build_provider(client).transcribe(request_for(expected_speaker_count=4, diarise=True))

    call = client.transcriptions.calls[0]
    assert "timestamp_granularities" not in call
    assert "num_speakers" not in call
    assert "include" not in call
    assert "diarize" not in call


def test_the_file_is_sent_with_a_name_and_a_content_type(request_for, audio):
    """The service rejects an upload it cannot identify."""
    client = FakeClient(transcription(SAMPLE_TEXT))
    build_provider(client).transcribe(request_for())

    name, _handle, content_type = client.transcriptions.calls[0]["file"]
    assert name == audio.name
    assert content_type == "audio/wav"


# -- Provenance ----------------------------------------------------------


def test_the_request_id_is_recorded_because_nothing_else_identifies_the_call(request_for):
    """The response has no model, no id and no fingerprint. The header is all there is."""
    client = FakeClient(transcription(SAMPLE_TEXT), headers={"x-request-id": "req_9f8e7d"})
    result = build_provider(client).transcribe(request_for())

    record = result.request
    assert record is not None
    assert record.provider is Provider.OPENAI
    assert record.model_identifier == MODEL_FAMILY
    assert record.request_parameters["response_request_id"] == "req_9f8e7d"
    assert record.chunks[0].provider_request_id == "req_9f8e7d"
    assert record.succeeded is True


def test_the_detected_languages_are_kept_as_a_second_opinion(request_for):
    """Whole-file rather than per-span, but no service reports language per word."""
    client = FakeClient(transcription(SAMPLE_TEXT, languages=("de", "en")))
    result = build_provider(client).transcribe(request_for())

    assert result.detected_language is Language.GERMAN
    assert result.request.request_parameters["response_languages"] == ["de", "en"]
    assert all(token.language is Language.UNKNOWN for token in result.tokens)


def test_the_body_the_server_sent_is_carried_back_unchanged(request_for):
    """The undecoded body is kept, not a copy rebuilt from the parsed answer.

    Only the original can tell a changed service from a changed application
    when a later run disagrees with this one. Anything reassembled from the
    parsed object would already have lost every field the installed library
    does not model, which is why the test body carries one.
    """
    body = json.dumps(
        {
            "text": SAMPLE_TEXT,
            "languages": ["en"],
            "usage": {"seconds": 12},
            "a_field_we_ignore": 42,
        }
    )
    client = FakeClient(transcription(SAMPLE_TEXT), body=body)

    result = build_provider(client).transcribe(request_for())

    assert result.raw_response == body
    assert "a_field_we_ignore" in result.raw_response
    assert json.loads(result.raw_response)["usage"] == {"seconds": 12}


def test_the_body_of_a_refusal_is_kept_as_well(request_for):
    """What a service said when it said no is often the most useful thing there."""
    body = '{"error": {"message": "Unrecognised file format", "code": "invalid_file"}}'
    failure = RuntimeError("Unrecognised file format")
    failure.status_code = 400
    failure.response = SimpleNamespace(text=body)
    client = FakeClient(error=failure)

    result = build_provider(client).transcribe(request_for())

    assert not result.succeeded
    assert result.raw_response == body


def test_no_credential_ever_reaches_the_record(request_for):
    client = FakeClient(transcription(SAMPLE_TEXT))
    result = build_provider(client).transcribe(
        request_for(extra_parameters={"api_key": API_KEY, "authorization": API_KEY, "seed": 3})
    )

    written = repr(result.request)
    assert API_KEY not in written
    extras = result.request.request_parameters["extra_parameters"]
    assert "api_key" not in extras
    assert "authorization" not in extras
    assert extras["seed"] == 3


# -- Failing well --------------------------------------------------------


def test_a_missing_key_is_reported_before_anything_is_sent(request_for):
    client = FakeClient(transcription(SAMPLE_TEXT))
    provider = build_provider(client, api_key="")

    assert provider.is_configured() is False
    assert provider.describe_configuration_problem() == "no API key has been entered"

    with pytest.raises(ProviderNotConfigured):
        provider._transcribe(request_for())

    result = provider.transcribe(request_for())
    assert not result.succeeded
    assert "Settings" in result.error
    assert client.transcriptions.calls == []


def test_a_missing_package_disables_only_this_service(request_for, monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    provider = build_provider(client=None)

    with pytest.raises(ProviderUnavailable):
        provider._transcribe(request_for())

    result = provider.transcribe(request_for())
    assert not result.succeeded
    assert "pip install openai" in result.error


def test_the_module_imports_with_no_package_installed(monkeypatch):
    """Importing must never need the library, or the application cannot start."""
    monkeypatch.setitem(sys.modules, "openai", None)
    reloaded = importlib.reload(openai_module)

    assert reloaded.MODEL_FAMILY == "gpt-transcribe"
    assert reloaded.OpenAiTranscriptionProvider(api_key="k").is_configured()


def test_a_service_error_becomes_a_failed_result(request_for):
    """Losing a service costs accuracy. It must never cost the transcript."""
    failure = RuntimeError("Rate limit reached for gpt-transcribe")
    failure.status_code = 429
    failure.request_id = "req_failed_1"
    client = FakeClient(error=failure)

    result = build_provider(client).transcribe(request_for())

    assert not result.succeeded
    assert "429" in result.error
    assert "Rate limit reached" in result.error
    assert result.tokens == []
    assert result.request is not None
    assert result.request.succeeded is False
    assert result.request.request_parameters["response_request_id"] == "req_failed_1"


def test_a_file_over_the_limit_is_refused_without_being_uploaded(request_for, monkeypatch):
    """Twenty-five megabytes is small, so this happens often and must be clear."""
    monkeypatch.setattr(openai_module, "_file_size", lambda path: 40 * 1024 * 1024)
    client = FakeClient(transcription(SAMPLE_TEXT))

    result = build_provider(client).transcribe(request_for())

    assert not result.succeeded
    assert "25 MB" in result.error
    assert client.transcriptions.calls == []


def test_the_run_can_be_stopped_before_the_request_goes(request_for):
    client = FakeClient(transcription(SAMPLE_TEXT))
    result = build_provider(client).transcribe(request_for(), cancelled=lambda: True)

    assert not result.succeeded
    assert client.transcriptions.calls == []


def test_an_empty_answer_is_an_empty_result_rather_than_a_failure(request_for):
    """Silence is a legitimate answer, and it is not an error."""
    client = FakeClient(transcription(""))
    result = build_provider(client).transcribe(request_for())

    assert result.succeeded
    assert result.tokens == []
