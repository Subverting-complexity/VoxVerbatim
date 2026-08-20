"""What the AssemblyAI adapter must get right.

Nothing here touches the network, and no API key is needed to run any of it.
A stand-in client records the parameters it was given and hands back the shape
the service documents.

Three of these tests guard against failures that are silent by nature. Word
times are milliseconds and the file duration beside them is seconds, so a
transcript can be a thousand times wrong and still look ordinary. Diarisation
without punctuation does not fail; it returns a transcript with no speakers in
it and no explanation. And the model chain means the model that ran is not
always the model that was asked for, which is invisible unless the answer's
own account of itself is written down.
"""

from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest

from audio_transcriber.transcription.model import AudioSpan, Language, Provider
from audio_transcriber.transcription.providers import assemblyai
from audio_transcriber.transcription.providers.assemblyai import AssemblyAiProvider
from audio_transcriber.transcription.providers.base import (
    ProviderError,
    ProviderNotConfigured,
    ProviderUnavailable,
    TranscriptionRequest,
)

API_KEY = "assemblyai-secret-key-4c71"

#: The answer the service documents. Word times are milliseconds; the
#: ``audio_duration`` beside them is seconds, and the two must not be treated
#: alike. Speakers are letters, which is this service's own convention.
TRANSCRIPT = {
    "id": "transcript-1",
    "status": "completed",
    "language_code": "en_us",
    "audio_duration": 57.2,
    "speech_model_used": "universal-2",
    "text": "Hello there again",
    "words": [
        {"text": "Hello", "start": 80, "end": 240, "confidence": 0.99, "speaker": "A"},
        {"text": "there", "start": 300, "end": 520, "confidence": 0.91, "speaker": "A"},
        {"text": "again", "start": 640, "end": 900, "confidence": 0.72, "speaker": "B"},
    ],
}


class FakeClient:
    """Stands in for the package's transcriber, with the two methods used here."""

    def __init__(
        self,
        transcript: Any = None,
        error: Exception | None = None,
        upload_url: str = "https://cdn.assemblyai.com/upload/abc",
        can_upload: bool = True,
    ) -> None:
        self.transcript = TRANSCRIPT if transcript is None else transcript
        self.error = error
        self.upload_url = upload_url
        self.calls: list[dict[str, Any]] = []
        self.uploaded: list[str] = []
        if not can_upload:
            # Some versions of the package cannot upload separately, and the
            # adapter has to notice rather than fall over.
            self.upload_file = None

    def transcribe(self, audio: Any, parameters: dict[str, Any]) -> Any:
        self.calls.append({"audio": audio, "parameters": dict(parameters)})
        if self.error is not None:
            raise self.error
        return self.transcript

    def upload_file(self, audio_path: str) -> str:
        self.uploaded.append(audio_path)
        return self.upload_url


class Boom(Exception):
    """A failure with a status, the way a client library reports one."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def audio_file(tmp_path: Path, name: str = "meeting.wav") -> Path:
    path = tmp_path / name
    path.write_bytes(b"\0" * 4096)
    return path


def make_request(tmp_path: Path, **overrides: Any) -> TranscriptionRequest:
    """A request with a non-zero offset, because zero hides the bug it exists for."""
    arguments: dict[str, Any] = {
        "audio_path": audio_file(tmp_path),
        "canonical_offset": 120.5,
        "languages": (Language.ENGLISH,),
    }
    arguments.update(overrides)
    return TranscriptionRequest(**arguments)


def make_provider(client: Any = None, **overrides: Any) -> AssemblyAiProvider:
    arguments: dict[str, Any] = {
        "api_key": API_KEY,
        "client": client if client is not None else FakeClient(),
    }
    arguments.update(overrides)
    return AssemblyAiProvider(**arguments)


def contains_text(value: Any, needle: str) -> bool:
    """Whether a recorded structure holds this text anywhere inside it."""
    if isinstance(value, dict):
        return any(
            contains_text(key, needle) or contains_text(item, needle)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(contains_text(item, needle) for item in value)
    return needle in str(value)


# -- Time, which is the thing that must not be wrong ---------------------


def test_word_times_are_canonical_seconds(tmp_path: Path) -> None:
    """Milliseconds become seconds and the canonical offset is added.

    The first word starts 80 milliseconds into audio that itself began 120.5
    seconds in, so it belongs at 120.58.
    """
    result = make_provider().transcribe(make_request(tmp_path))

    assert result.succeeded
    assert result.tokens[0].start == pytest.approx(120.58)
    assert result.tokens[0].end == pytest.approx(120.74)
    assert result.tokens[-1].end == pytest.approx(121.4)


def test_the_offset_is_actually_added(tmp_path: Path) -> None:
    at_zero = make_provider().transcribe(make_request(tmp_path, canonical_offset=0.0))
    later = make_provider().transcribe(make_request(tmp_path, canonical_offset=600.0))

    assert at_zero.tokens[0].start == pytest.approx(0.08)
    assert later.tokens[0].start == pytest.approx(600.08)


def test_the_file_duration_is_left_in_seconds(tmp_path: Path) -> None:
    """The classic bug on this API is a factor of a thousand in one direction."""
    result = make_provider().transcribe(make_request(tmp_path))

    assert result.request is not None
    recorded = result.request.request_parameters["response_audio_duration_seconds"]
    assert recorded == pytest.approx(57.2)


def test_word_confidence_and_speakers_are_kept_as_given(tmp_path: Path) -> None:
    """Speaker labels are letters here, and they stay letters."""
    result = make_provider().transcribe(make_request(tmp_path))

    assert [token.speaker for token in result.tokens] == ["A", "A", "B"]
    assert result.speakers == ["A", "B"]
    assert result.tokens[-1].confidence == pytest.approx(0.72)


# -- The model chain ------------------------------------------------------


def test_the_chain_is_the_main_model_then_the_afrikaans_one(tmp_path: Path) -> None:
    """One ordered list, not two requests. The fallback is the service's own."""
    client = FakeClient()
    make_provider(client).transcribe(make_request(tmp_path))

    parameters = client.calls[0]["parameters"]
    assert parameters["speech_models"] == ["universal-3-5-pro", "universal-2"]
    assert "speech_model" not in parameters


def test_a_model_override_goes_to_the_front_of_the_chain(tmp_path: Path) -> None:
    client = FakeClient()
    request = make_request(tmp_path, model_override="universal-2")
    make_provider(client).transcribe(request)

    assert client.calls[0]["parameters"]["speech_models"] == [
        "universal-2",
        "universal-3-5-pro",
    ]


def test_the_singular_parameter_is_never_sent(tmp_path: Path) -> None:
    """It was removed, and sending it returns an error rather than being ignored."""
    client = FakeClient()
    provider = make_provider(client, parameters={"speech_model": "universal-2"})
    provider.transcribe(make_request(tmp_path))

    assert "speech_model" not in client.calls[0]["parameters"]


def test_the_model_that_ran_is_recorded_not_the_one_asked_for(tmp_path: Path) -> None:
    """Falling through to Universal-2 is the thing provenance has to reveal."""
    result = make_provider().transcribe(make_request(tmp_path))

    record = result.request
    assert record is not None
    assert record.model_identifier == "universal-3-5-pro"
    assert record.model_version == "universal-2"
    assert record.request_parameters["response_speech_model_used"] == "universal-2"


# -- Diarisation, which fails quietly -------------------------------------


def test_speaker_labels_force_punctuation_on(tmp_path: Path) -> None:
    """Without punctuation the speakers simply do not appear, and nothing says so."""
    client = FakeClient()
    request = make_request(tmp_path, diarise=True, expected_speaker_count=3)
    make_provider(client).transcribe(request)

    parameters = client.calls[0]["parameters"]
    assert parameters["speaker_labels"] is True
    assert parameters["punctuate"] is True
    assert parameters["speakers_expected"] == 3


def test_a_setting_cannot_turn_punctuation_off_under_diarisation(tmp_path: Path) -> None:
    client = FakeClient()
    provider = make_provider(client, parameters={"punctuate": False})
    provider.transcribe(make_request(tmp_path, diarise=True))

    assert client.calls[0]["parameters"]["punctuate"] is True


def test_verbatim_is_always_asked_for(tmp_path: Path) -> None:
    client = FakeClient()
    make_provider(client).transcribe(make_request(tmp_path))

    assert client.calls[0]["parameters"]["disfluencies"] is True


# -- Windowed requests, which are the escalation path ---------------------


def test_a_window_is_sent_in_milliseconds_against_an_uploaded_url(tmp_path: Path) -> None:
    """Upload once, then ask about six seconds of it as often as needed.

    The audio path is never opened here: the whole point is that the recording
    is already on the service and only the question changes.
    """
    client = FakeClient()
    provider = make_provider(client)
    request = TranscriptionRequest(
        audio_path=tmp_path / "never-read.wav",
        canonical_offset=0.0,
        languages=(Language.ENGLISH,),
        purpose="escalation",
    )

    result = provider.transcribe_window(
        request,
        audio_url="https://cdn.assemblyai.com/upload/abc",
        window=AudioSpan(12.5, 18.25),
    )

    assert result.succeeded
    call = client.calls[0]
    assert call["audio"] == "https://cdn.assemblyai.com/upload/abc"
    assert call["parameters"]["audio_start_from"] == 12500
    assert call["parameters"]["audio_end_at"] == 18250
    assert result.request is not None and result.request.purpose == "escalation"


def test_a_window_on_the_request_is_used_without_being_repeated(tmp_path: Path) -> None:
    client = FakeClient()
    request = make_request(tmp_path, window=AudioSpan(4.0, 9.5))
    make_provider(client).transcribe(request)

    parameters = client.calls[0]["parameters"]
    assert parameters["audio_start_from"] == 4000
    assert parameters["audio_end_at"] == 9500


def test_a_windowed_answer_still_gets_the_canonical_offset(tmp_path: Path) -> None:
    """Escalation windows land on the same timeline as everything else."""
    client = FakeClient()
    request = make_request(tmp_path, canonical_offset=300.0)
    result = make_provider(client).transcribe_window(
        request, audio_url="https://cdn.assemblyai.com/upload/abc", window=AudioSpan(300.0, 306.0)
    )

    assert result.tokens[0].start == pytest.approx(300.08)


def test_uploading_returns_a_reusable_url(tmp_path: Path) -> None:
    client = FakeClient()
    url = make_provider(client).upload(audio_file(tmp_path))

    assert url == "https://cdn.assemblyai.com/upload/abc"
    assert len(client.uploaded) == 1


def test_a_package_that_cannot_upload_says_so_plainly(tmp_path: Path) -> None:
    """Losing this costs the cheap path, not transcription, so it is not fatal."""
    provider = make_provider(FakeClient(can_upload=False))

    with pytest.raises(ProviderError) as raised:
        provider.upload(audio_file(tmp_path))
    assert "cannot upload" in str(raised.value)


# -- Vocabulary -----------------------------------------------------------


def test_key_terms_are_cut_to_what_the_smaller_model_accepts(tmp_path: Path) -> None:
    client = FakeClient()
    terms = tuple(f"Term{number}" for number in range(250))
    make_provider(client).transcribe(make_request(tmp_path, vocabulary_terms=terms))

    sent = client.calls[0]["parameters"]["keyterms_prompt"]
    assert len(sent) == assemblyai.DEFAULT_MAXIMUM_KEYTERMS
    assert sent[0] == "Term0"


def test_a_term_longer_than_six_words_is_dropped_rather_than_shortened(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    terms = ("Contoso", "one two three four five six seven", "Jurgen Muller")
    make_provider(client).transcribe(make_request(tmp_path, vocabulary_terms=terms))

    assert client.calls[0]["parameters"]["keyterms_prompt"] == ["Contoso", "Jurgen Muller"]


def test_several_languages_ask_for_detection_rather_than_one_of_them(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    provider = make_provider(client, language_confidence_threshold=0.6)
    request = make_request(tmp_path, languages=(Language.ENGLISH, Language.AFRIKAANS))
    provider.transcribe(request)

    parameters = client.calls[0]["parameters"]
    assert parameters["language_detection"] is True
    assert parameters["language_confidence_threshold"] == 0.6
    assert "language_code" not in parameters


# -- Configuration, packaging and failure ---------------------------------


def test_a_missing_key_stops_the_request_before_it_is_made(tmp_path: Path) -> None:
    client = FakeClient()
    provider = make_provider(client, api_key="")

    with pytest.raises(ProviderNotConfigured):
        provider._transcribe(make_request(tmp_path))
    assert client.calls == []


def test_a_missing_package_disables_only_this_service(monkeypatch) -> None:
    block_import(monkeypatch, "assemblyai")
    provider = AssemblyAiProvider(api_key=API_KEY)

    with pytest.raises(ProviderUnavailable):
        provider._resolve_client()


def test_a_missing_package_becomes_a_result_rather_than_an_escape(
    monkeypatch, tmp_path: Path
) -> None:
    block_import(monkeypatch, "assemblyai")
    provider = AssemblyAiProvider(api_key=API_KEY)
    result = provider.transcribe(make_request(tmp_path))

    assert result.succeeded is False
    assert result.error is not None and "pip install assemblyai" in result.error


def test_the_module_imports_without_the_package(monkeypatch) -> None:
    block_import(monkeypatch, "assemblyai")
    reloaded = importlib.reload(assemblyai)

    assert reloaded.AssemblyAiProvider.provider is Provider.ASSEMBLYAI


def test_a_library_error_becomes_a_result(tmp_path: Path) -> None:
    client = FakeClient(error=Boom("rate limited", status_code=429))
    result = make_provider(client).transcribe(make_request(tmp_path))

    assert result.succeeded is False
    assert result.error is not None and "429" in result.error
    assert result.tokens == []
    assert result.request is not None and result.request.succeeded is False


def test_a_rejected_job_is_a_failure_even_though_nothing_was_raised(
    tmp_path: Path,
) -> None:
    """The library reports a failed job through the object it returns."""
    client = FakeClient({"id": "t2", "status": "error", "error": "audio was unreadable"})
    result = make_provider(client).transcribe(make_request(tmp_path))

    assert result.succeeded is False
    assert result.error is not None and "audio was unreadable" in result.error


class SdkTranscript:
    """Stands in for the library's own object, which keeps the JSON beside it."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.json_response = payload
        for key, value in payload.items():
            setattr(self, key, value)


def test_the_untouched_answer_comes_back_on_the_result(tmp_path: Path) -> None:
    """The response has to survive the adapter, or it cannot be stored at all."""
    result = make_provider().transcribe(make_request(tmp_path))

    assert result.raw_response == TRANSCRIPT


def test_the_libraries_own_json_is_preferred_to_the_parsed_object(
    tmp_path: Path,
) -> None:
    """Taken whole rather than rebuilt, because a rebuild is not the evidence."""
    payload = dict(TRANSCRIPT)
    client = FakeClient(SdkTranscript(payload))
    result = make_provider(client).transcribe(make_request(tmp_path))

    assert result.succeeded
    assert result.raw_response is payload


def test_a_rejected_job_keeps_the_body_that_explains_it(tmp_path: Path) -> None:
    rejection = {"id": "t2", "status": "error", "error": "audio was unreadable"}
    result = make_provider(FakeClient(rejection)).transcribe(make_request(tmp_path))

    assert result.succeeded is False
    assert result.raw_response == rejection


def test_nothing_is_invented_when_the_library_raised(tmp_path: Path) -> None:
    client = FakeClient(error=Boom("rate limited", status_code=429))
    result = make_provider(client).transcribe(make_request(tmp_path))

    assert result.raw_response is None


def test_the_recorded_request_holds_no_credential(tmp_path: Path) -> None:
    provider = make_provider(parameters={"api_key": API_KEY, "customField": "kept"})
    result = provider.transcribe(make_request(tmp_path, vocabulary_terms=("Contoso",)))

    record = result.request
    assert record is not None
    assert not contains_text(record.request_parameters, API_KEY)
    assert record.request_parameters["customField"] == "kept"
    assert record.vocabulary_terms == ("Contoso",)


def test_sdk_client_configures_api_key_and_http_timeout() -> None:
    """The SDK client must propagate timeout_seconds to assemblyai.settings.http_timeout."""
    import assemblyai as aai

    original_timeout = aai.settings.http_timeout
    try:
        assemblyai._build_client(API_KEY, 1234.5)
        assert aai.settings.api_key == API_KEY
        assert aai.settings.http_timeout == 1234.5
    finally:
        aai.settings.http_timeout = original_timeout


def block_import(monkeypatch, name: str) -> None:
    """Make one package look as though it was never installed."""
    real_import = builtins.__import__

    def fake_import(module: str, *args: Any, **kwargs: Any) -> Any:
        if module == name or module.startswith(f"{name}."):
            raise ImportError(f"No module named {module!r}")
        return real_import(module, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


# -- Parameters the library would drop on the floor ----------------------
#
# The package accepts any parameter set on its raw configuration and then
# builds the body it posts from a typed object that keeps only the fields it
# declares. Anything else is gone, and nothing is said. Settings promises that
# a parameter typed into the provider's list reaches the service, so the
# adapter has to notice and post the request itself. These tests hold that
# line, because the failure it prevents leaves no trace anywhere.


class FakeResponse:
    """One HTTP answer, with only what the code reads off it."""

    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def json(self) -> Any:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeHttpClient:
    """Stands in for the library's own HTTP client."""

    def __init__(self, answers: list[FakeResponse] | None = None) -> None:
        self.posts: list[dict[str, Any]] = []
        self.gets: list[str] = []
        self.answers = answers or []

    def post(self, url: str, json: dict[str, Any]) -> FakeResponse:
        self.posts.append({"url": url, "json": dict(json)})
        return self.answers.pop(0) if self.answers else FakeResponse({"id": "job-1"})

    def get(self, url: str) -> FakeResponse:
        self.gets.append(url)
        return self.answers.pop(0) if self.answers else FakeResponse(TRANSCRIPT)


class FakeRequestType:
    """The library's typed request object, which keeps only what it declares."""

    model_fields: ClassVar[dict[str, Any]] = {
        "audio_url": None,
        "speech_models": None,
        "punctuate": None,
        "disfluencies": None,
        "language_detection": None,
    }


class FakePackage:
    """The parts of the ``assemblyai`` package this client touches."""

    def __init__(self, request_type: Any = FakeRequestType) -> None:
        self.settings = type("Settings", (), {"api_key": "", "http_timeout": 0.0})()
        self.types = type("Types", (), {"TranscriptRequest": request_type})()
        self.transcribed: list[dict[str, Any]] = []
        package = self

        class Config:
            def __init__(self) -> None:
                self.raw = type("Raw", (), {})()

        class Transcriber:
            def __init__(self, config: Any = None) -> None:
                self.config = config

            def transcribe(self, audio: Any) -> Any:
                package.transcribed.append(
                    {"audio": audio, "raw": dict(vars(self.config.raw))}
                )
                return TRANSCRIPT

        self.TranscriptionConfig = Config
        self.Transcriber = Transcriber


def sdk_client_with(monkeypatch, http_client: FakeHttpClient, package: Any) -> Any:
    """An ``_SdkClient`` built on a stand-in package rather than the real one."""
    monkeypatch.setitem(sys.modules, "assemblyai", package)
    client = assemblyai._SdkClient(API_KEY, 30.0)
    client._http_client = http_client
    return client


def test_a_parameter_the_library_knows_goes_the_ordinary_way(monkeypatch) -> None:
    package = FakePackage()
    http = FakeHttpClient()
    client = sdk_client_with(monkeypatch, http, package)

    client.transcribe("https://cdn.assemblyai.com/a.wav", {"speech_models": ["universal-2"]})

    assert package.transcribed, "the library should have been used"
    assert http.posts == [], "nothing should have been posted directly"


def test_a_parameter_the_library_would_drop_is_posted_directly(monkeypatch) -> None:
    """The whole point. An unknown parameter must reach the service."""
    package = FakePackage()
    http = FakeHttpClient([FakeResponse({"id": "job-1"}), FakeResponse(TRANSCRIPT)])
    client = sdk_client_with(monkeypatch, http, package)

    answer = client.transcribe(
        "https://cdn.assemblyai.com/a.wav",
        {"speech_models": ["universal-2"], "some_new_option": "yes"},
    )

    assert package.transcribed == [], "the library would have dropped the new option"
    assert http.posts[0]["url"] == "/v2/transcript"
    assert http.posts[0]["json"]["some_new_option"] == "yes"
    assert http.posts[0]["json"]["speech_models"] == ["universal-2"]
    assert http.posts[0]["json"]["audio_url"] == "https://cdn.assemblyai.com/a.wav"
    assert answer == TRANSCRIPT


def test_a_local_file_is_uploaded_before_being_posted_directly(monkeypatch) -> None:
    package = FakePackage()
    http = FakeHttpClient([FakeResponse({"id": "job-1"}), FakeResponse(TRANSCRIPT)])
    client = sdk_client_with(monkeypatch, http, package)
    uploaded: list[str] = []

    def upload(path: str) -> str:
        uploaded.append(path)
        return "https://cdn.assemblyai.com/uploaded.wav"

    client.upload_file = upload

    client.transcribe("C:/recordings/meeting.wav", {"some_new_option": "yes"})

    assert uploaded == ["C:/recordings/meeting.wav"]
    assert http.posts[0]["json"]["audio_url"] == "https://cdn.assemblyai.com/uploaded.wav"


def test_a_directly_posted_job_is_waited_for(monkeypatch) -> None:
    package = FakePackage()
    http = FakeHttpClient(
        [
            FakeResponse({"id": "job-1", "status": "queued"}),
            FakeResponse({"id": "job-1", "status": "processing"}),
            FakeResponse(TRANSCRIPT),
        ]
    )
    client = sdk_client_with(monkeypatch, http, package)
    monkeypatch.setattr(assemblyai.time, "sleep", lambda _seconds: None)

    answer = client.transcribe("https://cdn.assemblyai.com/a.wav", {"some_new_option": 1})

    assert answer == TRANSCRIPT
    assert http.gets == ["/v2/transcript/job-1", "/v2/transcript/job-1"]


def test_a_refused_direct_request_carries_its_status(monkeypatch) -> None:
    """The status decides whether it is worth trying again, so it must survive."""
    package = FakePackage()
    http = FakeHttpClient([FakeResponse({"error": "speech_models is not valid"}, 400)])
    client = sdk_client_with(monkeypatch, http, package)

    with pytest.raises(assemblyai._DirectRequestError) as raised:
        client.transcribe("https://cdn.assemblyai.com/a.wav", {"some_new_option": 1})

    assert raised.value.status_code == 400
    assert "speech_models is not valid" in str(raised.value)

    _message, retryable, status = assemblyai._describe_failure(raised.value)
    assert status == 400
    assert retryable is False


def test_a_package_whose_shape_cannot_be_read_is_used_as_it_is(monkeypatch) -> None:
    """Not being able to tell is not a reason to take the long road."""
    package = FakePackage(request_type=type("Opaque", (), {}))
    http = FakeHttpClient()
    client = sdk_client_with(monkeypatch, http, package)

    client.transcribe("https://cdn.assemblyai.com/a.wav", {"some_new_option": 1})

    assert package.transcribed, "the library should have been used"
    assert http.posts == []


# -- Settings that would contradict each other ---------------------------


def test_a_language_named_in_settings_takes_the_detection_down_with_it(
    tmp_path: Path,
) -> None:
    """The service refuses a request that both names a language and detects one."""
    client = FakeClient()
    provider = make_provider(
        client,
        parameters={"language_code": "de"},
        language_confidence_threshold=0.6,
    )

    provider.transcribe(
        make_request(tmp_path, languages=(Language.ENGLISH, Language.AFRIKAANS))
    )

    sent = client.calls[0]["parameters"]
    assert sent["language_code"] == "de"
    assert "language_detection" not in sent
    assert "language_confidence_threshold" not in sent
