"""What the Deepgram adapter must get right.

Nothing here touches the network, and no API key is needed to run any of it.
A stand-in client records the options it was given and hands back the shape the
service documents.

Two of these tests guard against a request that fails outright rather than
degrading. Sending both diarisation parameters is rejected, not warned about,
and a key-term list one token over the cap fails the whole request instead of
being trimmed. Both would take a working service to nothing, and both are
settled inside the adapter so that no setting can cause them.

The offset test matters as much here as anywhere, and for a reason peculiar to
this service: its times are already in seconds, so there is no unit conversion
to remind anyone that a conversion is due. The canonical offset still has to be
added, and forgetting it produces a transcript in which no single number looks
wrong.

One group of tests near the end is different in kind from the rest. Everything
else drives the adapter through a stand-in client, which is the right way to
test what the adapter decides but says nothing about whether the real package
can be called at all. That gap once hid a complete outage: the library was
rebuilt around a different entry point, the declared floor still allowed the
new one to be installed, and every request raised on the first line of the
call while the whole suite stayed green. Those tests therefore pin the call
itself, against a stand-in package built to the library's shape, and one of
them checks the installed package really has that shape when it is there to
check.
"""

from __future__ import annotations

import builtins
import importlib
import inspect
import json
import logging
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from vox_verbatim.transcription.model import Language, Provider
from vox_verbatim.transcription.providers import deepgram
from vox_verbatim.transcription.providers.base import (
    ProviderNotConfigured,
    ProviderUnavailable,
    TranscriptionRequest,
)
from vox_verbatim.transcription.providers.deepgram import DeepgramProvider

API_KEY = "deepgram-secret-key-77b3"

#: The answer the service documents. Times are floating-point seconds and
#: speakers are integers, which is the opposite of AssemblyAI on both counts.
#: ``model_info`` is keyed by an identifier that is not known in advance and
#: holds the dated build string that makes this service's provenance the best
#: of the five.
RESPONSE = {
    "metadata": {
        "request_id": "dg-request-1",
        "duration": 57.187,
        "channels": 1,
        "model_info": {
            "1a2b3c": {"name": "nova-3", "version": "2026-06-01.4821", "arch": "nova"}
        },
        # An architecture and the identifier of the build that ran, which is
        # what the service documents here. Not a name and a version.
        "diarize_info": {"model_uuid": "e2ec-diarizer", "arch": "v2"},
    },
    "results": {
        "channels": [
            {
                "detected_language": "en",
                "alternatives": [
                    {
                        "transcript": "Hello there again",
                        "confidence": 0.98,
                        # A multilingual answer lists what it found here,
                        # most words first. This is the only place the usual
                        # request gets a language back.
                        "languages": ["en", "de"],
                        "words": [
                            {
                                "word": "hello",
                                "punctuated_word": "Hello",
                                "start": 0.44,
                                "end": 0.72,
                                "confidence": 0.99,
                                "speaker": 0,
                            },
                            {
                                "word": "there",
                                "punctuated_word": "there,",
                                "start": 0.8,
                                "end": 1.1,
                                "confidence": 0.95,
                                "speaker": 0,
                            },
                            {
                                "word": "again",
                                "punctuated_word": "again.",
                                "start": 1.2,
                                "end": 1.6,
                                "confidence": 0.61,
                                "speaker": 1,
                            },
                        ],
                    }
                ],
            }
        ]
    },
}


class FakeClient:
    """Stands in for the package's client, with the one method used here."""

    def __init__(
        self,
        response: Any = None,
        error: Exception | None = None,
    ) -> None:
        self.response = RESPONSE if response is None else response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, audio: Any, options: dict[str, Any], timeout: float | None = None) -> Any:
        self.calls.append(
            {
                "options": dict(options),
                "timeout": timeout,
                # Read here, because a real client would read the handle before
                # the adapter closes the file.
                "audio_bytes": audio.read() if hasattr(audio, "read") else audio,
            }
        )
        if self.error is not None:
            raise self.error
        return self.response


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
        "languages": (Language.ENGLISH, Language.GERMAN),
    }
    arguments.update(overrides)
    return TranscriptionRequest(**arguments)


def make_provider(client: Any = None, **overrides: Any) -> DeepgramProvider:
    arguments: dict[str, Any] = {
        "api_key": API_KEY,
        "client": client if client is not None else FakeClient(),
    }
    arguments.update(overrides)
    return DeepgramProvider(**arguments)


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


def test_times_are_seconds_with_the_canonical_offset_added(tmp_path: Path) -> None:
    """No unit conversion here, and the offset still has to be added."""
    result = make_provider().transcribe(make_request(tmp_path))

    assert result.succeeded
    assert result.tokens[0].start == pytest.approx(120.94)
    assert result.tokens[0].end == pytest.approx(121.22)
    assert result.tokens[-1].end == pytest.approx(122.1)


def test_the_offset_is_actually_added(tmp_path: Path) -> None:
    at_zero = make_provider().transcribe(make_request(tmp_path, canonical_offset=0.0))
    later = make_provider().transcribe(make_request(tmp_path, canonical_offset=600.0))

    assert at_zero.tokens[0].start == pytest.approx(0.44)
    assert later.tokens[0].start == pytest.approx(600.44)


def test_the_punctuated_word_is_what_the_service_said(tmp_path: Path) -> None:
    result = make_provider().transcribe(make_request(tmp_path))

    assert [token.text for token in result.tokens] == ["Hello", "there,", "again."]
    assert result.tokens[-1].confidence == pytest.approx(0.61)


def test_speaker_numbers_are_kept_as_the_service_gave_them(tmp_path: Path) -> None:
    """Integers here, letters at AssemblyAI. Neither is renamed at this level."""
    result = make_provider().transcribe(make_request(tmp_path))

    assert [token.speaker for token in result.tokens] == ["0", "0", "1"]
    assert result.speakers == ["0", "1"]
    assert result.detected_language is Language.ENGLISH


# -- The two parameters that would fail the whole request -----------------


def test_diarisation_uses_the_new_parameter_only(tmp_path: Path) -> None:
    """Sending both is rejected outright, so only one is ever sent."""
    client = FakeClient()
    make_provider(client).transcribe(make_request(tmp_path, diarise=True))

    options = client.calls[0]["options"]
    assert options["diarize_model"] == "latest"
    assert "diarize" not in options


def test_a_setting_cannot_add_the_deprecated_diarise_switch(
    tmp_path: Path, caplog
) -> None:
    client = FakeClient()
    provider = make_provider(client, parameters={"diarize": True})
    with caplog.at_level(logging.WARNING):
        provider.transcribe(make_request(tmp_path, diarise=True))

    options = client.calls[0]["options"]
    assert "diarize" not in options
    assert options["diarize_model"] == "latest"
    assert "rejected" in caplog.text


def test_key_terms_are_cut_before_they_can_fail_the_request(
    tmp_path: Path, caplog
) -> None:
    """Over the cap is an error rather than a truncation, so the cut happens here.

    Each of these terms is two words, which is budgeted at ten of the five
    hundred tokens a request may carry, so fifty of them fit and the rest are
    left out and said to be left out.
    """
    client = FakeClient()
    terms = tuple(f"Term{number} Muller" for number in range(200))
    with caplog.at_level(logging.INFO):
        make_provider(client).transcribe(make_request(tmp_path, vocabulary_terms=terms))

    sent = client.calls[0]["options"]["keyterm"]
    assert len(sent) == 50
    assert sent[0] == "Term0 Muller"
    assert "left out" in caplog.text


def test_a_short_vocabulary_is_sent_whole(tmp_path: Path) -> None:
    client = FakeClient()
    make_provider(client).transcribe(
        make_request(tmp_path, vocabulary_terms=("Contoso", "Jurgen Muller"))
    )

    assert client.calls[0]["options"]["keyterm"] == ["Contoso", "Jurgen Muller"]


# -- The rest of the request ----------------------------------------------


def test_two_languages_ask_for_the_multilingual_model(tmp_path: Path) -> None:
    """A single code would transcribe one language and ignore the other."""
    client = FakeClient()
    make_provider(client).transcribe(make_request(tmp_path))

    assert client.calls[0]["options"]["language"] == "multi"


def test_one_language_is_named_outright(tmp_path: Path) -> None:
    client = FakeClient()
    make_provider(client).transcribe(make_request(tmp_path, languages=(Language.GERMAN,)))

    assert client.calls[0]["options"]["language"] == "de"


def test_afrikaans_never_narrows_the_request(tmp_path: Path) -> None:
    """No Deepgram model speaks it, so it must not become the language asked for."""
    client = FakeClient()
    make_provider(client).transcribe(make_request(tmp_path, languages=(Language.AFRIKAANS,)))

    assert client.calls[0]["options"]["language"] == "multi"
    assert Language.AFRIKAANS not in DeepgramProvider.capabilities.languages


def test_the_model_and_the_verbatim_switch_are_always_sent(tmp_path: Path) -> None:
    client = FakeClient()
    make_provider(client).transcribe(make_request(tmp_path))

    options = client.calls[0]["options"]
    assert options["model"] == "nova-3"
    assert options["filler_words"] is True


# -- Provenance, which is this service's strong point ---------------------


def test_the_dated_build_of_the_model_is_recorded(tmp_path: Path) -> None:
    result = make_provider().transcribe(make_request(tmp_path))

    record = result.request
    assert record is not None
    assert record.model_identifier == "nova-3"
    assert record.model_version == "nova-3 2026-06-01.4821"
    assert record.request_parameters["response_diarizer"] == "v2 e2ec-diarizer"
    assert record.request_parameters["response_request_id"] == "dg-request-1"
    assert record.request_parameters["response_duration_seconds"] == pytest.approx(57.187)


class SdkResponse:
    """Stands in for the library's own object, which converts itself back."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.metadata = payload["metadata"]
        self.results = payload["results"]

    def to_dict(self) -> dict[str, Any]:
        return self._payload


def test_the_untouched_answer_comes_back_on_the_result(tmp_path: Path) -> None:
    """The response has to survive the adapter, or it cannot be stored at all."""
    result = make_provider().transcribe(make_request(tmp_path))

    assert result.raw_response == RESPONSE


def test_the_libraries_own_structure_is_preferred_to_a_rebuild(tmp_path: Path) -> None:
    """Taken whole rather than rebuilt, because a rebuild is not the evidence."""
    payload = dict(RESPONSE)
    client = FakeClient(SdkResponse(payload))
    result = make_provider(client).transcribe(make_request(tmp_path))

    assert result.succeeded
    assert result.raw_response is payload
    # And the words were still read correctly off the object itself.
    assert [token.text for token in result.tokens] == ["Hello", "there,", "again."]


def test_nothing_is_invented_when_the_library_raised(tmp_path: Path) -> None:
    client = FakeClient(error=Boom("service unavailable", status_code=503))
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


# -- Configuration, packaging and failure ---------------------------------


def test_a_missing_key_stops_the_request_before_it_is_made(tmp_path: Path) -> None:
    client = FakeClient()
    provider = make_provider(client, api_key="")

    with pytest.raises(ProviderNotConfigured):
        provider._transcribe(make_request(tmp_path))
    assert client.calls == []


def test_a_missing_package_disables_only_this_service(monkeypatch) -> None:
    block_import(monkeypatch, "deepgram")
    provider = DeepgramProvider(api_key=API_KEY)

    with pytest.raises(ProviderUnavailable):
        provider._resolve_client()


def test_a_missing_package_becomes_a_result_rather_than_an_escape(
    monkeypatch, tmp_path: Path
) -> None:
    block_import(monkeypatch, "deepgram")
    provider = DeepgramProvider(api_key=API_KEY)
    result = provider.transcribe(make_request(tmp_path))

    assert result.succeeded is False
    assert result.error is not None and "pip install deepgram-sdk" in result.error


def test_the_module_imports_without_the_package(monkeypatch) -> None:
    block_import(monkeypatch, "deepgram")
    reloaded = importlib.reload(deepgram)

    assert reloaded.DeepgramProvider.provider is Provider.DEEPGRAM


def test_a_library_error_becomes_a_result(tmp_path: Path) -> None:
    client = FakeClient(error=Boom("service unavailable", status_code=503))
    result = make_provider(client).transcribe(make_request(tmp_path))

    assert result.succeeded is False
    assert result.error is not None and "503" in result.error
    assert result.tokens == []
    assert result.request is not None and result.request.succeeded is False


def test_a_504_is_reported_as_the_processing_budget_it_is(tmp_path: Path) -> None:
    """It says nothing about how long the audio may be, so it must not read that way."""
    client = FakeClient(error=Boom("gateway timeout", status_code=504))
    result = make_provider(client).transcribe(make_request(tmp_path))

    assert result.succeeded is False
    assert result.error is not None
    assert "ten-minute processing budget" in result.error
    assert "callback mode" in result.error



def test_a_refusal_is_reported_in_the_words_the_service_used(tmp_path: Path) -> None:
    """Not as the block of headers the library prints itself as.

    The library raises one error type whose printed form is every response
    header, the status and the body run together. Handed on unchanged that
    whole block becomes the warning shown against the transcript, and the one
    sentence in it that says what went wrong is buried.
    """

    class ApiFailure(Exception):
        def __init__(self) -> None:
            super().__init__(
                "headers: {'dg-request-id': 'abc'}, status_code: 400, "
                "body: {'err_code': 'INVALID_AUTH', 'err_msg': 'Project not found'}"
            )
            self.status_code = 400
            self.body = {"err_code": "INVALID_AUTH", "err_msg": "Project not found"}

    result = make_provider(FakeClient(error=ApiFailure())).transcribe(make_request(tmp_path))

    assert result.error is not None
    assert "Project not found (INVALID_AUTH)" in result.error
    assert "headers:" not in result.error


def test_a_failure_with_nothing_to_read_still_says_something(tmp_path: Path) -> None:
    """A body that carries no explanation falls back on the printed exception."""
    result = make_provider(FakeClient(error=Boom("connection reset"))).transcribe(
        make_request(tmp_path)
    )

    assert result.error is not None and "connection reset" in result.error


def block_import(monkeypatch, name: str) -> None:
    """Make one package look as though it was never installed."""
    real_import = builtins.__import__

    def fake_import(module: str, *args: Any, **kwargs: Any) -> Any:
        if module == name or module.startswith(f"{name}."):
            raise ImportError(f"No module named {module!r}")
        return real_import(module, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


# -- The answer itself ----------------------------------------------------


def test_the_dominant_language_is_read_off_the_alternative(tmp_path: Path) -> None:
    """Where the usual request actually gets a language back.

    The request names ``multi``, and a multilingual answer lists what it found
    on the alternative rather than on the channel, ordered by how many words
    were in each. Reading only the channel meant no ordinary request ever
    reported a language at all.
    """
    result = make_provider().transcribe(make_request(tmp_path))

    assert result.detected_language is Language.ENGLISH


def test_a_language_on_the_channel_is_still_read(tmp_path: Path) -> None:
    """Asking for detection outright puts it in the other place."""
    payload = json.loads(json.dumps(RESPONSE))
    channel = payload["results"]["channels"][0]
    channel["alternatives"][0].pop("languages")
    channel["detected_language"] = "de"

    result = make_provider(FakeClient(payload)).transcribe(make_request(tmp_path))

    assert result.detected_language is Language.GERMAN


def test_an_answer_with_no_transcript_is_a_failure(tmp_path: Path) -> None:
    """Not a recording of silence, which is what an empty success would say.

    A reply carrying no channel is the acknowledgement a callback request
    gets, or a body that is not a transcript at all. Reported as a success it
    would be indistinguishable from a recording in which nobody spoke, and the
    run would be over with nothing to show that anything went wrong.
    """
    # The real shape: a request id at the top level and no metadata at all.
    client = FakeClient({"request_id": "dg-accepted-1"})
    result = make_provider(client).transcribe(make_request(tmp_path))

    assert result.succeeded is False
    assert result.error is not None and "without a transcript" in result.error
    assert result.tokens == []
    assert result.request is not None and result.request.succeeded is False
    # The reply is kept, because it is the only account of what happened.
    assert result.raw_response == {"request_id": "dg-accepted-1"}
    # And the identifier is recorded, because the transcript for a callback
    # request arrives elsewhere and this is the only way to go and find it.
    assert result.request.request_parameters["response_request_id"] == "dg-accepted-1"
    assert result.request.chunks[0].provider_request_id == "dg-accepted-1"


def test_an_answer_with_no_words_is_still_a_success(tmp_path: Path) -> None:
    """A channel that came back empty really is a piece of audio with no speech."""
    payload = json.loads(json.dumps(RESPONSE))
    payload["results"]["channels"][0]["alternatives"][0]["words"] = []

    result = make_provider(FakeClient(payload)).transcribe(make_request(tmp_path))

    assert result.succeeded is True
    assert result.tokens == []


def test_the_diarizer_is_read_from_the_keys_the_service_uses(tmp_path: Path) -> None:
    """It reports an architecture and a build identifier, not a name and a version."""
    result = make_provider().transcribe(make_request(tmp_path))

    record = result.request
    assert record is not None
    assert record.request_parameters["response_diarizer"] == "v2 e2ec-diarizer"


def test_a_response_that_only_dumps_itself_is_still_read(tmp_path: Path) -> None:
    """The current library has no ``to_dict``; it has ``model_dump``.

    Everything this adapter reads comes off the plain structure, so a response
    object that cannot be converted takes the provenance down with it.
    """

    class DumpingResponse:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def model_dump(self) -> dict[str, Any]:
            return self._payload

    payload = json.loads(json.dumps(RESPONSE))
    result = make_provider(FakeClient(DumpingResponse(payload))).transcribe(make_request(tmp_path))

    assert result.succeeded
    assert result.raw_response is payload
    assert result.request is not None
    assert result.request.model_version == "nova-3 2026-06-01.4821"


# -- Key terms and the model that has to understand them ------------------


def test_key_terms_are_withheld_from_a_model_that_cannot_use_them(tmp_path: Path, caplog) -> None:
    """Only Nova-3 and Flux take a key-term list. Others would ignore it or refuse it."""
    client = FakeClient()
    provider = make_provider(client, model="nova-2")
    terms = ("Contoso", "Fabrikam")

    with caplog.at_level(logging.WARNING):
        result = provider.transcribe(make_request(tmp_path, vocabulary_terms=terms))

    assert "keyterm" not in client.calls[0]["options"]
    assert result.request is not None and result.request.vocabulary_terms == ()
    assert "nova-2" in caplog.text


def test_a_nova_three_variant_still_takes_key_terms(tmp_path: Path) -> None:
    """The family has variants, and a medical Nova-3 is still a Nova-3."""
    client = FakeClient()
    provider = make_provider(client, model="nova-3-medical")
    provider.transcribe(make_request(tmp_path, vocabulary_terms=("Contoso",)))

    assert client.calls[0]["options"]["keyterm"] == ["Contoso"]


# -- Calling the package that is actually installed -----------------------
#
# These are the only tests here that care about the library's own shape. See
# the note at the top of this file for why they exist.


class FakeMedia:
    """The transcription entry point of the library, recording how it was called."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def transcribe_file(self, *, request: Any, request_options: dict[str, Any]) -> Any:
        self.calls.append({"audio_bytes": request.read(), "request_options": dict(request_options)})
        return RESPONSE


class FakeDeepgramClient:
    """The client of the library, with the accessor chain version 5 introduced."""

    last: FakeDeepgramClient | None = None

    def __init__(self, *, api_key: str) -> None:
        self.api_key = api_key
        self.media = FakeMedia()
        self.listen = type("Listen", (), {"v1": type("V1", (), {"media": self.media})()})()
        FakeDeepgramClient.last = self


def install_fake_package(monkeypatch, client_type: Any) -> None:
    """Put a stand-in ``deepgram`` package where the adapter will import it."""
    module = types.ModuleType("deepgram")
    module.DeepgramClient = client_type  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deepgram", module)


def test_the_key_is_given_to_the_library_by_keyword(monkeypatch) -> None:
    """Positionally it raises, which is how this service was silently switched off."""
    install_fake_package(monkeypatch, FakeDeepgramClient)

    deepgram._SdkClient(API_KEY)

    assert FakeDeepgramClient.last is not None
    assert FakeDeepgramClient.last.api_key == API_KEY


def test_every_option_goes_through_the_query_parameters(monkeypatch, tmp_path: Path) -> None:
    """The generated method raises on a name it does not declare, so nothing is passed to it.

    This is what keeps the promise Settings makes. A parameter typed into the
    provider's parameter list has to reach the service whether or not the
    installed package has heard of it.
    """
    install_fake_package(monkeypatch, FakeDeepgramClient)
    provider = DeepgramProvider(api_key=API_KEY, parameters={"some_future_parameter": "value"})

    result = provider.transcribe(make_request(tmp_path, vocabulary_terms=("Contoso",)))

    assert result.succeeded
    client = FakeDeepgramClient.last
    assert client is not None
    call = client.media.calls[0]
    sent = call["request_options"]["additional_query_parameters"]
    assert sent["model"] == "nova-3"
    assert sent["language"] == "multi"
    assert sent["keyterm"] == ["Contoso"]
    assert sent["some_future_parameter"] == "value"
    # The audio itself went, rather than being described.
    assert call["audio_bytes"] == b"\0" * 4096


def test_the_library_is_not_allowed_to_retry_by_itself(monkeypatch, tmp_path: Path) -> None:
    """Its retries repost a stream already read to the end, and lose the status code.

    The second attempt carries no audio, and what comes back is a protocol
    error with no status on it. A 503 this application would have retried
    arrives instead as a nameless failure that looks permanent.
    """
    install_fake_package(monkeypatch, FakeDeepgramClient)

    DeepgramProvider(api_key=API_KEY, timeout_seconds=42.0).transcribe(make_request(tmp_path))

    client = FakeDeepgramClient.last
    assert client is not None
    options = client.media.calls[0]["request_options"]
    assert options["max_retries"] == 0
    assert options["timeout"] == 42.0


def test_a_package_too_old_to_call_says_so(monkeypatch, tmp_path: Path) -> None:
    """Rather than an attribute error naming whichever link of the chain is missing."""

    class OldClient:
        def __init__(self, *, api_key: str) -> None:
            self.listen = object()

    install_fake_package(monkeypatch, OldClient)
    result = DeepgramProvider(api_key=API_KEY).transcribe(make_request(tmp_path))

    assert result.succeeded is False
    assert result.error is not None
    assert "too old" in result.error
    assert f"{deepgram.PACKAGE}>={deepgram.MINIMUM_PACKAGE_VERSION}" in result.error


def test_the_installed_package_has_the_shape_this_adapter_calls() -> None:
    """The one test that looks at the real library, and the one that would have caught it.

    Skipped where the package is not installed, because it is optional. Where
    it is installed, it has to have the entry point this adapter uses, take the
    key by keyword, and accept the two arguments the request is built from.
    """
    pytest.importorskip("deepgram", reason=f"{deepgram.PACKAGE} is not installed")
    from deepgram import DeepgramClient

    media = deepgram._media_client_of(DeepgramClient(api_key=API_KEY))
    parameters = inspect.signature(media.transcribe_file).parameters
    assert "request" in parameters
    assert "request_options" in parameters


# -- The answer has to survive being written down -------------------------


def test_the_answer_can_be_written_as_json(tmp_path: Path) -> None:
    """The one thing the stored answer has to be, and the easy way to lose it.

    The library parses the reply into Python objects, and the ``created``
    timestamp becomes a real ``datetime``. A dictionary holding one of those
    is written nowhere. It reaches the transcript store at the very end of a
    pass, after every service has answered and been paid for, and the run dies
    there rather than losing only this service.
    """
    typed = pytest.importorskip(
        "deepgram.types.listen_v1response", reason="deepgram-sdk is not installed"
    )
    from deepgram.core.pydantic_utilities import parse_obj_as

    body = json.loads(json.dumps(RESPONSE))
    # The fields the library declares and this fixture otherwise leaves out.
    body["metadata"].update(
        {"sha256": "154e", "created": "2024-05-12T18:57:13.426Z", "models": ["1a2b3c"]}
    )
    response = parse_obj_as(typed.ListenV1Response, body)

    result = make_provider(FakeClient(response)).transcribe(make_request(tmp_path))

    assert result.succeeded
    # Would raise TypeError on a datetime.
    written = json.dumps(result.raw_response)
    assert "2024-05-12T18:57:13" in written
    # And the provenance survived the conversion.
    assert result.request is not None
    assert result.request.model_version == "nova-3 2026-06-01.4821"


# -- The model, which two decisions depend on -----------------------------


def test_a_setting_that_names_a_model_decides_the_key_terms(tmp_path: Path) -> None:
    """The gate has to ask the model that is really sent, not the one configured.

    The free-form settings are merged last and may name a model of their own.
    A gate reading the constructor would pass a Nova-3 while a Nova-2 went out
    beside the key terms, which Deepgram refuses outright.
    """
    client = FakeClient()
    provider = make_provider(client, model="nova-3", parameters={"model": "nova-2"})
    provider.transcribe(make_request(tmp_path, vocabulary_terms=("Contoso",)))

    options = client.calls[0]["options"]
    assert options["model"] == "nova-2"
    assert "keyterm" not in options


def test_a_setting_can_also_open_the_key_terms_up(tmp_path: Path) -> None:
    """The same reading, the other way round."""
    client = FakeClient()
    provider = make_provider(client, model="nova-2", parameters={"model": "nova-3"})
    provider.transcribe(make_request(tmp_path, vocabulary_terms=("Contoso",)))

    options = client.calls[0]["options"]
    assert options["model"] == "nova-3"
    assert options["keyterm"] == ["Contoso"]


def test_the_model_goes_out_tidied(tmp_path: Path) -> None:
    """Deepgram takes the name literally, so a stray space is a rejected request."""
    client = FakeClient()
    make_provider(client, model="  Nova-3  ").transcribe(make_request(tmp_path))

    assert client.calls[0]["options"]["model"] == "nova-3"


def test_provenance_names_the_model_that_was_sent(tmp_path: Path) -> None:
    """A record whose named model disagrees with its own parameters explains nothing."""
    provider = make_provider(model="nova-3", parameters={"model": "nova-2"})
    result = provider.transcribe(make_request(tmp_path))

    record = result.request
    assert record is not None
    assert record.model_identifier == "nova-2"
    assert record.request_parameters["model"] == "nova-2"


# -- Trying again, which nothing else in the application does -------------


class FailingMedia(FakeMedia):
    """Fails a set number of times before answering, recording each body."""

    def __init__(self, error: Exception, failures: int) -> None:
        super().__init__()
        self.error = error
        self.failures = failures
        self.bodies: list[int] = []

    def transcribe_file(self, *, request: Any, request_options: dict[str, Any]) -> Any:
        self.bodies.append(len(request.read()))
        self.calls.append({"request_options": dict(request_options)})
        if len(self.calls) <= self.failures:
            raise self.error
        return RESPONSE


def failing_package(monkeypatch, media: FakeMedia) -> None:
    """A stand-in package whose client hands back this particular entry point."""

    class Client:
        def __init__(self, *, api_key: str) -> None:
            self.listen = type("Listen", (), {"v1": type("V1", (), {"media": media})()})()

    install_fake_package(monkeypatch, Client)
    # The waiting is real time, and nothing here is worth waiting for.
    monkeypatch.setattr(deepgram.time, "sleep", lambda seconds: None)


def test_a_failure_worth_repeating_is_repeated_with_the_audio_rewound(
    monkeypatch, tmp_path: Path
) -> None:
    """The library repeats too, but posts nothing the second time.

    That is the whole reason the retry lives in this adapter: an attempt reads
    the handle to the end, and a repeat that does not go back to the start
    sends an empty body and gets a meaningless answer to it.
    """
    media = FailingMedia(Boom("service unavailable", status_code=503), failures=2)
    failing_package(monkeypatch, media)

    result = DeepgramProvider(api_key=API_KEY).transcribe(make_request(tmp_path))

    assert result.succeeded
    assert len(media.calls) == 3
    # Every attempt carried the whole recording, not just the first.
    assert media.bodies == [4096, 4096, 4096]


def test_a_failure_not_worth_repeating_is_not_repeated(monkeypatch, tmp_path: Path) -> None:
    """A rejected key is the same rejection however many times it is sent."""
    media = FailingMedia(Boom("unauthorised", status_code=401), failures=1)
    failing_package(monkeypatch, media)

    result = DeepgramProvider(api_key=API_KEY).transcribe(make_request(tmp_path))

    assert result.succeeded is False
    assert len(media.calls) == 1


def test_the_processing_budget_is_never_spent_twice(monkeypatch, tmp_path: Path) -> None:
    """A 504 here means the request took too long, and repeating it takes as long again."""
    media = FailingMedia(Boom("gateway timeout", status_code=504), failures=1)
    failing_package(monkeypatch, media)

    result = DeepgramProvider(api_key=API_KEY).transcribe(make_request(tmp_path))

    assert result.succeeded is False
    assert len(media.calls) == 1
    assert result.error is not None and "ten-minute processing budget" in result.error


def test_giving_up_reports_the_last_failure(monkeypatch, tmp_path: Path) -> None:
    """Three attempts and no answer is still a failure, described as one."""
    media = FailingMedia(Boom("service unavailable", status_code=503), failures=99)
    failing_package(monkeypatch, media)

    result = DeepgramProvider(api_key=API_KEY).transcribe(make_request(tmp_path))

    assert result.succeeded is False
    assert len(media.calls) == 3
    assert result.error is not None and "503" in result.error
