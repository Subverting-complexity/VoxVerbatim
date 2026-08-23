"""What the Microsoft adapter must get right.

Nothing here touches the network, and no API key is needed to run any of it.
The adapter is driven with a stand-in for the HTTP client that records what it
was posted and hands back the exact response shape Microsoft documents, which
is enough to check every decision the adapter makes.

Several of these tests exist because the mistake they catch would not show
itself in the output. A missing canonical offset produces a transcript where
every word points at the wrong sound while every number looks reasonable. A
stored confidence of zero reads later as a service that was certain it heard
nothing of the sort. A profanity filter left on returns a transcript that is
merely a little different. So the offset is always non-zero in these tests,
and the two dangerous defaults are checked directly.
"""

from __future__ import annotations

import builtins
import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from vox_verbatim.transcription.model import Language, Provider
from vox_verbatim.transcription.providers import microsoft
from vox_verbatim.transcription.providers.base import (
    ProviderNotConfigured,
    ProviderUnavailable,
    TranscriptionRequest,
)
from vox_verbatim.transcription.providers.microsoft import MicrosoftProvider

API_KEY = "microsoft-secret-key-9f2a"
ENDPOINT = "https://my-resource.cognitiveservices.azure.com"

#: The answer Microsoft documents, with two phrases in two languages so that
#: the per-phrase locale can be seen arriving on the right words. Times are
#: milliseconds given as an offset and a duration, which is the shape that
#: makes this service different from the other four.
RESPONSE = {
    "durationMilliseconds": 57187,
    "combinedPhrases": [{"text": "With lockdown. Guten Tag."}],
    "phrases": [
        {
            "offsetMilliseconds": 80,
            "durationMilliseconds": 6960,
            "text": "With lockdown.",
            "locale": "en-us",
            "confidence": 0,
            "words": [
                {"text": "With", "offsetMilliseconds": 80, "durationMilliseconds": 160},
                {"text": "lockdown.", "offsetMilliseconds": 300, "durationMilliseconds": 500},
            ],
        },
        {
            "offsetMilliseconds": 7040,
            "durationMilliseconds": 2000,
            "text": "Guten Tag.",
            "locale": "de-DE",
            "confidence": 0,
            "words": [
                {"text": "Guten", "offsetMilliseconds": 7040, "durationMilliseconds": 400},
                {"text": "Tag.", "offsetMilliseconds": 7500, "durationMilliseconds": 540},
            ],
        },
    ],
}


class FakeResponse:
    """What a real HTTP client would hand back, with nothing else on it."""

    def __init__(
        self,
        payload: Any = None,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {"apim-request-id": "azure-request-1"}
        self.text = payload if isinstance(payload, str) else ""

    def json(self) -> Any:
        if isinstance(self._payload, dict):
            return self._payload
        raise ValueError("the body is not JSON")


class FakeHttpClient:
    """Stands in for ``httpx.Client`` and remembers what it was asked to post."""

    def __init__(self, response: FakeResponse | None = None, error: Exception | None = None):
        self.response = response or FakeResponse(RESPONSE)
        self.error = error
        self.posts: list[dict[str, Any]] = []

    def post(self, url: str, headers=None, files=None, timeout=None) -> FakeResponse:
        definition = files["definition"][1] if files else "{}"
        audio_part = files["audio"] if files else None
        self.posts.append(
            {
                "url": url,
                "headers": dict(headers or {}),
                "definition": json.loads(definition),
                "audio_name": audio_part[0] if audio_part else None,
                # Read here, because a real client would read the handle before
                # the adapter closes the file.
                "audio_bytes": audio_part[1].read() if audio_part else b"",
                "timeout": timeout,
            }
        )
        if self.error is not None:
            raise self.error
        return self.response


class Boom(Exception):
    """An HTTP failure with a status, the way a client library reports one."""

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


def make_provider(client: Any = None, **overrides: Any) -> MicrosoftProvider:
    arguments: dict[str, Any] = {
        "api_key": API_KEY,
        "endpoint": ENDPOINT,
        "client": client if client is not None else FakeHttpClient(),
    }
    arguments.update(overrides)
    return MicrosoftProvider(**arguments)


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


def test_times_are_canonical_seconds(tmp_path: Path) -> None:
    """Milliseconds become seconds, duration becomes an end, and the offset lands.

    The first word starts 80 milliseconds into audio that itself began 120.5
    seconds into the recording, and lasts 160 milliseconds, so it belongs at
    120.58 to 120.74 on the canonical timeline.
    """
    provider = make_provider()
    result = provider.transcribe(make_request(tmp_path))

    assert result.succeeded
    first = result.tokens[0]
    assert first.text == "With"
    assert first.start == pytest.approx(120.58)
    assert first.end == pytest.approx(120.74)

    last = result.tokens[-1]
    assert last.start == pytest.approx(120.5 + 7.5)
    assert last.end == pytest.approx(120.5 + 8.04)


def test_the_offset_is_actually_added(tmp_path: Path) -> None:
    """The same audio at two offsets must not produce the same times."""
    at_zero = make_provider().transcribe(make_request(tmp_path, canonical_offset=0.0))
    later = make_provider().transcribe(make_request(tmp_path, canonical_offset=600.0))

    assert at_zero.tokens[0].start == pytest.approx(0.08)
    assert later.tokens[0].start == pytest.approx(600.08)


def test_the_file_duration_is_recorded_in_seconds(tmp_path: Path) -> None:
    provider = make_provider()
    result = provider.transcribe(make_request(tmp_path))

    assert result.request is not None
    assert result.request.request_parameters["response_duration_seconds"] == pytest.approx(57.187)


# -- The two things peculiar to this service -----------------------------


def test_a_confidence_of_zero_is_not_stored(tmp_path: Path) -> None:
    """The service reports zero for everything, so it reports nothing.

    A stored zero would be read further on as a measurement, and it would say
    that Microsoft was certain it had heard something other than what it wrote
    down. An absent confidence says the truth, which is that this service
    contributes no acoustic confidence at all.
    """
    result = make_provider().transcribe(make_request(tmp_path))

    assert [token.confidence for token in result.tokens] == [None, None, None, None]
    assert MicrosoftProvider.capabilities.word_confidence is False


def test_each_phrase_lends_its_language_to_its_words(tmp_path: Path) -> None:
    """The per-phrase locale is the point of this service, so it must arrive."""
    result = make_provider().transcribe(make_request(tmp_path))

    assert [token.language for token in result.tokens] == [
        Language.ENGLISH,
        Language.ENGLISH,
        Language.GERMAN,
        Language.GERMAN,
    ]
    assert MicrosoftProvider.capabilities.per_word_language is True


def test_no_speakers_are_invented(tmp_path: Path) -> None:
    result = make_provider().transcribe(make_request(tmp_path))

    assert result.speakers == []
    assert all(token.speaker is None for token in result.tokens)
    assert MicrosoftProvider.capabilities.diarisation is False


# -- The request itself ---------------------------------------------------


def test_the_two_dangerous_defaults_are_always_overridden(tmp_path: Path) -> None:
    """Verbatim style and no profanity filter, on every request without exception."""
    client = FakeHttpClient()
    make_provider(client).transcribe(make_request(tmp_path))

    definition = client.posts[0]["definition"]
    assert definition["enhancedMode"]["transcribeStyle"] == "verbatim"
    assert definition["profanityFilterMode"] == "None"
    assert definition["enhancedMode"]["model"] == microsoft.DEFAULT_MODEL
    assert definition["enhancedMode"]["enabled"] is True


def test_settings_cannot_turn_the_verbatim_style_off(tmp_path: Path) -> None:
    """A setting that would tidy the transcript is refused rather than obeyed."""
    client = FakeHttpClient()
    provider = make_provider(
        client,
        parameters={
            "profanityFilterMode": "Masked",
            "enhancedMode": {"enabled": True, "transcribeStyle": "readable"},
        },
    )
    provider.transcribe(make_request(tmp_path))

    definition = client.posts[0]["definition"]
    assert definition["profanityFilterMode"] == "None"
    assert definition["enhancedMode"]["transcribeStyle"] == "verbatim"
    assert definition["enhancedMode"]["model"] == microsoft.DEFAULT_MODEL


def test_the_definition_travels_as_a_multipart_part(tmp_path: Path) -> None:
    """It reads like a JSON API and is not one, which is the easy mistake here."""
    client = FakeHttpClient()
    make_provider(client).transcribe(make_request(tmp_path))

    post = client.posts[0]
    assert post["url"] == (
        f"{ENDPOINT}/speechtotext/transcriptions:transcribe"
        f"?api-version={microsoft.DEFAULT_API_VERSION}"
    )
    assert post["headers"] == {"Ocp-Apim-Subscription-Key": API_KEY}
    assert post["audio_name"] == "meeting.wav"
    assert post["audio_bytes"] == b"\0" * 4096


def test_vocabulary_goes_in_the_phrase_list(tmp_path: Path) -> None:
    client = FakeHttpClient()
    request = make_request(tmp_path, vocabulary_terms=("Contoso", "Jurgen Muller"))
    make_provider(client).transcribe(request)

    assert client.posts[0]["definition"]["phraseList"] == {
        "phrases": ["Contoso", "Jurgen Muller"]
    }


def test_afrikaans_is_never_offered(tmp_path: Path) -> None:
    """The service has no Afrikaans, so asking for it would invite a wrong answer."""
    client = FakeHttpClient()
    request = make_request(tmp_path, languages=(Language.AFRIKAANS, Language.ENGLISH))
    make_provider(client).transcribe(request)

    assert client.posts[0]["definition"]["locales"] == ["en"]


# -- Configuration, packaging and failure ---------------------------------


def test_a_missing_key_stops_the_request_before_it_is_made(tmp_path: Path) -> None:
    client = FakeHttpClient()
    provider = make_provider(client, api_key="")

    with pytest.raises(ProviderNotConfigured):
        provider._transcribe(make_request(tmp_path))
    assert client.posts == []


def test_a_missing_endpoint_says_what_it_wants(tmp_path: Path) -> None:
    provider = make_provider(FakeHttpClient(), endpoint="")

    problem = provider.describe_configuration_problem()
    assert problem is not None
    assert "Speech resource URL" in problem
    assert provider.is_configured() is False


def test_a_missing_package_disables_only_this_service(monkeypatch) -> None:
    """No httpx means no Microsoft, and a sentence saying how to fix it."""
    block_import(monkeypatch, "httpx")
    provider = MicrosoftProvider(api_key=API_KEY, endpoint=ENDPOINT)

    with pytest.raises(ProviderUnavailable):
        provider._resolve_client()


def test_the_module_imports_without_the_package(monkeypatch) -> None:
    """Importing the adapter must never need the client library."""
    block_import(monkeypatch, "httpx")
    reloaded = importlib.reload(microsoft)

    assert reloaded.MicrosoftProvider.provider is Provider.MICROSOFT


def test_a_network_failure_becomes_a_result(tmp_path: Path) -> None:
    client = FakeHttpClient(error=Boom("the connection dropped"))
    result = make_provider(client).transcribe(make_request(tmp_path))

    assert result.succeeded is False
    assert result.error is not None
    assert "could not be reached" in result.error
    assert result.tokens == []
    assert result.request is not None and result.request.succeeded is False


def test_an_http_error_becomes_a_result(tmp_path: Path) -> None:
    client = FakeHttpClient(
        FakeResponse({"error": {"code": "TooManyRequests", "message": "slow down"}}, 429)
    )
    result = make_provider(client).transcribe(make_request(tmp_path))

    assert result.succeeded is False
    assert result.error is not None
    assert "429" in result.error and "slow down" in result.error


def test_a_rejected_model_names_the_regions_that_serve_it(tmp_path: Path) -> None:
    """A resource in the wrong region looks exactly like a misspelled model."""
    client = FakeHttpClient(
        FakeResponse(
            {"error": {"code": "InvalidRequest", "message": "The model is not supported."}},
            400,
        )
    )
    result = make_provider(client).transcribe(make_request(tmp_path))

    assert result.error is not None
    for region in microsoft.MAI_REGIONS:
        assert region in result.error


def test_the_wrong_container_is_refused_before_uploading(tmp_path: Path) -> None:
    client = FakeHttpClient()
    request = make_request(tmp_path, audio_path=audio_file(tmp_path, "meeting.m4a"))
    result = make_provider(client).transcribe(request)

    assert result.succeeded is False
    assert result.error is not None and "flac, mp3, wav" in result.error
    assert client.posts == []


def test_the_untouched_answer_comes_back_on_the_result(tmp_path: Path) -> None:
    """The response has to survive the adapter, or it cannot be stored at all.

    This service is the clean case: the request went over plain HTTP, so the
    parsed body is exactly what arrived, with no library objects in between.
    The adapter carries it and writes nothing itself.
    """
    result = make_provider().transcribe(make_request(tmp_path))

    assert result.raw_response == RESPONSE
    # The same object, not a rebuilt copy. A reconstruction would be evidence
    # of what the adapter understood rather than of what the service said.
    assert result.raw_response is RESPONSE


def test_a_refusal_keeps_the_body_that_explains_it(tmp_path: Path) -> None:
    """Whether it refused the model or the region is the whole question."""
    body = {"error": {"code": "InvalidRequest", "message": "The model is not supported."}}
    client = FakeHttpClient(FakeResponse(body, 400))
    result = make_provider(client).transcribe(make_request(tmp_path))

    assert result.succeeded is False
    assert result.raw_response == body


def test_nothing_is_invented_when_nothing_came_back(tmp_path: Path) -> None:
    client = FakeHttpClient(error=Boom("the connection dropped"))
    result = make_provider(client).transcribe(make_request(tmp_path))

    assert result.raw_response is None


def test_the_recorded_request_holds_no_credential(tmp_path: Path) -> None:
    """A request record is written beside the recording and read by people."""
    request = make_request(tmp_path, vocabulary_terms=("Contoso",))
    result = make_provider().transcribe(request)

    record = result.request
    assert record is not None
    assert not contains_text(record.request_parameters, API_KEY)
    assert not contains_text(record.request_parameters, "Ocp-Apim-Subscription-Key")
    assert record.request_parameters["url"].startswith(ENDPOINT)


def test_a_credential_in_the_settings_extras_is_stripped(tmp_path: Path) -> None:
    provider = make_provider(parameters={"api_key": API_KEY, "customField": "kept"})
    result = provider.transcribe(make_request(tmp_path))

    record = result.request
    assert record is not None
    assert not contains_text(record.request_parameters, API_KEY)
    assert record.request_parameters["definition"]["customField"] == "kept"


# -- The language the service is asked to listen for ----------------------


def test_a_bilingual_recording_is_not_forced_into_one_language(tmp_path: Path) -> None:
    """No locale at all, which is what leaves the service multi-lingual.

    This is the whole reason Microsoft is here. Naming a locale forces
    recognition into that language, so a recording that may be English or
    German has to name none, or every phrase comes back English and the one
    per-segment language signal in the application becomes a constant.
    """
    client = FakeHttpClient()
    make_provider(client).transcribe(make_request(tmp_path))

    assert "locales" not in client.posts[0]["definition"]


def test_one_language_is_named(tmp_path: Path) -> None:
    """Where there is only one, saying so is what the field is for."""
    client = FakeHttpClient()
    request = make_request(tmp_path, languages=(Language.GERMAN,))
    make_provider(client).transcribe(request)

    assert client.posts[0]["definition"]["locales"] == ["de"]


def test_an_unknown_language_names_none(tmp_path: Path) -> None:
    """Knowing nothing is not a reason to guess English."""
    client = FakeHttpClient()
    request = make_request(tmp_path, languages=(Language.UNKNOWN,))
    make_provider(client).transcribe(request)

    assert "locales" not in client.posts[0]["definition"]
    assert client.posts[0]["definition"]["enhancedMode"]["model"] == microsoft.DEFAULT_MODEL


def test_a_settings_locale_displaces_the_adapters_own(tmp_path: Path) -> None:
    """The adapter's rule is a default, not a ban: a setting still wins.

    The recording here is in one language, so the adapter would name a locale
    of its own. That is what the setting has to displace. A bilingual recording
    would prove nothing, because there would be no locale to displace.
    """
    client = FakeHttpClient()
    provider = make_provider(client, parameters={"locales": ["en-GB"]})
    provider.transcribe(make_request(tmp_path, languages=(Language.GERMAN,)))

    assert client.posts[0]["definition"]["locales"] == ["en-GB"]


# -- An answer that says nothing -----------------------------------------


def test_an_unreadable_answer_is_a_failure_not_a_silent_recording(tmp_path: Path) -> None:
    """A 200 whose body is not a transcript must not read as a silent recording.

    This is the one failure here that would otherwise be invisible. No phrases
    means no words, and a result with no words and no error is exactly what a
    recording in which nobody spoke produces. The run would be called a
    success and the recording left looking silent.
    """
    client = FakeHttpClient(FakeResponse("<html>504 Gateway Timeout</html>", 200))
    result = make_provider(client).transcribe(make_request(tmp_path))

    assert result.succeeded is False
    assert result.tokens == []
    assert result.error is not None and "not a transcription result" in result.error
    # Whatever arrived is the only account of what went wrong.
    assert result.raw_response == "<html>504 Gateway Timeout</html>"


def test_an_answer_with_no_phrases_field_is_a_failure(tmp_path: Path) -> None:
    """A truncated object is as unreadable as a page of HTML."""
    client = FakeHttpClient(FakeResponse({"durationMilliseconds": 4000}, 200))
    result = make_provider(client).transcribe(make_request(tmp_path))

    assert result.succeeded is False
    assert result.error is not None and "not a transcription result" in result.error


def test_a_transcript_with_no_phrases_is_a_failure(tmp_path: Path) -> None:
    """Words with nowhere to put them are reported, not quietly dropped."""
    body = {
        "durationMilliseconds": 4000,
        "combinedPhrases": [{"text": "Good morning."}],
        "phrases": [],
    }
    client = FakeHttpClient(FakeResponse(body, 200))
    result = make_provider(client).transcribe(make_request(tmp_path))

    assert result.succeeded is False
    assert result.error is not None and "no phrases" in result.error
    assert result.raw_response == body


def test_a_single_phrase_where_a_list_belongs_is_a_failure(tmp_path: Path) -> None:
    """Iterating an object yields its keys, and keys make words out of nothing.

    This is the shape the guard exists for. Every field would read as missing,
    every word would come out empty, and no error would be raised anywhere.
    """
    body = {
        "durationMilliseconds": 4000,
        "combinedPhrases": [{"text": "Good morning."}],
        "phrases": {"text": "Good morning.", "offsetMilliseconds": 0},
    }
    client = FakeHttpClient(FakeResponse(body, 200))
    result = make_provider(client).transcribe(make_request(tmp_path))

    assert result.succeeded is False
    assert result.tokens == []


def test_a_genuinely_silent_recording_still_succeeds(tmp_path: Path) -> None:
    """Nothing said and nothing returned is an answer, not a fault."""
    body = {"durationMilliseconds": 4000, "combinedPhrases": [{"text": ""}], "phrases": []}
    client = FakeHttpClient(FakeResponse(body, 200))
    result = make_provider(client).transcribe(make_request(tmp_path))

    assert result.succeeded is True
    assert result.tokens == []


# -- The model that takes neither a style nor a phrase list ---------------


def test_the_older_model_is_sent_no_style_or_phrase_list(tmp_path: Path) -> None:
    """Both arrived with 1.5. Sending them to the model before it is a refusal."""
    client = FakeHttpClient()
    provider = make_provider(client, model="mai-transcribe-1")
    provider.transcribe(make_request(tmp_path, vocabulary_terms=("Contoso",)))

    definition = client.posts[0]["definition"]
    assert "transcribeStyle" not in definition["enhancedMode"]
    assert "phraseList" not in definition
    assert definition["enhancedMode"]["model"] == "mai-transcribe-1"


def test_the_older_model_cannot_be_given_a_style_by_settings(tmp_path: Path) -> None:
    client = FakeHttpClient()
    provider = make_provider(
        client,
        model="mai-transcribe-1",
        parameters={
            "enhancedMode": {"transcribeStyle": "verbatim"},
            "phraseList": {"biasingWeight": 1.5},
        },
    )
    provider.transcribe(make_request(tmp_path, vocabulary_terms=("Contoso",)))

    definition = client.posts[0]["definition"]
    assert "transcribeStyle" not in definition["enhancedMode"]
    assert "phraseList" not in definition


def test_an_unrecognised_model_keeps_the_style(tmp_path: Path) -> None:
    """A model newer than this file is assumed to take what 1.5 takes.

    The other assumption is worse: a model released next year would quietly
    lose its verbatim style and return a tidied transcript, and nothing would
    say so.
    """
    client = FakeHttpClient()
    provider = make_provider(client, model="mai-transcribe-2")
    provider.transcribe(make_request(tmp_path))

    assert client.posts[0]["definition"]["enhancedMode"]["transcribeStyle"] == "verbatim"


# -- Settings extras must not take the vocabulary away --------------------


def test_a_phrase_list_setting_keeps_the_recording_vocabulary(tmp_path: Path) -> None:
    """A biasing weight added in Settings must not replace the terms.

    The two are separate fields of one object, and a straight replacement
    would send the weight with no phrases to weight, which is a request that
    succeeds and biases nothing.
    """
    client = FakeHttpClient()
    provider = make_provider(client, parameters={"phraseList": {"biasingWeight": 1.5}})
    provider.transcribe(make_request(tmp_path, vocabulary_terms=("Contoso", "Rehaan")))

    assert client.posts[0]["definition"]["phraseList"] == {
        "phrases": ["Contoso", "Rehaan"],
        "biasingWeight": 1.5,
    }


def test_a_phrase_list_setting_adds_to_the_recording_vocabulary(tmp_path: Path) -> None:
    """Phrases named in Settings join the recording's terms rather than replacing them.

    A setting applies to everything transcribed on this machine and the terms
    belong to this one recording, so the general one must not displace the
    particular one. Both lists are biased for, and the recording's come first.
    """
    client = FakeHttpClient()
    provider = make_provider(client, parameters={"phraseList": {"phrases": ["Fabrikam"]}})
    provider.transcribe(make_request(tmp_path, vocabulary_terms=("Contoso",)))

    assert client.posts[0]["definition"]["phraseList"] == {
        "phrases": ["Contoso", "Fabrikam"]
    }


def test_a_biasing_weight_with_nothing_to_weight_is_not_sent(tmp_path: Path) -> None:
    """A phrase list of one number and no phrases is not a phrase list.

    It would go out on every recording that has no vocabulary of its own, which
    is most of them, and it biases nothing at best.
    """
    client = FakeHttpClient()
    provider = make_provider(client, parameters={"phraseList": {"biasingWeight": 1.5}})
    provider.transcribe(make_request(tmp_path))

    assert "phraseList" not in client.posts[0]["definition"]


# -- The record has to match the request ----------------------------------


def test_the_record_names_the_terms_that_were_actually_sent(tmp_path: Path) -> None:
    """A model that takes no phrase list was sent no terms, and the record says so.

    The record exists to answer one question: whether a later run that produced
    different words made the same request. A record naming terms the service
    never received cannot answer it.
    """
    client = FakeHttpClient()
    provider = make_provider(client, model="mai-transcribe-1")
    result = provider.transcribe(make_request(tmp_path, vocabulary_terms=("Contoso",)))

    assert "phraseList" not in client.posts[0]["definition"]
    record = result.request
    assert record is not None
    assert record.vocabulary_terms == ()
    assert record.request_parameters["vocabulary_term_count"] == 0


def test_the_record_names_the_phrases_a_setting_added(tmp_path: Path) -> None:
    client = FakeHttpClient()
    provider = make_provider(client, parameters={"phraseList": {"phrases": ["Fabrikam"]}})
    result = provider.transcribe(make_request(tmp_path, vocabulary_terms=("Contoso",)))

    record = result.request
    assert record is not None
    assert record.vocabulary_terms == ("Contoso", "Fabrikam")


def test_the_record_names_the_locale_that_was_actually_sent(tmp_path: Path) -> None:
    """A locale from Settings is what the service heard, so it is what is written."""
    client = FakeHttpClient()
    provider = make_provider(client, parameters={"locales": ["en-GB"]})
    result = provider.transcribe(make_request(tmp_path))

    record = result.request
    assert record is not None
    assert record.language_configuration == "en-GB"


def test_a_multi_lingual_request_is_recorded_as_automatic(tmp_path: Path) -> None:
    client = FakeHttpClient()
    result = make_provider(client).transcribe(make_request(tmp_path))

    record = result.request
    assert record is not None
    assert record.language_configuration == "auto"


# -- The limits one request may carry -------------------------------------


def test_the_limits_are_the_ones_the_endpoint_documents() -> None:
    """250 MB and two hours, which is the smaller of the two published pairs.

    The MAI page asks for under 300 MB and says nothing about length; the API
    definition for this dated version says under 250 MB and under two hours.
    Planning to the smaller pair costs one more chunk boundary, and a file over
    the endpoint's own ceiling is refused outright.
    """
    assert microsoft.MAXIMUM_FILE_BYTES == 250 * 1024 * 1024
    assert microsoft.MAXIMUM_DURATION_SECONDS == 2 * 60 * 60
    assert microsoft.MICROSOFT_CAPABILITIES.maximum_file_bytes == microsoft.MAXIMUM_FILE_BYTES
    assert (
        microsoft.MICROSOFT_CAPABILITIES.maximum_duration_seconds
        == microsoft.MAXIMUM_DURATION_SECONDS
    )


def block_import(monkeypatch, name: str) -> None:
    """Make one package look as though it was never installed."""
    real_import = builtins.__import__

    def fake_import(module: str, *args: Any, **kwargs: Any) -> Any:
        if module == name or module.startswith(f"{name}."):
            raise ImportError(f"No module named {module!r}")
        return real_import(module, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


# -- Retrying --------------------------------------------------------------
#
# This adapter talks to the service over a bare HTTP client, which retries
# nothing. The retry is the base class's; these tests hold the two together.


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    from vox_verbatim.transcription.providers import base

    monkeypatch.setattr(base, "_sleep", lambda seconds: None)


class FlakyHttpClient(FakeHttpClient):
    """Fails a scripted number of times before answering."""

    def __init__(self, errors: list[Exception], response: FakeResponse | None = None) -> None:
        super().__init__(response)
        self.errors = errors

    def post(self, url: str, headers=None, files=None, timeout=None) -> FakeResponse:
        if self.errors:
            # Read the file first, as a real client would before the
            # connection dropped.
            files["audio"][1].read()
            self.posts.append({"dropped": True})
            raise self.errors.pop(0)
        return super().post(url, headers=headers, files=files, timeout=timeout)


def test_a_dropped_connection_is_tried_again_with_the_whole_file(tmp_path: Path) -> None:
    class ConnectError(Exception):
        pass

    class RemoteProtocolError(Exception):
        pass

    client = FlakyHttpClient([ConnectError("reset"), RemoteProtocolError("closed")])

    result = make_provider(client, maximum_retries=2).transcribe(make_request(tmp_path))

    assert result.succeeded
    assert len(client.posts) == 3
    assert client.posts[-1]["audio_bytes"] == b"\0" * 4096


def test_a_server_error_is_tried_again_and_the_count_is_reported(tmp_path: Path) -> None:
    client = FakeHttpClient(FakeResponse({"error": {"message": "busy"}}, status_code=503))

    result = make_provider(client, maximum_retries=2).transcribe(make_request(tmp_path))

    assert not result.succeeded
    assert len(client.posts) == 3
    assert "failed after 3 attempts" in result.error
    assert result.request is not None and result.request.error == result.error
    # The body of the last refusal is still kept.
    assert result.raw_response == {"error": {"message": "busy"}}


def test_a_rejected_key_is_not_tried_again(tmp_path: Path) -> None:
    client = FakeHttpClient(FakeResponse({"error": {"message": "no"}}, status_code=401))

    result = make_provider(client, maximum_retries=3).transcribe(make_request(tmp_path))

    assert not result.succeeded
    assert len(client.posts) == 1
    assert "attempts" not in result.error


def test_the_wait_a_rate_limit_asks_for_is_read_off_the_response(
    monkeypatch, tmp_path: Path
) -> None:
    from vox_verbatim.transcription.providers import base

    waits: list[float] = []
    monkeypatch.setattr(base, "_sleep", waits.append)
    refused = FakeResponse(
        {"error": {"message": "slow down"}},
        status_code=429,
        headers={"Retry-After": "9", "apim-request-id": "r"},
    )
    client = FakeHttpClient(refused)

    make_provider(client, maximum_retries=1).transcribe(make_request(tmp_path))

    assert sum(waits) == pytest.approx(9.0)


def test_a_negative_duration_does_not_put_the_end_before_the_start(tmp_path: Path) -> None:
    body = {
        "phrases": [
            {
                "offsetMilliseconds": 1000,
                "durationMilliseconds": 500,
                "text": "Odd",
                "locale": "en-us",
                "words": [{"text": "Odd", "offsetMilliseconds": 1000, "durationMilliseconds": -200}],
            }
        ]
    }
    client = FakeHttpClient(FakeResponse(body))

    result = make_provider(client).transcribe(make_request(tmp_path))

    token = result.tokens[0]
    assert token.end == token.start == pytest.approx(120.5 + 1.0)
