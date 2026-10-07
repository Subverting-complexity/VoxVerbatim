"""Tests for a key with a character that cannot be sent.

A key goes in a request header, and a header carries only ASCII. A key
copied through Word or Outlook can come back with an en dash in place of a
hyphen, and the client library then fails before anything is sent, with a
complaint about a codec that the report used to word as the service being
unreachable. These tests hold every service that takes a key to the same
promise: the report names the key and where the bad character is, nothing
is sent, and no part of the key is repeated anywhere.
"""

from __future__ import annotations

import pytest

from vox_verbatim.settings import TranscriptionSettings
from vox_verbatim.transcription.adjudication import Adjudicator
from vox_verbatim.transcription.model import Language, Provider
from vox_verbatim.transcription.providers import registry
from vox_verbatim.transcription.providers.base import (
    ProviderNotConfigured,
    TranscriptionRequest,
    describe_api_key_characters,
)
from vox_verbatim.transcription.providers.elevenlabs import ElevenLabsForcedAligner

#: An en dash at position 7, between two pieces that must never appear in
#: a message.
KEY_BEFORE = "sk_abc"
KEY_AFTER = "xyz789"
BAD_KEY = f"{KEY_BEFORE}–{KEY_AFTER}"

SERVICES = (
    (Provider.ELEVENLABS, "elevenlabs"),
    (Provider.OPENAI, "openai_transcription"),
    (Provider.MICROSOFT, "microsoft"),
    (Provider.ASSEMBLYAI, "assemblyai"),
    (Provider.DEEPGRAM, "deepgram"),
)


@pytest.fixture
def settings() -> TranscriptionSettings:
    settings = TranscriptionSettings()
    for _, section in SERVICES:
        getattr(settings, section).api_key = "good-key"
    settings.microsoft.endpoint = "https://example.cognitiveservices.azure.com"
    settings.deepgram.enabled = True
    return settings


def assert_names_the_key_and_hides_it(message: str, service_name: str) -> None:
    assert f"{service_name} API key has a character that is not allowed" in message
    assert "position 7" in message
    assert "could not be reached" not in message
    for piece in (KEY_BEFORE, KEY_AFTER, "–"):
        assert piece not in message


def test_the_first_character_outside_ascii_is_found_by_its_position():
    message = describe_api_key_characters("ElevenLabs", BAD_KEY)

    assert message is not None
    assert_names_the_key_and_hides_it(message, "ElevenLabs")
    assert "Word, Outlook or a notes app" in message


def test_a_key_of_plain_characters_has_nothing_to_report():
    assert describe_api_key_characters("ElevenLabs", "sk_abc-xyz.789_+/=") is None


@pytest.mark.parametrize("provider, section", SERVICES)
def test_each_service_reports_the_key_before_anything_is_sent(
    settings, tmp_path, provider, section
):
    getattr(settings, section).api_key = BAD_KEY
    adapter = registry.build_provider(provider, settings)
    assert adapter is not None

    problem = adapter.describe_configuration_problem()
    assert problem is not None
    assert_names_the_key_and_hides_it(problem, provider.display_name)
    assert adapter.is_configured() is False

    request = TranscriptionRequest(audio_path=tmp_path / "missing.wav", duration=1.0)
    with pytest.raises(ProviderNotConfigured):
        adapter._transcribe(request)
    result = adapter.transcribe(request)
    assert not result.succeeded
    assert_names_the_key_and_hides_it(result.error, provider.display_name)


@pytest.mark.parametrize("provider, section", SERVICES)
def test_a_plain_key_is_accepted_as_before(settings, provider, section):
    adapter = registry.build_provider(provider, settings)

    assert adapter.describe_configuration_problem() is None


def test_the_aligner_reports_the_key_too(tmp_path):
    aligner = ElevenLabsForcedAligner(api_key=BAD_KEY)

    assert aligner.is_configured() is False
    with pytest.raises(ProviderNotConfigured) as raised:
        aligner.align(tmp_path / "missing.wav", "Hello world", Language.ENGLISH)
    assert_names_the_key_and_hides_it(str(raised.value), Provider.ELEVENLABS.display_name)


def test_the_adjudication_model_reports_the_key_too():
    adjudicator = Adjudicator(api_key=BAD_KEY, model="some-model")

    problem = adjudicator.describe_configuration_problem()
    assert problem is not None
    assert_names_the_key_and_hides_it(problem, "OpenAI")
