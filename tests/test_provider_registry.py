"""Tests for turning the user's settings into the services that get called.

The registry is a joint, and a joint is where things come apart. Two
questions are asked of it here. Does a service the user switched off stay
switched off, all the way down to there being no adapter for anything to
call? And does what the user typed into Settings arrive at the adapter
unchanged, under the name that adapter expects?

The second question is answered by putting a recorder in place of each
adapter class and reading what it was handed, rather than by building a
real adapter and inspecting its insides. What matters is the wiring, and a
test that reached into an adapter's private attributes would break every
time an adapter was tidied up without the wiring having changed at all.

Nothing here needs a provider library installed, and one test deliberately
takes one away.
"""

from __future__ import annotations

import sys

import pytest

from vox_verbatim.settings import TranscriptionSettings
from vox_verbatim.transcription.model import Provider
from vox_verbatim.transcription.providers import (
    assemblyai as assemblyai_module,
    deepgram as deepgram_module,
    elevenlabs as elevenlabs_module,
    microsoft as microsoft_module,
    openai as openai_module,
    registry,
)


class Recorded:
    """Stands in for an adapter class and remembers how it was built."""

    def __init__(self, **arguments) -> None:
        self.arguments = arguments


#: Each service, the settings section it reads, and the name of the class
#: the registry is expected to build. Kept as one table because every test
#: below wants the same five facts and writing them out five times invites
#: one of them to fall behind.
SERVICES = (
    (Provider.ELEVENLABS, "elevenlabs", elevenlabs_module, "ElevenLabsProvider"),
    (Provider.OPENAI, "openai_transcription", openai_module, "OpenAiTranscriptionProvider"),
    (Provider.MICROSOFT, "microsoft", microsoft_module, "MicrosoftProvider"),
    (Provider.ASSEMBLYAI, "assemblyai", assemblyai_module, "AssemblyAiProvider"),
    (Provider.DEEPGRAM, "deepgram", deepgram_module, "DeepgramProvider"),
)


@pytest.fixture
def settings() -> TranscriptionSettings:
    """Settings with every service switched on and filled in.

    Deepgram is off by default because nothing in version 1 depends on it,
    so it is switched on here to give the tests below a fifth service to
    reason about rather than a special case.
    """
    settings = TranscriptionSettings()
    settings.elevenlabs.api_key = "elevenlabs-key"
    settings.openai_transcription.api_key = "openai-key"
    settings.microsoft.api_key = "microsoft-key"
    settings.microsoft.endpoint = "https://example.cognitiveservices.azure.com"
    settings.assemblyai.api_key = "assemblyai-key"
    settings.deepgram.api_key = "deepgram-key"
    settings.deepgram.enabled = True
    return settings


def record(monkeypatch, module, class_name: str) -> None:
    """Put the recorder in place of one adapter class."""
    monkeypatch.setattr(module, class_name, Recorded)


# -- Switching a service off ---------------------------------------------


@pytest.mark.parametrize("provider, section, module, class_name", SERVICES)
def test_a_service_that_is_switched_off_produces_no_adapter(
    settings, provider, section, module, class_name
):
    """Nothing to call, rather than an adapter that refuses to run.

    The difference matters upstream: the caller's list of services is meant
    to be the list it will actually pay for, and an adapter that exists only
    to say no would be counted, priced and reported as a service.
    """
    setattr(getattr(settings, section), "enabled", False)

    assert registry.build_provider(provider, settings) is None


def test_switching_a_service_off_leaves_the_others_alone(settings):
    settings.microsoft.enabled = False

    built = registry.build_providers(settings)

    assert set(built) == {Provider.ELEVENLABS, Provider.OPENAI}


def test_switching_everything_off_gives_a_run_with_nothing_in_it(settings):
    settings.elevenlabs.enabled = False
    settings.openai_transcription.enabled = False
    settings.microsoft.enabled = False

    assert registry.build_providers(settings) == {}


def test_a_service_that_is_on_but_not_filled_in_still_gets_an_adapter(settings):
    """The adapter is the thing that can say what is missing, so it is built."""
    settings.microsoft.api_key = ""
    settings.microsoft.endpoint = ""

    adapter = registry.build_provider(Provider.MICROSOFT, settings)

    assert adapter is not None
    assert adapter.is_configured() is False
    assert "API key" in (adapter.describe_configuration_problem() or "")


# -- Which services a full pass uses -------------------------------------


def test_a_full_pass_uses_the_three_services_that_transcribe_whole_recordings(settings):
    """AssemblyAI is the second opinion and Deepgram is optional.

    Sending every recording to all five would cost a great deal more for
    very little, so neither of those two is part of a full pass even when
    both are switched on and configured.
    """
    built = registry.build_providers(settings)

    assert set(built) == {Provider.ELEVENLABS, Provider.OPENAI, Provider.MICROSOFT}
    assert registry.DEFAULT_FULL_PASS_PROVIDERS == (
        Provider.ELEVENLABS,
        Provider.OPENAI,
        Provider.MICROSOFT,
    )


def test_a_caller_may_ask_for_a_different_set_of_services(settings):
    built = registry.build_providers(settings, (Provider.DEEPGRAM, Provider.ASSEMBLYAI))

    assert set(built) == {Provider.DEEPGRAM, Provider.ASSEMBLYAI}


# -- A library that is not installed -------------------------------------


def test_a_missing_client_library_loses_one_service_rather_than_raising(
    settings, monkeypatch
):
    """Every provider package is optional, and one of them will be missing.

    Putting ``None`` in ``sys.modules`` is how Python itself marks a module
    as unimportable, so an import of it raises exactly the ImportError a
    machine without the package would raise.
    """
    monkeypatch.setitem(
        sys.modules, "vox_verbatim.transcription.providers.elevenlabs", None
    )

    assert registry.build_provider(Provider.ELEVENLABS, settings) is None


def test_a_missing_client_library_does_not_take_the_other_services_with_it(
    settings, monkeypatch
):
    monkeypatch.setitem(
        sys.modules, "vox_verbatim.transcription.providers.elevenlabs", None
    )

    built = registry.build_providers(settings)

    assert set(built) == {Provider.OPENAI, Provider.MICROSOFT}


def test_a_missing_library_also_switches_off_forced_alignment(settings, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "vox_verbatim.transcription.providers.elevenlabs", None
    )

    assert registry.build_forced_aligner(settings) is None


def test_a_service_with_no_adapter_at_all_is_reported_rather_than_raised(
    settings, monkeypatch
):
    monkeypatch.delitem(registry._BUILDERS, Provider.DEEPGRAM)

    assert registry.build_provider(Provider.DEEPGRAM, settings) is None


# -- What the user typed reaching the adapter ----------------------------


def test_the_elevenlabs_settings_reach_its_adapter(settings, monkeypatch):
    settings.elevenlabs.transcription_model = "scribe-test"
    settings.elevenlabs.transcription_parameters = {"diarize": True}
    settings.elevenlabs.tag_audio_events = False
    settings.elevenlabs.diarise = False
    settings.processing.provider_timeout_seconds = 123.0
    settings.processing.provider_retry_attempts = 4
    record(monkeypatch, elevenlabs_module, "ElevenLabsProvider")

    built = registry.build_provider(Provider.ELEVENLABS, settings)

    assert built.arguments == {
        "api_key": "elevenlabs-key",
        "model": "scribe-test",
        "parameters": {"diarize": True},
        "tag_audio_events": False,
        "diarise": False,
        "timeout_seconds": 123.0,
        "maximum_retries": 4,
    }


def test_the_openai_settings_reach_its_adapter(settings, monkeypatch):
    settings.openai_transcription.model = "openai-test"
    settings.openai_transcription.parameters = {"temperature": 0}
    settings.openai_transcription.chunk_target_bytes = 5_000_000
    record(monkeypatch, openai_module, "OpenAiTranscriptionProvider")

    built = registry.build_provider(Provider.OPENAI, settings)

    assert built.arguments["api_key"] == "openai-key"
    assert built.arguments["model"] == "openai-test"
    assert built.arguments["parameters"] == {"temperature": 0}
    assert built.arguments["chunk_target_bytes"] == 5_000_000


def test_the_microsoft_settings_reach_its_adapter_including_its_own_endpoint(
    settings, monkeypatch
):
    """Microsoft is the only service whose address the user has to supply."""
    settings.microsoft.model = "mai-test"
    settings.microsoft.api_version = "2099-01-01"
    record(monkeypatch, microsoft_module, "MicrosoftProvider")

    built = registry.build_provider(Provider.MICROSOFT, settings)

    assert built.arguments["endpoint"] == "https://example.cognitiveservices.azure.com"
    assert built.arguments["model"] == "mai-test"
    assert built.arguments["api_version"] == "2099-01-01"


def test_both_assemblyai_model_names_reach_its_adapter(settings, monkeypatch):
    """The user manages two models; the service wants them in order.

    One model normally runs and a second, older one hears Afrikaans. That is
    how the user thinks about them and how Settings presents them, and this
    is the one place the two ideas are joined up.
    """
    settings.assemblyai.primary_model = "universal-3-5-pro"
    settings.assemblyai.afrikaans_model = "universal-2"
    record(monkeypatch, assemblyai_module, "AssemblyAiProvider")

    built = registry.build_provider(Provider.ASSEMBLYAI, settings)

    assert built.arguments["model"] == "universal-3-5-pro"
    assert built.arguments["afrikaans_model"] == "universal-2"


def test_the_deepgram_settings_reach_its_adapter(settings, monkeypatch):
    settings.deepgram.model = "deepgram-test"
    record(monkeypatch, deepgram_module, "DeepgramProvider")

    built = registry.build_provider(Provider.DEEPGRAM, settings)

    assert built.arguments["api_key"] == "deepgram-key"
    assert built.arguments["model"] == "deepgram-test"


def test_the_parameters_handed_over_are_a_copy_rather_than_the_settings_themselves(
    settings, monkeypatch
):
    """An adapter that edited its parameters must not edit Settings with them."""
    settings.elevenlabs.transcription_parameters = {"diarize": True}
    record(monkeypatch, elevenlabs_module, "ElevenLabsProvider")

    built = registry.build_provider(Provider.ELEVENLABS, settings)
    built.arguments["parameters"]["diarize"] = False

    assert settings.elevenlabs.transcription_parameters == {"diarize": True}


# -- The second opinion --------------------------------------------------


def test_the_second_opinion_comes_from_assemblyai(settings, monkeypatch):
    record(monkeypatch, assemblyai_module, "AssemblyAiProvider")

    built = registry.build_escalation_provider(settings)

    assert built.arguments["api_key"] == "assemblyai-key"


def test_switching_assemblyai_off_leaves_no_second_opinion(settings):
    settings.assemblyai.enabled = False

    assert registry.build_escalation_provider(settings) is None


# -- Measuring a corrected word again ------------------------------------


def test_the_forced_aligner_is_built_from_the_elevenlabs_settings(settings, monkeypatch):
    settings.elevenlabs.forced_alignment_parameters = {"granularity": "word"}
    settings.processing.provider_timeout_seconds = 90.0
    record(monkeypatch, elevenlabs_module, "ElevenLabsForcedAligner")

    built = registry.build_forced_aligner(settings)

    assert built.arguments["api_key"] == "elevenlabs-key"
    assert built.arguments["parameters"] == {"granularity": "word"}
    assert built.arguments["timeout_seconds"] == 90.0


def test_switching_forced_alignment_off_means_no_aligner_is_built(settings):
    settings.processing.forced_alignment_enabled = False

    assert registry.build_forced_aligner(settings) is None


def test_switching_elevenlabs_off_also_takes_the_forced_aligner_with_it(settings):
    """The aligner is the same service, so it cannot outlive it."""
    settings.elevenlabs.enabled = False

    assert registry.build_forced_aligner(settings) is None
