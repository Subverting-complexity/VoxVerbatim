"""Turning the user's settings into the services that will be called.

The adapters deliberately know nothing about the settings file. Each takes
its API key, its model name and its parameters as plain arguments, which is
what lets them be tested without a settings object and what stops a change
to the settings shape from rippling through five files. The cost of that
choice is that something has to join the two together, and this is it.

Keeping the join in one small module has a second benefit worth the file.
The names on each side genuinely differ, and the differences are not
mistakes. ElevenLabs spells its parameter ``diarize`` where the rest of this
application spells it ``diarise``. AssemblyAI takes an ordered list of
models where the user manages two separate settings. Microsoft needs an
endpoint that none of the others have. Every one of those translations
happens here, once, where it can be seen.
"""

from __future__ import annotations

import logging
from typing import Any

from audio_transcriber.settings import TranscriptionSettings
from audio_transcriber.transcription.model import Provider
from audio_transcriber.transcription.providers.base import (
    ForcedAligner,
    TranscriptionProvider,
)

_log = logging.getLogger(__name__)

#: The services run over the whole recording by default. AssemblyAI is
#: missing on purpose: it is the escalation provider, and a full pass over
#: every recording would cost far more than the specification wants for the
#: value it adds. Deepgram is missing because version 1 must not depend on
#: it.
DEFAULT_FULL_PASS_PROVIDERS: tuple[Provider, ...] = (
    Provider.ELEVENLABS,
    Provider.OPENAI,
    Provider.MICROSOFT,
)


def build_provider(
    provider: Provider,
    settings: TranscriptionSettings,
) -> TranscriptionProvider | None:
    """Return the adapter for one service, or None if it is switched off.

    A service that is switched off returns None rather than an adapter that
    refuses to run, so that the caller's list of services is the list it
    will actually call. A service that is switched on but not configured
    does return an adapter, because the adapter is the thing that can
    explain what is missing.

    The import is done here rather than at the top of the module so that a
    missing client library disables one service instead of stopping the
    application from starting. That is the same rule the adapters follow
    internally, applied one level up.
    """
    try:
        return _BUILDERS[provider](settings)
    except KeyError:
        _log.warning("There is no adapter for %s.", provider)
        return None
    except ImportError:
        # The adapter module itself could not be imported, which means a
        # client library is missing. One service is lost; the run goes on.
        _log.warning(
            "%s is unavailable because its library is not installed.",
            provider.display_name,
        )
        return None


def build_providers(
    settings: TranscriptionSettings,
    wanted: tuple[Provider, ...] = DEFAULT_FULL_PASS_PROVIDERS,
) -> dict[Provider, TranscriptionProvider]:
    """Return the adapters for every wanted service that is switched on."""
    built: dict[Provider, TranscriptionProvider] = {}
    for provider in wanted:
        adapter = build_provider(provider, settings)
        if adapter is not None:
            built[provider] = adapter
    return built


def build_escalation_provider(
    settings: TranscriptionSettings,
) -> TranscriptionProvider | None:
    """Return the service used for second opinions, or None if switched off."""
    return build_provider(Provider.ASSEMBLYAI, settings)


def build_forced_aligner(settings: TranscriptionSettings) -> ForcedAligner | None:
    """Return the forced aligner, or None if it is switched off or unavailable.

    This goes through the aligner interface rather than naming ElevenLabs
    anywhere above it, because the service that measures word boundaries
    today is not the one that will do it forever, and because this one
    cannot do Afrikaans at all.
    """
    if not settings.processing.forced_alignment_enabled:
        return None
    if not settings.elevenlabs.enabled:
        return None
    try:
        from audio_transcriber.transcription.providers.elevenlabs import (
            ElevenLabsForcedAligner,
        )
    except ImportError:
        _log.warning("Forced alignment is unavailable because its library is missing.")
        return None
    return ElevenLabsForcedAligner(
        api_key=settings.elevenlabs.api_key,
        parameters=dict(settings.elevenlabs.forced_alignment_parameters),
        timeout_seconds=settings.processing.provider_timeout_seconds,
        maximum_retries=settings.processing.provider_retry_attempts,
    )


# -- One builder per service ---------------------------------------------
#
# Each is small and dull on purpose. The interesting content is the
# translation between what the user manages and what the service expects,
# and each of those is commented where it happens.


def _build_elevenlabs(settings: TranscriptionSettings) -> TranscriptionProvider | None:
    if not settings.elevenlabs.enabled:
        return None
    from audio_transcriber.transcription.providers.elevenlabs import ElevenLabsProvider

    return ElevenLabsProvider(
        api_key=settings.elevenlabs.api_key,
        model=settings.elevenlabs.transcription_model,
        parameters=dict(settings.elevenlabs.transcription_parameters),
        tag_audio_events=settings.elevenlabs.tag_audio_events,
        timeout_seconds=settings.processing.provider_timeout_seconds,
        maximum_retries=settings.processing.provider_retry_attempts,
    )


def _build_openai(settings: TranscriptionSettings) -> TranscriptionProvider | None:
    if not settings.openai_transcription.enabled:
        return None
    from audio_transcriber.transcription.providers.openai import (
        OpenAiTranscriptionProvider,
    )

    return OpenAiTranscriptionProvider(
        api_key=settings.openai_transcription.api_key,
        model=settings.openai_transcription.model,
        parameters=dict(settings.openai_transcription.parameters),
        timeout_seconds=settings.processing.provider_timeout_seconds,
        maximum_retries=settings.processing.provider_retry_attempts,
    )


def _build_microsoft(settings: TranscriptionSettings) -> TranscriptionProvider | None:
    if not settings.microsoft.enabled:
        return None
    from audio_transcriber.transcription.providers.microsoft import MicrosoftProvider

    return MicrosoftProvider(
        api_key=settings.microsoft.api_key,
        endpoint=settings.microsoft.endpoint,
        model=settings.microsoft.model,
        parameters=dict(settings.microsoft.parameters),
        api_version=settings.microsoft.api_version,
        timeout_seconds=settings.processing.provider_timeout_seconds,
    )


def _build_assemblyai(settings: TranscriptionSettings) -> TranscriptionProvider | None:
    if not settings.assemblyai.enabled:
        return None
    from audio_transcriber.transcription.providers.assemblyai import AssemblyAiProvider

    # The user manages two model names because that is how they think about
    # them: the one that normally runs, and the one that can do Afrikaans.
    # AssemblyAI wants an ordered chain and falls through it by itself when
    # the first model cannot serve the language, which happens to be exactly
    # the behaviour those two settings describe.
    return AssemblyAiProvider(
        api_key=settings.assemblyai.api_key,
        model=settings.assemblyai.primary_model,
        afrikaans_model=settings.assemblyai.afrikaans_model,
        parameters=dict(settings.assemblyai.parameters),
        timeout_seconds=settings.processing.provider_timeout_seconds,
    )


def _build_deepgram(settings: TranscriptionSettings) -> TranscriptionProvider | None:
    if not settings.deepgram.enabled:
        return None
    from audio_transcriber.transcription.providers.deepgram import DeepgramProvider

    return DeepgramProvider(
        api_key=settings.deepgram.api_key,
        model=settings.deepgram.model,
        parameters=dict(settings.deepgram.parameters),
        timeout_seconds=settings.processing.provider_timeout_seconds,
    )


_BUILDERS: dict[Provider, Any] = {
    Provider.ELEVENLABS: _build_elevenlabs,
    Provider.OPENAI: _build_openai,
    Provider.MICROSOFT: _build_microsoft,
    Provider.ASSEMBLYAI: _build_assemblyai,
    Provider.DEEPGRAM: _build_deepgram,
}
