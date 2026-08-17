"""Tests for the transcription provider and processing settings.

Nothing here talks to a service or needs a key. These settings are plain
data, and the point of the tests is that they stay usable however badly the
settings file is mangled, and that a credential cannot escape into a
transcript.
"""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any

from audio_transcriber.settings import (
    DEFAULT_ASSEMBLYAI_AFRIKAANS_MODEL,
    DEFAULT_ASSEMBLYAI_PRIMARY_MODEL,
    DEFAULT_CHUNK_OVERLAP_SECONDS,
    DEFAULT_CHUNK_TARGET_BYTES,
    DEFAULT_DEEPGRAM_MODEL,
    DEFAULT_ELEVENLABS_TRANSCRIPTION_MODEL,
    DEFAULT_ESCALATION_CONTEXT_SECONDS_AFTER,
    DEFAULT_ESCALATION_CONTEXT_SECONDS_BEFORE,
    DEFAULT_ESCALATIONS_PER_RECORDING,
    DEFAULT_EXPECTED_SPEAKER_COUNT,
    DEFAULT_MICROSOFT_API_VERSION,
    DEFAULT_MICROSOFT_MODEL,
    DEFAULT_OPENAI_ADJUDICATION_MODEL,
    DEFAULT_OPENAI_TRANSCRIPTION_MODEL,
    DEFAULT_PROVIDER_RETRY_ATTEMPTS,
    DEFAULT_PROVIDER_RETRY_BACKOFF_SECONDS,
    DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    DEFAULT_TRANSCRIPT_FOLDER_SUFFIX,
    MAXIMUM_CHUNK_TARGET_BYTES,
    MAXIMUM_COST,
    MAXIMUM_ESCALATION_CONTEXT_SECONDS,
    MAXIMUM_ESCALATIONS_PER_RECORDING,
    MAXIMUM_EXPECTED_SPEAKER_COUNT,
    MAXIMUM_PROVIDER_RETRY_ATTEMPTS,
    MAXIMUM_PROVIDER_TIMEOUT_SECONDS,
    MINIMUM_CHUNK_TARGET_BYTES,
    MINIMUM_COST,
    MINIMUM_ESCALATION_CONTEXT_SECONDS,
    MINIMUM_ESCALATIONS_PER_RECORDING,
    MINIMUM_EXPECTED_SPEAKER_COUNT,
    MINIMUM_PROVIDER_RETRY_ATTEMPTS,
    MINIMUM_PROVIDER_TIMEOUT_SECONDS,
    SETTINGS_FORMAT_VERSION,
    AssemblyAiSettings,
    CostSettings,
    DeepgramSettings,
    ElevenLabsSettings,
    EnhanceSettings,
    MicrosoftMaiSettings,
    OpenAiAdjudicationSettings,
    OpenAiTranscriptionSettings,
    ProcessingSettings,
    Settings,
    SettingsStore,
    TranscriptionSettings,
)

# The redaction rule itself, reached for directly by one test below. That
# test has to redact a structure holding fields nobody has written yet, so it
# cannot go through real settings to get at the rule.
from audio_transcriber.settings import _without_secrets
from audio_transcriber.transcription.model import Provider

# -- Defaults ------------------------------------------------------------


def test_a_new_installation_knows_what_every_service_is_called():
    transcription = TranscriptionSettings()

    assert transcription.openai_transcription.model == DEFAULT_OPENAI_TRANSCRIPTION_MODEL
    assert transcription.openai_adjudication.model == DEFAULT_OPENAI_ADJUDICATION_MODEL
    assert transcription.elevenlabs.transcription_model == DEFAULT_ELEVENLABS_TRANSCRIPTION_MODEL
    assert transcription.microsoft.model == DEFAULT_MICROSOFT_MODEL
    assert transcription.microsoft.api_version == DEFAULT_MICROSOFT_API_VERSION
    assert transcription.assemblyai.primary_model == DEFAULT_ASSEMBLYAI_PRIMARY_MODEL
    assert transcription.assemblyai.afrikaans_model == DEFAULT_ASSEMBLYAI_AFRIKAANS_MODEL
    assert transcription.deepgram.model == DEFAULT_DEEPGRAM_MODEL


def test_a_new_installation_has_no_credentials_and_no_azure_address():
    transcription = TranscriptionSettings()

    assert transcription.openai_transcription.api_key == ""
    assert transcription.openai_adjudication.api_key == ""
    assert transcription.elevenlabs.api_key == ""
    assert transcription.microsoft.api_key == ""
    assert transcription.assemblyai.api_key == ""
    assert transcription.deepgram.api_key == ""
    # There is no address to guess: it names a resource in the user's own
    # Azure subscription.
    assert transcription.microsoft.endpoint == ""


def test_only_deepgram_starts_switched_off():
    """Deepgram is the optional challenger, so nobody pays for it unasked."""
    transcription = TranscriptionSettings()

    assert transcription.elevenlabs.enabled is True
    assert transcription.openai_transcription.enabled is True
    assert transcription.openai_adjudication.enabled is True
    assert transcription.microsoft.enabled is True
    assert transcription.assemblyai.enabled is True
    assert transcription.deepgram.enabled is False


def test_the_processing_defaults_are_the_ones_the_specification_asks_for():
    processing = ProcessingSettings()

    assert processing.default_afrikaans_enabled is False
    assert processing.default_expected_speaker_count == DEFAULT_EXPECTED_SPEAKER_COUNT == 1
    assert processing.escalation_context_seconds_before == 5.0
    assert processing.escalation_context_seconds_after == 5.0
    assert processing.provider_timeout_seconds == DEFAULT_PROVIDER_TIMEOUT_SECONDS
    assert processing.provider_retry_attempts == DEFAULT_PROVIDER_RETRY_ATTEMPTS
    assert processing.provider_retry_backoff_seconds == DEFAULT_PROVIDER_RETRY_BACKOFF_SECONDS
    assert processing.provider_chunk_overlap_seconds == DEFAULT_CHUNK_OVERLAP_SECONDS == 2.0
    assert processing.forced_alignment_enabled is True
    assert processing.escalation_enabled is True
    assert processing.adjudication_enabled is True
    assert processing.maximum_escalations_per_recording == DEFAULT_ESCALATIONS_PER_RECORDING
    assert processing.transcript_folder_suffix == DEFAULT_TRANSCRIPT_FOLDER_SUFFIX == ".transcript"


def test_openai_chunks_are_aimed_well_below_the_limit_openai_enforces():
    """The limit is 25 MB per request, and a chunk that lands over it fails."""
    assert OpenAiTranscriptionSettings().chunk_target_bytes == DEFAULT_CHUNK_TARGET_BYTES
    assert DEFAULT_CHUNK_TARGET_BYTES <= 20_000_000
    assert MAXIMUM_CHUNK_TARGET_BYTES <= 25_000_000


def test_the_run_is_confirmed_before_any_money_is_spent():
    assert CostSettings().confirm_before_running is True


def test_every_service_has_a_rate_under_the_name_the_rest_of_the_code_uses():
    cost = CostSettings()

    for provider in Provider:
        assert cost.per_minute_for(provider) > 0.0
    assert cost.adjudication_per_request > 0.0


def test_the_one_rate_that_was_checked_is_the_published_one():
    """OpenAI's price for ``gpt-transcribe``, verified on 17 August 2026.

    The others are estimates and are labelled as such. A user is far more
    likely to check a figure that sits beside a verified one and admits to
    being unchecked.
    """
    assert CostSettings().openai_transcription_per_minute == 0.0045
    assert CostSettings().per_minute_for(Provider.OPENAI) == 0.0045


def test_every_range_holds_its_own_default():
    """The dialog sets its spin box ranges from these, so a default outside
    one of them would be a value the user could never type back in."""
    ranges = (
        (MINIMUM_CHUNK_TARGET_BYTES, DEFAULT_CHUNK_TARGET_BYTES, MAXIMUM_CHUNK_TARGET_BYTES),
        (
            MINIMUM_EXPECTED_SPEAKER_COUNT,
            DEFAULT_EXPECTED_SPEAKER_COUNT,
            MAXIMUM_EXPECTED_SPEAKER_COUNT,
        ),
        (
            MINIMUM_ESCALATION_CONTEXT_SECONDS,
            DEFAULT_ESCALATION_CONTEXT_SECONDS_BEFORE,
            MAXIMUM_ESCALATION_CONTEXT_SECONDS,
        ),
        (
            MINIMUM_ESCALATION_CONTEXT_SECONDS,
            DEFAULT_ESCALATION_CONTEXT_SECONDS_AFTER,
            MAXIMUM_ESCALATION_CONTEXT_SECONDS,
        ),
        (
            MINIMUM_PROVIDER_TIMEOUT_SECONDS,
            DEFAULT_PROVIDER_TIMEOUT_SECONDS,
            MAXIMUM_PROVIDER_TIMEOUT_SECONDS,
        ),
        (
            MINIMUM_PROVIDER_RETRY_ATTEMPTS,
            DEFAULT_PROVIDER_RETRY_ATTEMPTS,
            MAXIMUM_PROVIDER_RETRY_ATTEMPTS,
        ),
        (
            MINIMUM_ESCALATIONS_PER_RECORDING,
            DEFAULT_ESCALATIONS_PER_RECORDING,
            MAXIMUM_ESCALATIONS_PER_RECORDING,
        ),
    )
    for lowest, default, highest in ranges:
        assert lowest <= default <= highest


# -- Whether a service can be called at all ------------------------------


def test_a_service_without_a_key_is_not_configured():
    assert OpenAiTranscriptionSettings().is_configured is False
    assert OpenAiTranscriptionSettings(api_key="sk-test").is_configured is True

    assert ElevenLabsSettings(api_key="  ").is_configured is False
    assert ElevenLabsSettings(api_key="el-test").is_configured is True

    assert AssemblyAiSettings(api_key="aai-test").is_configured is True
    assert DeepgramSettings(api_key="dg-test").is_configured is True
    assert OpenAiAdjudicationSettings(api_key="sk-test").is_configured is True


def test_microsoft_needs_an_address_as_well_as_a_key():
    """The key alone has nowhere to go: the endpoint is the user's own resource."""
    assert MicrosoftMaiSettings(api_key="az-test").is_configured is False
    assert MicrosoftMaiSettings(endpoint="https://example.example").is_configured is False

    complete = MicrosoftMaiSettings(api_key="az-test", endpoint="https://example.example")

    assert complete.is_configured is True


def test_a_service_without_a_model_name_is_not_configured():
    assert OpenAiTranscriptionSettings(api_key="sk-test", model="").is_configured is False


def test_the_missing_pieces_are_named_one_by_one():
    transcription = TranscriptionSettings()
    transcription.elevenlabs.api_key = "el-test"
    transcription.openai_transcription.api_key = "sk-test"
    transcription.openai_adjudication.api_key = "sk-test"
    transcription.microsoft.enabled = False
    transcription.assemblyai.enabled = False

    assert transcription.missing_requirements() == []

    transcription.microsoft.enabled = True
    problems = transcription.missing_requirements()

    assert len(problems) == 1
    assert "Microsoft MAI" in problems[0]
    assert "endpoint" in problems[0]


def test_a_required_service_that_is_switched_off_stops_a_run():
    transcription = TranscriptionSettings()
    transcription.elevenlabs.api_key = "el-test"
    transcription.elevenlabs.enabled = False
    transcription.openai_transcription.api_key = "sk-test"
    transcription.openai_adjudication.api_key = "sk-test"
    transcription.microsoft.enabled = False
    transcription.assemblyai.enabled = False

    problems = transcription.missing_requirements()

    assert len(problems) == 1
    assert "ElevenLabs Scribe is switched off" in problems[0]


def test_a_service_nobody_asked_for_is_not_a_problem():
    """Deepgram is off and unconfigured, and that is a normal installation."""
    transcription = TranscriptionSettings()
    transcription.elevenlabs.api_key = "el-test"
    transcription.openai_transcription.api_key = "sk-test"
    transcription.openai_adjudication.api_key = "sk-test"
    transcription.microsoft.enabled = False
    transcription.assemblyai.enabled = False

    assert transcription.missing_requirements() == []


# -- The round trip to disk and back -------------------------------------


def _fully_populated() -> TranscriptionSettings:
    """Settings with every field moved off its default, for round trips."""
    return TranscriptionSettings(
        openai_transcription=OpenAiTranscriptionSettings(
            api_key="sk-transcribe",
            model="gpt-transcribe-next",
            parameters={"prompt": "Board meeting", "temperature": 0.0},
            chunk_target_bytes=12_000_000,
            enabled=False,
        ),
        openai_adjudication=OpenAiAdjudicationSettings(
            api_key="sk-adjudicate",
            model="a-model-announced-this-morning",
            reasoning_effort="high",
            parameters={"max_output_tokens": 4096},
            enabled=False,
        ),
        elevenlabs=ElevenLabsSettings(
            api_key="el-key",
            transcription_model="scribe_v3",
            transcription_parameters={"keyterms": ["Jürgen Müller"], "language_code": "eng"},
            forced_alignment_parameters={"timeout": 30},
            diarise=False,
            tag_audio_events=False,
            enabled=False,
        ),
        microsoft=MicrosoftMaiSettings(
            api_key="az-key",
            endpoint="https://example.cognitiveservices.azure.com",
            model="mai-transcribe-2",
            api_version="2026-01-01",
            parameters={"enhancedMode": {"model": "mai-transcribe-2"}},
            enabled=False,
        ),
        assemblyai=AssemblyAiSettings(
            api_key="aai-key",
            primary_model="universal-4",
            afrikaans_model="universal-3",
            parameters={"speech_models": ["universal-4", "universal-2"]},
            enabled=False,
        ),
        deepgram=DeepgramSettings(
            api_key="dg-key",
            model="nova-4",
            parameters={"smart_format": True},
            enabled=True,
        ),
        processing=ProcessingSettings(
            default_afrikaans_enabled=True,
            default_expected_speaker_count=4,
            escalation_context_seconds_before=8.5,
            escalation_context_seconds_after=3.25,
            provider_timeout_seconds=1200.0,
            provider_retry_attempts=5,
            provider_retry_backoff_seconds=10.0,
            provider_chunk_overlap_seconds=4.0,
            forced_alignment_enabled=False,
            escalation_enabled=False,
            adjudication_enabled=False,
            maximum_escalations_per_recording=25,
            transcript_folder_suffix=".transcripts",
        ),
        cost=CostSettings(
            elevenlabs_per_minute=0.01,
            openai_transcription_per_minute=0.02,
            microsoft_per_minute=0.03,
            assemblyai_per_minute=0.04,
            deepgram_per_minute=0.05,
            adjudication_per_request=0.5,
            confirm_before_running=False,
        ),
    )


def test_everything_survives_being_written_and_read_back(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    settings = Settings(transcription=_fully_populated())

    assert store.save(settings)

    assert store.load() == settings


def test_everything_survives_a_trip_through_a_plain_dictionary():
    settings = Settings(transcription=_fully_populated())

    assert Settings.from_dict(settings.to_dict()) == settings


def test_the_saved_file_is_stamped_with_the_format_version(tmp_path):
    path = tmp_path / "settings.json"
    SettingsStore(path).save(Settings())

    assert json.loads(path.read_text(encoding="utf-8"))["version"] == SETTINGS_FORMAT_VERSION


# -- Backward compatibility ----------------------------------------------


def test_a_file_written_by_the_current_application_still_loads(tmp_path):
    """The settings this application writes today have no transcription
    section at all, and must keep working with every new field defaulted."""
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "version": SETTINGS_FORMAT_VERSION,
                "reopen_last_folder": False,
                "short_skip_seconds": 10,
                "medium_skip_seconds": 60,
                "long_skip_seconds": 600,
                "enhance": {
                    "output_folder": r"D:\Enhanced",
                    "target_lufs": -16.0,
                    "ceiling_dbtp": -2.0,
                    "maximum_gain_db": 18.0,
                    "use_limiter": True,
                    "output_format": "flac",
                    "replace_existing": True,
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = SettingsStore(path).load()

    assert loaded.reopen_last_folder is False
    assert loaded.skip_seconds == (10, 60, 600)
    assert loaded.enhance.output_format == "flac"
    assert loaded.transcription == TranscriptionSettings()
    assert loaded.transcription.openai_transcription.model == DEFAULT_OPENAI_TRANSCRIPTION_MODEL


def test_a_file_holding_only_some_of_the_new_settings_keeps_the_rest(tmp_path):
    """Settings arrive one release at a time, and a file written between two
    of them has some sections and not others."""
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"transcription": {"elevenlabs": {"api_key": "el-key"}}}),
        encoding="utf-8",
    )

    loaded = SettingsStore(path).load().transcription

    assert loaded.elevenlabs.api_key == "el-key"
    assert loaded.elevenlabs.transcription_model == DEFAULT_ELEVENLABS_TRANSCRIPTION_MODEL
    assert loaded.elevenlabs.diarise is True
    assert loaded.deepgram == DeepgramSettings()
    assert loaded.processing == ProcessingSettings()


def test_the_older_settings_are_untouched_by_the_new_ones(tmp_path):
    """Adding transcription must not disturb what was already there."""
    store = SettingsStore(tmp_path / "settings.json")
    settings = Settings(short_skip_seconds=7, enhance=EnhanceSettings(use_limiter=True))

    assert store.save(settings)
    loaded = store.load()

    assert loaded.short_skip_seconds == 7
    assert loaded.enhance.use_limiter is True


# -- Falling back when the file has been mangled -------------------------


def test_a_section_that_is_not_a_section_is_ignored(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"transcription": {"elevenlabs": "el-key", "deepgram": [1, 2, 3]}}),
        encoding="utf-8",
    )

    loaded = SettingsStore(path).load().transcription

    assert loaded == TranscriptionSettings()


def test_transcription_that_is_not_a_section_at_all_is_ignored(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"transcription": "yes please"}), encoding="utf-8")

    assert SettingsStore(path).load().transcription == TranscriptionSettings()


def test_an_empty_model_name_falls_back_rather_than_being_kept():
    """A request sent with no model name fails obscurely; the default does not."""
    loaded = OpenAiTranscriptionSettings.from_dict({"model": "   "})

    assert loaded.model == DEFAULT_OPENAI_TRANSCRIPTION_MODEL


def test_a_model_name_that_is_not_text_falls_back():
    assert DeepgramSettings.from_dict({"model": 3}).model == DEFAULT_DEEPGRAM_MODEL
    assert AssemblyAiSettings.from_dict({"primary_model": None}).primary_model == (
        DEFAULT_ASSEMBLYAI_PRIMARY_MODEL
    )
    assert MicrosoftMaiSettings.from_dict({"api_version": []}).api_version == (
        DEFAULT_MICROSOFT_API_VERSION
    )


def test_a_pasted_key_keeps_none_of_the_whitespace_around_it():
    loaded = ElevenLabsSettings.from_dict({"api_key": "  el-key\n"})

    assert loaded.api_key == "el-key"


def test_a_key_that_is_not_text_becomes_no_key_at_all():
    assert MicrosoftMaiSettings.from_dict({"api_key": 12345, "endpoint": None}).api_key == ""
    assert MicrosoftMaiSettings.from_dict({"endpoint": 12345}).endpoint == ""


def test_an_empty_reasoning_effort_is_kept_because_it_means_do_not_send_it():
    """Models differ in which parameters they accept, and clearing this is
    how the user says the parameter must be left out."""
    assert OpenAiAdjudicationSettings.from_dict({"reasoning_effort": ""}).reasoning_effort == ""
    assert OpenAiAdjudicationSettings.from_dict({}).reasoning_effort == "medium"
    assert OpenAiAdjudicationSettings.from_dict({"reasoning_effort": 7}).reasoning_effort == (
        "medium"
    )


def test_a_switch_that_is_not_a_switch_falls_back():
    assert DeepgramSettings.from_dict({"enabled": "yes"}).enabled is False
    assert ElevenLabsSettings.from_dict({"diarise": 1}).diarise is True
    assert ElevenLabsSettings.from_dict({"tag_audio_events": None}).tag_audio_events is True
    assert ProcessingSettings.from_dict({"forced_alignment_enabled": "on"}).forced_alignment_enabled


def test_a_chunk_size_over_the_openai_limit_falls_back():
    assert OpenAiTranscriptionSettings.from_dict({"chunk_target_bytes": 30_000_000}) == (
        OpenAiTranscriptionSettings()
    )
    assert OpenAiTranscriptionSettings.from_dict({"chunk_target_bytes": 0}).chunk_target_bytes == (
        DEFAULT_CHUNK_TARGET_BYTES
    )
    assert OpenAiTranscriptionSettings.from_dict(
        {"chunk_target_bytes": 5.5}
    ).chunk_target_bytes == DEFAULT_CHUNK_TARGET_BYTES


def test_a_count_of_true_is_not_a_count_of_one():
    """True counts as 1 in Python and would slip through a plain number check."""
    assert ProcessingSettings.from_dict({"default_expected_speaker_count": True}) == (
        ProcessingSettings()
    )
    assert OpenAiTranscriptionSettings.from_dict({"chunk_target_bytes": True}) == (
        OpenAiTranscriptionSettings()
    )
    assert CostSettings.from_dict({"elevenlabs_per_minute": True}) == CostSettings()


def test_every_processing_number_is_held_inside_its_range():
    mangled = {
        "default_expected_speaker_count": 0,
        "escalation_context_seconds_before": -1.0,
        "escalation_context_seconds_after": 10_000.0,
        "provider_timeout_seconds": 0.5,
        "provider_retry_attempts": 99,
        "provider_retry_backoff_seconds": -3.0,
        "provider_chunk_overlap_seconds": 600.0,
        "maximum_escalations_per_recording": -1,
    }

    assert ProcessingSettings.from_dict(mangled) == ProcessingSettings()


def test_a_processing_number_that_is_text_falls_back():
    loaded = ProcessingSettings.from_dict(
        {"provider_timeout_seconds": "quite a while", "provider_retry_attempts": "twice"}
    )

    assert loaded == ProcessingSettings()


def test_one_bad_processing_value_does_not_throw_away_the_good_ones():
    loaded = ProcessingSettings.from_dict(
        {"provider_retry_attempts": 99, "provider_chunk_overlap_seconds": 3.0}
    )

    assert loaded.provider_retry_attempts == DEFAULT_PROVIDER_RETRY_ATTEMPTS
    assert loaded.provider_chunk_overlap_seconds == 3.0


def test_a_cost_outside_what_anybody_charges_falls_back():
    loaded = CostSettings.from_dict(
        {"elevenlabs_per_minute": MAXIMUM_COST + 1.0, "deepgram_per_minute": MINIMUM_COST - 1.0}
    )

    assert loaded == CostSettings()


def test_a_free_service_is_allowed():
    """Zero is a real rate: a user may have a plan that does not meter."""
    assert CostSettings.from_dict({"deepgram_per_minute": 0.0}).deepgram_per_minute == 0.0


# -- Free-form request parameters ----------------------------------------


def test_parameters_survive_a_json_round_trip(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    parameters = {
        "prompt": "Names: Jürgen Müller, Suzanne",
        "temperature": 0.0,
        "nested": {"list": [1, 2, 3], "flag": True, "nothing": None},
    }
    settings = Settings(
        transcription=TranscriptionSettings(
            openai_transcription=OpenAiTranscriptionSettings(parameters=parameters)
        )
    )

    assert store.save(settings)

    assert store.load().transcription.openai_transcription.parameters == parameters


def test_parameters_that_are_not_an_object_are_refused():
    for rubbish in ([1, 2, 3], "temperature=0", 7, None, True):
        assert OpenAiTranscriptionSettings.from_dict({"parameters": rubbish}).parameters == {}


def test_parameters_with_names_that_are_not_names_are_refused():
    """JSON has no way to write a number as a field name, so a dictionary
    with one would come back looking different from how it went in."""
    assert OpenAiTranscriptionSettings.from_dict({"parameters": {1: "one"}}).parameters == {}


def test_parameters_holding_something_json_cannot_write_are_refused():
    unwritable = OpenAiTranscriptionSettings.from_dict({"parameters": {"when": object()}})
    assert unwritable.parameters == {}
    # NaN is worse than unwritable: Python writes it happily and produces a
    # file that no other reader accepts as JSON.
    not_a_number = OpenAiTranscriptionSettings.from_dict({"parameters": {"temperature": math.nan}})
    assert not_a_number.parameters == {}


def test_unusable_parameters_are_thrown_away_whole():
    """Half a set of parameters is a request nobody meant to make."""
    loaded = OpenAiTranscriptionSettings.from_dict(
        {"model": "gpt-transcribe-next", "parameters": {"good": 1, "bad": object()}}
    )

    assert loaded.parameters == {}
    assert loaded.model == "gpt-transcribe-next"


def test_both_of_the_elevenlabs_parameter_sets_are_checked():
    loaded = ElevenLabsSettings.from_dict(
        {"transcription_parameters": {"keyterms": ["Müller"]}, "forced_alignment_parameters": 5}
    )

    assert loaded.transcription_parameters == {"keyterms": ["Müller"]}
    assert loaded.forced_alignment_parameters == {}


# -- Where the transcript folder is written ------------------------------


def test_a_folder_suffix_that_would_escape_the_recording_folder_falls_back():
    """The suffix decides where transcripts land, so a separator in it would
    quietly write them somewhere else entirely."""
    for escape in (r".trans\cript", ".trans/cript", "..", ".", "../elsewhere", ""):
        loaded = ProcessingSettings.from_dict({"transcript_folder_suffix": escape})
        assert loaded.transcript_folder_suffix == DEFAULT_TRANSCRIPT_FOLDER_SUFFIX


def test_a_folder_suffix_windows_would_refuse_falls_back():
    for illegal in ('.trans"cript', ".trans?cript", ".trans:cript", ".trans*", ".trans|cript"):
        loaded = ProcessingSettings.from_dict({"transcript_folder_suffix": illegal})
        assert loaded.transcript_folder_suffix == DEFAULT_TRANSCRIPT_FOLDER_SUFFIX


def test_a_sensible_folder_suffix_is_kept():
    loaded = ProcessingSettings.from_dict({"transcript_folder_suffix": " .transcripts "})

    assert loaded.transcript_folder_suffix == ".transcripts"


# -- Keeping credentials out of provenance -------------------------------


def _secret_fields() -> list[tuple[Any, str]]:
    """Every credential field on every section, found rather than listed.

    Written this way on purpose. A list of the fields to check would be
    complete on the day it was written and quietly incomplete the first time
    somebody adds a provider, which is exactly the failure the redaction
    rule exists to prevent, so the test must not repeat it.

    The net cast here is wider than the redaction rule, and intentionally
    so. The rule matches the end of a field name; this matches anywhere for
    the three nouns that mean nothing else. A field called ``secret_value``
    would therefore be found here and missed by the rule, and the leak test
    would fail rather than the credential escaping. The one noun left out of
    the wide sweep is "key", because ``keyterms`` is a real and legitimate
    field name that has nothing to do with credentials.
    """
    found: list[tuple[Any, str]] = []
    transcription = TranscriptionSettings()
    for section_field in dataclasses.fields(transcription):
        section = getattr(transcription, section_field.name)
        for inner in dataclasses.fields(section):
            if not isinstance(getattr(section, inner.name), str):
                continue
            looks_like_a_credential = inner.name.endswith("key") or any(
                part in inner.name for part in ("secret", "token", "credential")
            )
            if looks_like_a_credential:
                found.append((section, inner.name))
    return found


def test_every_credential_field_is_found_by_the_test_itself():
    """A guard on the guard: if this finds nothing, the leak test proves nothing."""
    names = {name for _section, name in _secret_fields()}

    assert names == {"api_key"}
    assert len(_secret_fields()) == 6


def test_no_credential_can_reach_provenance():
    transcription = TranscriptionSettings()
    secrets = []
    for index, (section, name) in enumerate(_secret_fields()):
        secret = f"never-write-this-{index}"
        setattr(section, name, secret)
        secrets.append(secret)

    written = json.dumps(transcription.for_provenance())

    assert secrets
    for secret in secrets:
        assert secret not in written


def test_a_credential_hidden_in_free_form_parameters_is_stripped_too():
    """We have no idea what a user puts in a parameter dictionary, and a key
    is as easy to put there as beside a model name."""
    transcription = TranscriptionSettings()
    transcription.deepgram.parameters = {
        "api_key": "never-write-this",
        "nested": {"access_token": "nor-this", "smart_format": True},
        "list": [{"secret": "nor-this-either"}],
    }

    written = json.dumps(transcription.for_provenance())

    assert "never-write-this" not in written
    assert "nor-this" not in written
    assert "nor-this-either" not in written
    assert "smart_format" in written


def test_keyterms_survive_redaction_although_a_key_beside_them_does_not():
    """The boundary the redaction rule turns on, pinned in one assertion.

    Keyterms are the names and specialised vocabulary a service was primed
    with, and the specification requires them in the record of every
    request. They begin with the same three letters as a credential and are
    not one. A rule that matched a name anywhere rather than at its end
    would take them out along with the key, and nothing would say so: the
    transcript would simply stop recording what the service was told, which
    is the harder kind of mistake to notice because it looks like caution.
    """
    transcription = TranscriptionSettings()
    transcription.elevenlabs.transcription_parameters = {
        "keyterms": ["Jürgen Müller", "Subverting Complexity"],
        "keyterm_boost": 1.5,
        "api_key": "never-write-this",
    }

    written = transcription.for_provenance()
    parameters = written["elevenlabs"]["transcription_parameters"]

    assert parameters["keyterms"] == ["Jürgen Müller", "Subverting Complexity"]
    assert parameters["keyterm_boost"] == 1.5
    assert "api_key" not in parameters
    assert "never-write-this" not in json.dumps(written)


def test_a_credential_added_later_is_stripped_without_anybody_remembering():
    """The redaction works by the name of the field, so a field nobody has
    written yet is already covered."""
    transcription = TranscriptionSettings()
    invented = {
        "openai_transcription": {
            "organisation_api_key": "leak",
            "api_key_for_billing": "leak",
            "subscription_key": "leak",
            "refresh_token": "leak",
            "client_secret": "leak",
            "azure_credential": "leak",
            "password": "leak",
            "model": "gpt-transcribe-next",
        }
    }

    written = json.dumps(_without_secrets(invented))

    assert "leak" not in written
    assert "gpt-transcribe-next" in written
    # And the same rule is the one provenance itself uses.
    assert "api_key" not in json.dumps(transcription.for_provenance())


def test_provenance_still_says_what_the_run_actually_used():
    """Redaction must not empty the record: the point of it is explaining a
    transcript afterwards."""
    transcription = _fully_populated()

    written = transcription.for_provenance()

    assert written["openai_transcription"]["model"] == "gpt-transcribe-next"
    assert written["openai_adjudication"]["reasoning_effort"] == "high"
    assert written["elevenlabs"]["transcription_model"] == "scribe_v3"
    assert written["microsoft"]["endpoint"] == "https://example.cognitiveservices.azure.com"
    assert written["assemblyai"]["afrikaans_model"] == "universal-3"
    assert written["processing"]["default_afrikaans_enabled"] is True
    assert written["processing"]["provider_chunk_overlap_seconds"] == 4.0
    assert written["cost"]["adjudication_per_request"] == 0.5


def test_provenance_is_a_copy_rather_than_the_settings_themselves():
    """It is handed to the transcript, which outlives the settings object."""
    transcription = TranscriptionSettings()

    written = transcription.for_provenance()
    written["processing"]["forced_alignment_enabled"] = False

    assert transcription.processing.forced_alignment_enabled is True


def test_provenance_can_be_written_as_json():
    """It ends up inside a transcript file, so it has to be writable as it is."""
    written = json.dumps(_fully_populated().for_provenance())

    assert json.loads(written)["deepgram"]["model"] == "nova-4"
