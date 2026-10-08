"""The settings the user chooses, and how they are kept.

These are preferences: things the user decides once and expects to hold
until they decide otherwise. They are deliberately separate from the
working session, which is the state of what they happened to be doing at
the time. Deleting the session loses your place; deleting the settings
loses your preferences.

Transcription brings a second kind of preference with it. Every speech
service the application talks to needs credentials, a model name, an
endpoint or a handful of request parameters, and all of those change on
somebody else's schedule rather than ours. A model name that lives in the
source code has to be released to change; a model name that lives here can
be typed in the morning it appears. So everything a provider needs is a
setting, and nothing about a provider is hidden in the code.

That makes this the one file in the application that holds secrets, which
brings one hard rule with it. API keys are written to the settings file
and are read from it by the adapters, and they go nowhere else. In
particular they never reach transcript provenance, which is why
:meth:`TranscriptionSettings.for_provenance` exists and why it redacts by
rule rather than by a list of fields to remove.

Nothing here depends on Qt, so it can be tested on its own.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar

from vox_verbatim.audio.enhance import (
    DEFAULT_CEILING_DBTP,
    DEFAULT_MAXIMUM_GAIN_DB,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_TARGET_LUFS,
    MAXIMUM_CEILING_DBTP,
    MAXIMUM_MAXIMUM_GAIN_DB,
    MAXIMUM_TARGET_LUFS,
    MINIMUM_CEILING_DBTP,
    MINIMUM_MAXIMUM_GAIN_DB,
    MINIMUM_TARGET_LUFS,
    output_format_for,
)
from vox_verbatim.json_store import read_json_object, write_json_object
from vox_verbatim.transcription.model import Provider

SETTINGS_FILE_NAME = "settings.json"

#: Bumped only if the shape on disk changes in a way that needs migrating.
#: Version 2 moved files that saved the old default adjudication model on
#: to the new one; see ``_migrate_adjudication_model``.
SETTINGS_FORMAT_VERSION = 2

#: The default adjudication model before version 2 of the file.
_RETIRED_ADJUDICATION_MODEL = "gpt-5.6"

#: Efforts GPT-6.1 Sol refuses. A file that saved one of these beside the
#: old model would fail every request once the model is moved on.
_EFFORTS_THE_NEW_MODEL_REFUSES = ("none", "minimal")

#: A skip of less than a second is no use, and more than an hour is beyond
#: what these buttons are for.
MINIMUM_SKIP_SECONDS = 1
MAXIMUM_SKIP_SECONDS = 3600


# -- What each transcription service is called today ---------------------
#
# These are starting points, not the truth. Every one of them is editable
# text in Settings, because the services rename and replace their models
# far more often than this application is released, and a user who has to
# wait for a new build to type in a new model name is a user who cannot
# use the model they are paying for.

#: The latest OpenAI ``gpt-transcribe`` model. The specification is
#: emphatic that this family is the only one the transcription stage may
#: use: no Whisper, and no older GPT-4o transcription model, even to gain a
#: feature such as word timing. Timing comes from ElevenLabs instead.
DEFAULT_OPENAI_TRANSCRIPTION_MODEL = "gpt-transcribe"

#: The OpenAI model that settles disputes the deterministic rules could
#: not. It is a different model from the one above and is chosen freely,
#: because adjudication is a reasoning task rather than a listening one.
#: This is the model family that moves fastest of everything named here,
#: which is exactly why the setting is free text.
DEFAULT_OPENAI_ADJUDICATION_MODEL = "gpt-6.1-sol"

#: How hard the adjudicating model should think. Which words a model
#: accepts here is its own business, and they have changed more than once,
#: so this is free text rather than a list. Empty is meaningful: it means
#: the parameter is left out of the request altogether, which is what a
#: model that does not accept it at all needs. Low is the least effort
#: GPT-6.1 Sol accepts: it takes neither none nor minimal.
DEFAULT_OPENAI_REASONING_EFFORT = "low"

#: The OpenAI model that edits a finished transcript into the smooth copy,
#: and how hard it thinks. Free text, for the same reason as the
#: adjudication model above.
DEFAULT_SMOOTHING_MODEL = "gpt-6.1-sol"
DEFAULT_SMOOTHING_REASONING_EFFORT = "low"

#: The style prompt the smooth transcript is made with. A person may edit
#: it in Settings. It describes only the style: the format the application
#: depends on is added by the smoothing stage itself and cannot be edited.
DEFAULT_SMOOTHING_PROMPT = """\
You are editing the transcript of a recorded conversation into clean text that is easy to read. The transcript is literal. It contains every filler, stutter, repeat and false start.

Do:
- Remove fillers such as "um", "uh", "er" and "ah", and remove "you know" or "like" when they carry no meaning.
- Remove stutters, repeated words and false starts. "We, we, we were" becomes "We were". Remove a sentence the speaker started and dropped, unless it tells the reader something.
- Fix punctuation and capital letters. Join pieces of one sentence that a pause broke apart.
- Keep the speaker's own words, word order, dialect and way of speaking. Change grammar only where a reader would otherwise stumble.
- Leave out a turn that is only a short reply such as "Mm-hmm", "Yeah" or "Okay" and adds nothing. Keep it when it answers a question.
- Keep words in Afrikaans, German or any other language as they were spoken, and put an English translation in square brackets after them, for example: "Ek wil, ek gaan, ek sal, ek kan [I want to, I am going to, I will, I can]".
- Keep every marker such as [UNCERTAIN: board / place] exactly as it is. Never choose between its alternatives.
- Keep sounds such as [laughs] where they show the tone.

Do not:
- Add, guess or explain anything the speaker did not say.
- Summarise, shorten a story or change the order of what was said.
- Change names, places, dates, numbers or amounts.
- Make the speaker sound more formal than they are.
"""

#: ElevenLabs Scribe v2, the structural backbone: word timings, speaker
#: labels and log probabilities all come from it.
DEFAULT_ELEVENLABS_TRANSCRIPTION_MODEL = "scribe_v2"

#: Microsoft MAI-Transcribe-1.5, reached through Azure AI Foundry. The
#: model is named in the request body rather than in the URL, so the
#: endpoint is the resource address and the model is a separate setting.
DEFAULT_MICROSOFT_MODEL = "mai-transcribe-1.5"
DEFAULT_MICROSOFT_API_VERSION = "2025-10-15"

#: AssemblyAI's two models. The first is the better one and covers fewer
#: languages; the second is the fallback, and is the one that can hear
#: Afrikaans at all, which is why it is named separately rather than being
#: buried in a parameter list.
DEFAULT_ASSEMBLYAI_PRIMARY_MODEL = "universal-3-5-pro"
DEFAULT_ASSEMBLYAI_AFRIKAANS_MODEL = "universal-2"

#: Deepgram Nova-3, the optional challenger. Nothing depends on it.
DEFAULT_DEEPGRAM_MODEL = "nova-3"


# -- Limits on the numbers the user can choose ---------------------------
#
# Each pair is exported so that the Settings dialog can set its spin box
# ranges from them. A dialog that repeats the numbers is a dialog that will
# one day disagree with the validation behind it, and the user would then
# be able to enter a value that is silently thrown away on the next load.

#: OpenAI accepts 25 MB in one request. Chunks are aimed well under that
#: rather than at it, because the size of an encoded chunk cannot be known
#: exactly until it has been encoded, and a chunk that comes out slightly
#: over the limit is a failed request rather than a slightly large one.
MINIMUM_CHUNK_TARGET_BYTES = 1_000_000
MAXIMUM_CHUNK_TARGET_BYTES = 25_000_000
DEFAULT_CHUNK_TARGET_BYTES = 20_000_000

#: How many people the recording is expected to contain. One is the common
#: case and the honest default; the ceiling is what the diarisation
#: services themselves will entertain.
MINIMUM_EXPECTED_SPEAKER_COUNT = 1
MAXIMUM_EXPECTED_SPEAKER_COUNT = 32
DEFAULT_EXPECTED_SPEAKER_COUNT = 1

#: How much audio surrounds a disputed word when it is sent for a second
#: opinion. Too little and the service has no sentence to work with and
#: does worse than it would on the whole recording; too much and the second
#: opinion is no longer about the disputed word.
MINIMUM_ESCALATION_CONTEXT_SECONDS = 0.0
MAXIMUM_ESCALATION_CONTEXT_SECONDS = 60.0
DEFAULT_ESCALATION_CONTEXT_SECONDS_BEFORE = 5.0
DEFAULT_ESCALATION_CONTEXT_SECONDS_AFTER = 5.0

#: How long one request may take before it is abandoned. An hour-long
#: recording sent whole to a service takes minutes, so a short timeout
#: would fail perfectly healthy requests.
MINIMUM_PROVIDER_TIMEOUT_SECONDS = 10.0
MAXIMUM_PROVIDER_TIMEOUT_SECONDS = 7200.0
DEFAULT_PROVIDER_TIMEOUT_SECONDS = 900.0

#: How many times a failed request is tried again, and how long the wait
#: before the first retry is. Zero attempts is allowed, because a user
#: watching the cost of a metered service may prefer to be told about a
#: failure rather than to pay for another try.
MINIMUM_PROVIDER_RETRY_ATTEMPTS = 0
MAXIMUM_PROVIDER_RETRY_ATTEMPTS = 10
DEFAULT_PROVIDER_RETRY_ATTEMPTS = 2

MINIMUM_PROVIDER_RETRY_BACKOFF_SECONDS = 0.0
MAXIMUM_PROVIDER_RETRY_BACKOFF_SECONDS = 300.0
DEFAULT_PROVIDER_RETRY_BACKOFF_SECONDS = 2.0

#: How much of the previous chunk each chunk repeats. A word cut in half by
#: a chunk boundary is heard properly by neither request, so the boundary
#: is crossed twice and the duplicated words are dropped afterwards.
#:
#: Eight seconds rather than two, and the reason is the join rather than the
#: cut. The duplicated words are dropped by matching the text either side of
#: the boundary, and that match needs at least three shared words to be sure
#: of itself. Two seconds of speech very often holds a pause, a breath or a
#: single word, and then the two chunks cannot be matched at all and the
#: words at the join are either doubled or lost. Eight seconds holds enough
#: speech for the match almost always, at the cost of eight seconds of audio
#: being sent twice per join, which on an hour-long recording is under a
#: minute.
MINIMUM_CHUNK_OVERLAP_SECONDS = 0.0
MAXIMUM_CHUNK_OVERLAP_SECONDS = 30.0
DEFAULT_CHUNK_OVERLAP_SECONDS = 8.0

#: A ceiling on how many disputes one recording may send for a second
#: opinion. Escalation is charged per request, and a recording that
#: confuses every service could otherwise send thousands of them before
#: anybody noticed. Reaching the ceiling is not an error: the remaining
#: disputes simply go to the review queue for a person to settle.
MINIMUM_ESCALATIONS_PER_RECORDING = 0
MAXIMUM_ESCALATIONS_PER_RECORDING = 10_000
DEFAULT_ESCALATIONS_PER_RECORDING = 200

#: The folder written beside each recording, holding the transcript, the
#: untouched provider responses and the provenance. The name is the
#: recording's own name with this on the end.
DEFAULT_TRANSCRIPT_FOLDER_SUFFIX = ".transcript"

#: What a minute of audio or one adjudication request may be said to cost.
#: Nobody is charged more than this by any of these services, and a figure
#: above it is far more likely to be a typing mistake than a real price.
MINIMUM_COST = 0.0
MAXIMUM_COST = 100.0


@dataclass
class EnhanceSettings:
    """What the Enhance Audio dialog opens with.

    These are kept separately from the rest because they are set in their
    own dialog rather than in Settings, and because there are enough of
    them that mixing them in would bury the handful of settings that are
    about the player.
    """

    output_folder: str | None = None
    """Where enhanced copies were last written. Empty until the first run."""

    target_lufs: float = DEFAULT_TARGET_LUFS
    ceiling_dbtp: float = DEFAULT_CEILING_DBTP
    maximum_gain_db: float = DEFAULT_MAXIMUM_GAIN_DB
    use_limiter: bool = False
    output_format: str = DEFAULT_OUTPUT_FORMAT.key
    replace_existing: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnhanceSettings":
        settings = cls()
        folder = data.get("output_folder")
        if isinstance(folder, str) and folder.strip():
            settings.output_folder = folder
        settings.target_lufs = _clean_number(
            data.get("target_lufs"), settings.target_lufs, MINIMUM_TARGET_LUFS, MAXIMUM_TARGET_LUFS
        )
        settings.ceiling_dbtp = _clean_number(
            data.get("ceiling_dbtp"),
            settings.ceiling_dbtp,
            MINIMUM_CEILING_DBTP,
            MAXIMUM_CEILING_DBTP,
        )
        settings.maximum_gain_db = _clean_number(
            data.get("maximum_gain_db"),
            settings.maximum_gain_db,
            MINIMUM_MAXIMUM_GAIN_DB,
            MAXIMUM_MAXIMUM_GAIN_DB,
        )
        limiter = data.get("use_limiter")
        if isinstance(limiter, bool):
            settings.use_limiter = limiter
        # An unknown format name falls back to the default rather than
        # being kept, so a hand-edited file cannot ask for an encoder that
        # does not exist.
        settings.output_format = output_format_for(data.get("output_format")).key
        replace = data.get("replace_existing")
        if isinstance(replace, bool):
            settings.replace_existing = replace
        return settings


@dataclass
class OpenAiTranscriptionSettings:
    """How the application talks to OpenAI for speech-to-text.

    OpenAI's job in the ensemble is to work out what was probably said. It
    is not asked for timing or for speakers, so there is nothing here about
    either.

    The chunk size is a setting rather than a constant because it is a
    judgement about somebody else's limit. OpenAI takes 25 MB per request,
    and the size a chunk will encode to can only be guessed at before it is
    encoded, so the target is set below the limit and can be moved down
    again by a user whose recordings keep landing just over it.
    """

    requirements: ClassVar[str] = "an API key and a model name"

    api_key: str = ""
    model: str = DEFAULT_OPENAI_TRANSCRIPTION_MODEL
    parameters: dict[str, Any] = field(default_factory=dict)
    """Extra request parameters, passed through as they are written.

    Free-form on purpose. Services add parameters between our releases, and
    a user who knows about a new one should be able to use it by typing it
    in rather than by waiting for us.
    """

    chunk_target_bytes: int = DEFAULT_CHUNK_TARGET_BYTES
    enabled: bool = True

    @property
    def is_configured(self) -> bool:
        """Whether this service has what it needs to be called at all."""
        return bool(self.api_key.strip()) and bool(self.model.strip())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpenAiTranscriptionSettings":
        settings = cls()
        settings.api_key = _clean_optional_text(data.get("api_key"))
        settings.model = _clean_text(data.get("model"), settings.model)
        settings.parameters = _clean_parameters(data.get("parameters"))
        settings.chunk_target_bytes = _clean_count(
            data.get("chunk_target_bytes"),
            settings.chunk_target_bytes,
            MINIMUM_CHUNK_TARGET_BYTES,
            MAXIMUM_CHUNK_TARGET_BYTES,
        )
        settings.enabled = _clean_flag(data.get("enabled"), settings.enabled)
        return settings


@dataclass
class OpenAiAdjudicationSettings:
    """How the application asks OpenAI to settle a dispute the rules could not.

    This is a different model from the transcription one and is chosen
    freely, because deciding between two candidate readings is a reasoning
    task rather than a listening one. The specification allows no other
    company's model in this role.

    The model name is plain text rather than a list of known names. A list
    would mean that using a model announced this morning needs a release
    this afternoon, and there is no benefit to us in that: an unknown name
    is rejected by OpenAI with a perfectly clear message.
    """

    requirements: ClassVar[str] = "an API key and a model name"

    api_key: str = ""
    model: str = DEFAULT_OPENAI_ADJUDICATION_MODEL
    reasoning_effort: str = DEFAULT_OPENAI_REASONING_EFFORT
    """How hard the model should think, or empty to leave it unsaid.

    Empty is a real answer rather than a missing one. Models differ in
    which parameters they accept, and sending a reasoning control to a
    model that has none is an error, so clearing this is how a user says
    "do not send it".
    """

    parameters: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key.strip()) and bool(self.model.strip())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpenAiAdjudicationSettings":
        settings = cls()
        settings.api_key = _clean_optional_text(data.get("api_key"))
        settings.model = _clean_text(data.get("model"), settings.model)
        # Read through the default rather than past it. An older file has no
        # reasoning effort at all and should get the default, but a file
        # that holds an empty one is saying "do not send it", and those two
        # cases would otherwise be indistinguishable.
        settings.reasoning_effort = _clean_optional_text(
            data.get("reasoning_effort", settings.reasoning_effort), settings.reasoning_effort
        )
        settings.parameters = _clean_parameters(data.get("parameters"))
        settings.enabled = _clean_flag(data.get("enabled"), settings.enabled)
        return settings


@dataclass
class SmoothingSettings:
    """How the smooth transcript is made from the literal one.

    The smooth transcript is an edited copy that is easy to read: fillers,
    stutters and false starts removed. It uses the OpenAI API key already
    entered for adjudication, so it holds no key of its own.
    """

    run_after_transcription: bool = True
    """Whether the smooth copy is made as soon as a transcription finishes."""

    model: str = DEFAULT_SMOOTHING_MODEL
    reasoning_effort: str = DEFAULT_SMOOTHING_REASONING_EFFORT
    """How hard the model should think, or empty to leave it out of the request."""

    prompt: str = DEFAULT_SMOOTHING_PROMPT
    """The style prompt. Blank text falls back to the default prompt."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SmoothingSettings":
        settings = cls()
        settings.run_after_transcription = _clean_flag(
            data.get("run_after_transcription"), settings.run_after_transcription
        )
        settings.model = _clean_text(data.get("model"), settings.model)
        # Read through the default, as for adjudication: a missing value
        # takes the default, and a saved empty one means "do not send it".
        settings.reasoning_effort = _clean_optional_text(
            data.get("reasoning_effort", settings.reasoning_effort), settings.reasoning_effort
        )
        prompt = data.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            # Kept as typed rather than stripped, because line breaks and
            # indentation are part of how a person laid the prompt out.
            settings.prompt = prompt
        return settings


@dataclass
class ElevenLabsSettings:
    """How the application talks to ElevenLabs Scribe.

    Scribe is the structural backbone: the word timings, the speaker labels
    and the log probabilities that drive the review queue all come from it,
    so a run without it is not a run at all.

    Diarisation and audio-event tagging have their own settings rather than
    living in the parameter dictionary, because the reconciliation stage
    behaves differently depending on them and needs to be able to read them
    without picking through free-form text.

    Forced alignment is listed here as parameters only. It is the same
    service measuring where a corrected word actually falls in the audio,
    and it takes no model name of its own.
    """

    requirements: ClassVar[str] = "an API key and a transcription model name"

    api_key: str = ""
    transcription_model: str = DEFAULT_ELEVENLABS_TRANSCRIPTION_MODEL
    transcription_parameters: dict[str, Any] = field(default_factory=dict)
    forced_alignment_parameters: dict[str, Any] = field(default_factory=dict)

    diarise: bool = True
    """Whether Scribe is asked to tell the speakers apart.

    On by default because the speaker of each word is one of the three
    questions a transcript answers, and Scribe is the initial authority on
    it. The service itself spells the parameter ``diarize``; the adapter
    translates, so that the name a user reads matches the rest of the
    application.
    """

    tag_audio_events: bool = True
    """Whether laughter, applause and the like come back as marked entries.

    They arrive as ordinary-looking words otherwise, which would put things
    nobody said into a verbatim transcript.
    """

    enabled: bool = True

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key.strip()) and bool(self.transcription_model.strip())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ElevenLabsSettings":
        settings = cls()
        settings.api_key = _clean_optional_text(data.get("api_key"))
        settings.transcription_model = _clean_text(
            data.get("transcription_model"), settings.transcription_model
        )
        settings.transcription_parameters = _clean_parameters(data.get("transcription_parameters"))
        settings.forced_alignment_parameters = _clean_parameters(
            data.get("forced_alignment_parameters")
        )
        settings.diarise = _clean_flag(data.get("diarise"), settings.diarise)
        settings.tag_audio_events = _clean_flag(
            data.get("tag_audio_events"), settings.tag_audio_events
        )
        settings.enabled = _clean_flag(data.get("enabled"), settings.enabled)
        return settings


@dataclass
class MicrosoftMaiSettings:
    """How the application talks to Microsoft MAI-Transcribe-1.5.

    This one is reached through Azure AI Foundry, which means the address
    is a resource the user created in their own subscription rather than a
    fixed address we could put in the code. That is why the endpoint is a
    setting with no default: there is no sensible value to guess, and a
    plausible-looking wrong one would be worse than an empty box.

    MAI is a third opinion on English and German and is never depended
    upon. It is a preview service, it is not to be trusted on Afrikaans,
    and a transcript must not fail because it did not answer.
    """

    requirements: ClassVar[str] = "an API key, a resource endpoint and a model name"

    api_key: str = ""
    endpoint: str = ""
    """The Azure resource address, such as
    ``https://your-resource.cognitiveservices.azure.com``."""

    model: str = DEFAULT_MICROSOFT_MODEL
    api_version: str = DEFAULT_MICROSOFT_API_VERSION
    """Which revision of the Azure API to speak.

    Azure requires this on every request and changes it independently of
    the model, so it is a setting of its own rather than part of the
    endpoint the user pastes in.
    """

    parameters: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    @property
    def is_configured(self) -> bool:
        """Whether this service can be called.

        The endpoint counts as much as the key here. A key on its own has
        nowhere to go, because the address belongs to the user's own Azure
        resource and cannot be assumed.
        """
        return (
            bool(self.api_key.strip()) and bool(self.endpoint.strip()) and bool(self.model.strip())
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MicrosoftMaiSettings":
        settings = cls()
        settings.api_key = _clean_optional_text(data.get("api_key"))
        settings.endpoint = _clean_optional_text(data.get("endpoint"))
        settings.model = _clean_text(data.get("model"), settings.model)
        settings.api_version = _clean_text(data.get("api_version"), settings.api_version)
        settings.parameters = _clean_parameters(data.get("parameters"))
        settings.enabled = _clean_flag(data.get("enabled"), settings.enabled)
        return settings


@dataclass
class AssemblyAiSettings:
    """How the application asks AssemblyAI for a second opinion.

    AssemblyAI is not asked to transcribe whole recordings. It is sent
    short context-padded windows around the words the other services could
    not agree on, and around the speaker boundaries that look wrong.

    There are two model names because the better model hears fewer
    languages. Afrikaans is not among them, so a recording that may contain
    Afrikaans needs the older model for its Afrikaans spans. Keeping the
    two apart as named settings is what lets that choice be made per span
    rather than per account.
    """

    requirements: ClassVar[str] = "an API key and a primary model name"

    api_key: str = ""
    primary_model: str = DEFAULT_ASSEMBLYAI_PRIMARY_MODEL
    afrikaans_model: str = DEFAULT_ASSEMBLYAI_AFRIKAANS_MODEL
    parameters: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key.strip()) and bool(self.primary_model.strip())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssemblyAiSettings":
        settings = cls()
        settings.api_key = _clean_optional_text(data.get("api_key"))
        settings.primary_model = _clean_text(data.get("primary_model"), settings.primary_model)
        settings.afrikaans_model = _clean_text(
            data.get("afrikaans_model"), settings.afrikaans_model
        )
        settings.parameters = _clean_parameters(data.get("parameters"))
        settings.enabled = _clean_flag(data.get("enabled"), settings.enabled)
        return settings


@dataclass
class DeepgramSettings:
    """How the application talks to Deepgram, if the user wants it at all.

    Deepgram is the optional challenger. It is brought in when the other
    services remain split and for benchmarking, and nothing in version 1
    depends on it, which is why this is the one provider that is switched
    off until somebody asks for it.
    """

    requirements: ClassVar[str] = "an API key and a model name"

    api_key: str = ""
    model: str = DEFAULT_DEEPGRAM_MODEL
    parameters: dict[str, Any] = field(default_factory=dict)
    enabled: bool = False

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key.strip()) and bool(self.model.strip())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeepgramSettings":
        settings = cls()
        settings.api_key = _clean_optional_text(data.get("api_key"))
        settings.model = _clean_text(data.get("model"), settings.model)
        settings.parameters = _clean_parameters(data.get("parameters"))
        settings.enabled = _clean_flag(data.get("enabled"), settings.enabled)
        return settings


@dataclass
class ProcessingSettings:
    """How a run behaves, as opposed to who it talks to.

    These are the defaults a new recording starts from. A recording may
    override some of them for itself, which is why they are named as
    defaults where they can be overridden and not where they cannot.
    """

    default_afrikaans_enabled: bool = False
    """Whether new recordings expect Afrikaans.

    Off, deliberately. Off does not merely change a weighting: it switches
    the whole Afrikaans path out, so no span of a German recording is ever
    mistaken for Afrikaans and handed to the model chosen for it. A user
    who records in Afrikaans switches this on once and forgets it.
    """

    default_expected_speaker_count: int = DEFAULT_EXPECTED_SPEAKER_COUNT

    escalation_context_seconds_before: float = DEFAULT_ESCALATION_CONTEXT_SECONDS_BEFORE
    escalation_context_seconds_after: float = DEFAULT_ESCALATION_CONTEXT_SECONDS_AFTER
    """How much audio surrounds a disputed word when it is sent out again."""

    provider_timeout_seconds: float = DEFAULT_PROVIDER_TIMEOUT_SECONDS
    provider_retry_attempts: int = DEFAULT_PROVIDER_RETRY_ATTEMPTS
    provider_retry_backoff_seconds: float = DEFAULT_PROVIDER_RETRY_BACKOFF_SECONDS
    provider_chunk_overlap_seconds: float = DEFAULT_CHUNK_OVERLAP_SECONDS

    forced_alignment_enabled: bool = True
    """Whether a changed word's timing is measured again against the audio.

    On, because the alternative is a transcript whose words point at the
    wrong moment of the recording. Switching it off is honest rather than
    free: the affected words then keep the widest span that can be
    defended, and say so.
    """

    escalation_enabled: bool = True
    adjudication_enabled: bool = True
    maximum_escalations_per_recording: int = DEFAULT_ESCALATIONS_PER_RECORDING

    transcript_folder_suffix: str = DEFAULT_TRANSCRIPT_FOLDER_SUFFIX
    """What the folder written beside each recording is called.

    Beside the recording rather than in one central place, so that moving a
    recording to another drive takes its transcript, its provider responses
    and its provenance with it.
    """

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessingSettings":
        settings = cls()
        settings.default_afrikaans_enabled = _clean_flag(
            data.get("default_afrikaans_enabled"), settings.default_afrikaans_enabled
        )
        settings.default_expected_speaker_count = _clean_count(
            data.get("default_expected_speaker_count"),
            settings.default_expected_speaker_count,
            MINIMUM_EXPECTED_SPEAKER_COUNT,
            MAXIMUM_EXPECTED_SPEAKER_COUNT,
        )
        settings.escalation_context_seconds_before = _clean_number(
            data.get("escalation_context_seconds_before"),
            settings.escalation_context_seconds_before,
            MINIMUM_ESCALATION_CONTEXT_SECONDS,
            MAXIMUM_ESCALATION_CONTEXT_SECONDS,
        )
        settings.escalation_context_seconds_after = _clean_number(
            data.get("escalation_context_seconds_after"),
            settings.escalation_context_seconds_after,
            MINIMUM_ESCALATION_CONTEXT_SECONDS,
            MAXIMUM_ESCALATION_CONTEXT_SECONDS,
        )
        settings.provider_timeout_seconds = _clean_number(
            data.get("provider_timeout_seconds"),
            settings.provider_timeout_seconds,
            MINIMUM_PROVIDER_TIMEOUT_SECONDS,
            MAXIMUM_PROVIDER_TIMEOUT_SECONDS,
        )
        settings.provider_retry_attempts = _clean_count(
            data.get("provider_retry_attempts"),
            settings.provider_retry_attempts,
            MINIMUM_PROVIDER_RETRY_ATTEMPTS,
            MAXIMUM_PROVIDER_RETRY_ATTEMPTS,
        )
        settings.provider_retry_backoff_seconds = _clean_number(
            data.get("provider_retry_backoff_seconds"),
            settings.provider_retry_backoff_seconds,
            MINIMUM_PROVIDER_RETRY_BACKOFF_SECONDS,
            MAXIMUM_PROVIDER_RETRY_BACKOFF_SECONDS,
        )
        settings.provider_chunk_overlap_seconds = _clean_number(
            data.get("provider_chunk_overlap_seconds"),
            settings.provider_chunk_overlap_seconds,
            MINIMUM_CHUNK_OVERLAP_SECONDS,
            MAXIMUM_CHUNK_OVERLAP_SECONDS,
        )
        settings.forced_alignment_enabled = _clean_flag(
            data.get("forced_alignment_enabled"), settings.forced_alignment_enabled
        )
        settings.escalation_enabled = _clean_flag(
            data.get("escalation_enabled"), settings.escalation_enabled
        )
        settings.adjudication_enabled = _clean_flag(
            data.get("adjudication_enabled"), settings.adjudication_enabled
        )
        settings.maximum_escalations_per_recording = _clean_count(
            data.get("maximum_escalations_per_recording"),
            settings.maximum_escalations_per_recording,
            MINIMUM_ESCALATIONS_PER_RECORDING,
            MAXIMUM_ESCALATIONS_PER_RECORDING,
        )
        settings.transcript_folder_suffix = _clean_folder_suffix(
            data.get("transcript_folder_suffix"), settings.transcript_folder_suffix
        )
        return settings


@dataclass
class CostSettings:
    """What a run is expected to cost, so the user can be told before it starts.

    Five metered services and a reasoning model can add up to a real amount
    of money on a long recording, and finding that out afterwards is not
    acceptable. These rates turn a recording's length into a figure the
    user sees before anything is sent.

    The rates below are in United States dollars per minute of audio. One
    of them was checked against what the service publishes, and is marked as
    such. The rest are estimates, and they were plausible when they were
    written and not necessarily since. Prices change, and they change
    without telling us, so check each of them against what the service
    charges today before trusting a figure that came out of them. They are
    settings for exactly that reason: a wrong number that looks
    authoritative is worse than one the user knows to check.
    """

    openai_transcription_per_minute: float = 0.0045
    """Verified against OpenAI's published price for ``gpt-transcribe`` on
    17 August 2026. The only rate here that is not a guess."""

    elevenlabs_per_minute: float = 0.007
    """Estimate. Not checked."""

    microsoft_per_minute: float = 0.017
    """Estimate. Not checked."""

    assemblyai_per_minute: float = 0.005
    """Estimate. Not checked."""

    deepgram_per_minute: float = 0.005
    """Estimate. Not checked."""

    adjudication_per_request: float = 0.02
    """One adjudication call, averaged. It depends on how much evidence the
    disputed region carries and on how hard the model is asked to think, so
    it is an average of very unequal requests rather than a price."""

    smoothing_per_request: float = 0.03
    """One request that edits about 2,000 words into the smooth transcript.
    Estimate. Not checked."""

    confirm_before_running: bool = True
    """Whether the estimate is shown and agreed to before a run starts."""

    def per_minute_for(self, provider: Provider) -> float:
        """The rate for one service, by the same name the rest of the code uses."""
        return {
            Provider.ELEVENLABS: self.elevenlabs_per_minute,
            Provider.OPENAI: self.openai_transcription_per_minute,
            Provider.MICROSOFT: self.microsoft_per_minute,
            Provider.ASSEMBLYAI: self.assemblyai_per_minute,
            Provider.DEEPGRAM: self.deepgram_per_minute,
        }[provider]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CostSettings":
        settings = cls()
        for name in (
            "elevenlabs_per_minute",
            "openai_transcription_per_minute",
            "microsoft_per_minute",
            "assemblyai_per_minute",
            "deepgram_per_minute",
            "adjudication_per_request",
            "smoothing_per_request",
        ):
            setattr(
                settings,
                name,
                _clean_number(data.get(name), getattr(settings, name), MINIMUM_COST, MAXIMUM_COST),
            )
        settings.confirm_before_running = _clean_flag(
            data.get("confirm_before_running"), settings.confirm_before_running
        )
        return settings


@dataclass
class TranscriptionSettings:
    """Everything transcription needs to know, gathered in one place.

    It is one object rather than a scattering of fields on :class:`Settings`
    because the whole of it travels together: the dialog edits it, the
    pipeline reads it, and a redacted copy of it is written into every
    transcript so that a run can be explained a year later.
    """

    openai_transcription: OpenAiTranscriptionSettings = field(
        default_factory=OpenAiTranscriptionSettings
    )
    openai_adjudication: OpenAiAdjudicationSettings = field(
        default_factory=OpenAiAdjudicationSettings
    )
    elevenlabs: ElevenLabsSettings = field(default_factory=ElevenLabsSettings)
    microsoft: MicrosoftMaiSettings = field(default_factory=MicrosoftMaiSettings)
    assemblyai: AssemblyAiSettings = field(default_factory=AssemblyAiSettings)
    deepgram: DeepgramSettings = field(default_factory=DeepgramSettings)
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    cost: CostSettings = field(default_factory=CostSettings)
    smoothing: SmoothingSettings = field(default_factory=SmoothingSettings)

    def for_provenance(self) -> dict[str, Any]:
        """Return these settings with every secret taken out.

        This is what is written into a transcript so that the run behind it
        can be reproduced or explained. It must never carry a credential,
        because a transcript is a document people send to each other.

        The redaction works by the name of the field rather than by a list
        of the fields to remove. That is the important part. A list would be
        correct on the day it was written and wrong the first time somebody
        adds a provider, and it would be wrong silently: the key would
        simply start appearing in every transcript with nothing to notice
        it. A rule cannot be forgotten in the same way, because a new field
        called ``api_key`` is caught by the same rule as the old ones.

        The rule reaches everywhere, not only into the fields we defined. A
        credential sits just as comfortably inside a free-form parameter
        dictionary, where we have no idea what a user has put, as it does
        beside a model name, so the whole structure is walked.

        What the rule matches on is the *end* of the field name rather than
        any part of it, and that boundary is deliberate. Matching anywhere
        would also catch ``keyterms``, the names and specialised vocabulary
        a service was primed with, which the specification requires in the
        record of every request. Losing those would be failing a stated
        requirement, silently, while appearing to be careful. Credential
        fields are named for what they hold, so the noun falls at the end,
        and matching there catches every one of them without the collateral
        damage.
        """
        return _without_secrets(asdict(self))

    def missing_requirements(self) -> list[str]:
        """Say what would stop a run, as sentences a person can act on.

        Checked before anything is sent, because the useful moment to learn
        that a key is missing is before half the services have been paid.
        Only what is switched on is checked: an unconfigured Deepgram that
        nobody asked for is not a problem.
        """
        problems: list[str] = []
        for name, section, required in (
            ("ElevenLabs Scribe", self.elevenlabs, True),
            ("OpenAI transcription", self.openai_transcription, True),
            ("Microsoft MAI", self.microsoft, False),
            ("AssemblyAI", self.assemblyai, False),
            ("Deepgram", self.deepgram, False),
        ):
            if required and not section.enabled:
                problems.append(
                    f"{name} is switched off. It provides part of every transcript, so a "
                    "run cannot be made without it."
                )
            elif section.enabled and not section.is_configured:
                problems.append(
                    f"{name} is switched on but is not set up. It needs "
                    f"{section.requirements} in Settings."
                )
        # A switched-off adjudication service needs no key. The run goes
        # ahead without it and says so in the transcript's warnings.
        if (
            self.processing.adjudication_enabled
            and self.openai_adjudication.enabled
            and not self.openai_adjudication.is_configured
        ):
            problems.append(
                "Adjudication is switched on but is not set up. It needs "
                f"{self.openai_adjudication.requirements} in Settings, or it can be "
                "switched off, in which case unsettled words wait for review instead."
            )
        return problems

    def missing_libraries(
        self,
        probe: Callable[[str, str | None], str | None] | None = None,
    ) -> list[str]:
        """Say which vendor libraries a run would need and cannot load.

        This is the other half of :meth:`missing_requirements`, and it exists
        because of what a missing library does to a run that has already
        started. Every service's library is imported only at the moment the
        service is first called, never when the application starts, so a
        package that was never installed, or that was half installed, is not
        found until the recording has been prepared and the other services
        have been sent the audio and paid for. The run then reports that one
        service "did not answer", which reads like a network fault and is
        nothing of the kind.

        Only the libraries of services that are switched on are tried, in the
        same spirit as the key check: a Deepgram that nobody asked for is not
        a problem. The try is a real import rather than a look at what is
        installed, and for ElevenLabs it reaches for the client class rather
        than the top-level package, because a broken installation of that
        package imports its top level perfectly well and fails one line
        later.

        ``probe`` is how the check is tried without the libraries: it is given
        the module name and the attribute wanted from it, and answers with
        what went wrong or ``None``. The default does the real import.
        """
        tried = probe if probe is not None else _probe_library
        problems: list[str] = []
        wanted: list[tuple[str, str, str | None]] = []
        if self.elevenlabs.enabled:
            wanted.append(("ElevenLabs Scribe", "elevenlabs", "ElevenLabs"))
        adjudicating = self.processing.adjudication_enabled and self.openai_adjudication.enabled
        if self.openai_transcription.enabled or adjudicating:
            wanted.append(("OpenAI", "openai", "OpenAI"))
        if self.microsoft.enabled:
            wanted.append(("Microsoft MAI", "httpx", None))
        if self.assemblyai.enabled:
            wanted.append(("AssemblyAI", "assemblyai", None))
        if self.deepgram.enabled:
            wanted.append(("Deepgram", "deepgram", None))
        for name, module, attribute in wanted:
            fault = tried(module, attribute)
            if fault is None:
                continue
            problems.append(
                f"{name} is switched on, but the {module} library it talks through "
                f"could not be loaded: {fault} Run \"VoxVerbatim.cmd\" again to "
                "install the libraries, or switch the service off in Settings."
            )
        return problems

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptionSettings":
        settings = cls()
        for name, builder in (
            ("openai_transcription", OpenAiTranscriptionSettings),
            ("openai_adjudication", OpenAiAdjudicationSettings),
            ("elevenlabs", ElevenLabsSettings),
            ("microsoft", MicrosoftMaiSettings),
            ("assemblyai", AssemblyAiSettings),
            ("deepgram", DeepgramSettings),
            ("processing", ProcessingSettings),
            ("cost", CostSettings),
            ("smoothing", SmoothingSettings),
        ):
            section = data.get(name)
            if isinstance(section, dict):
                setattr(settings, name, builder.from_dict(section))
        return settings


def _probe_library(module: str, attribute: str | None) -> str | None:
    """Try to load one vendor library, and say what went wrong if it cannot be.

    Anything at all is caught, not only ImportError. A half-installed package
    can raise almost anything while it is being imported -- a missing
    compiled extension, a version check, a syntax error in a file that was
    cut short -- and every one of those means the same thing to the person
    about to press Start.
    """
    try:
        loaded = importlib.import_module(module)
        if attribute is not None:
            getattr(loaded, attribute)
    except Exception as error:  # any failure here means the run cannot use it
        reason = str(error).strip() or type(error).__name__
        return f"{reason}."
    return None


@dataclass
class Settings:
    """Everything the user can choose in the Settings dialog."""

    reopen_last_folder: bool = True
    """Whether the folder from the last run is opened again on start-up."""

    short_skip_seconds: int = 15
    medium_skip_seconds: int = 120
    long_skip_seconds: int = 300
    """How far the three pairs of skip buttons move."""

    enhance: EnhanceSettings = field(default_factory=EnhanceSettings)
    """What the Enhance Audio dialog opens with next time."""

    transcription: TranscriptionSettings = field(default_factory=TranscriptionSettings)
    """The services transcription talks to, and how a run behaves."""

    @property
    def skip_seconds(self) -> tuple[int, int, int]:
        """The three skip intervals, shortest first."""
        return (self.short_skip_seconds, self.medium_skip_seconds, self.long_skip_seconds)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["version"] = SETTINGS_FORMAT_VERSION
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        """Build settings from loaded JSON, ignoring anything unusable.

        Every value is checked rather than trusted. A hand-edited file with
        nonsense in it should fall back to the default for that one setting,
        not stop the application or leave a skip button that moves zero
        seconds.
        """
        settings = cls()
        reopen = data.get("reopen_last_folder")
        if isinstance(reopen, bool):
            settings.reopen_last_folder = reopen
        settings.short_skip_seconds = _clean_skip(
            data.get("short_skip_seconds"), settings.short_skip_seconds
        )
        settings.medium_skip_seconds = _clean_skip(
            data.get("medium_skip_seconds"), settings.medium_skip_seconds
        )
        settings.long_skip_seconds = _clean_skip(
            data.get("long_skip_seconds"), settings.long_skip_seconds
        )
        enhance = data.get("enhance")
        if isinstance(enhance, dict):
            settings.enhance = EnhanceSettings.from_dict(enhance)
        # A settings file written before transcription existed has no
        # section for it, and must keep working. Every field then takes its
        # documented default, which is what makes adding settings safe.
        transcription = data.get("transcription")
        if isinstance(transcription, dict):
            settings.transcription = TranscriptionSettings.from_dict(transcription)
        version = data.get("version")
        if isinstance(version, bool) or not isinstance(version, int) or version < 2:
            _migrate_adjudication_model(settings.transcription.openai_adjudication)
        return settings


def _migrate_adjudication_model(section: OpenAiAdjudicationSettings) -> None:
    """Move a file that saved the old default model on to the new default.

    Saving writes every field, so anyone who ever pressed Save has the old
    default stored as if they had chosen it, and a changed default alone
    would never reach them. This runs only for files written before
    version 2. A file saved since then carries version 2, so someone who
    deliberately picks the old model again keeps it.

    A saved effort is kept, because the new model accepts it too, except
    for the two the new model refuses, which become its default.
    """
    if section.model != _RETIRED_ADJUDICATION_MODEL:
        return
    section.model = DEFAULT_OPENAI_ADJUDICATION_MODEL
    if section.reasoning_effort in _EFFORTS_THE_NEW_MODEL_REFUSES:
        section.reasoning_effort = DEFAULT_OPENAI_REASONING_EFFORT


def _clean_skip(value: Any, default: int) -> int:
    """Return a usable number of seconds, falling back to ``default``."""
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    if not MINIMUM_SKIP_SECONDS <= value <= MAXIMUM_SKIP_SECONDS:
        return default
    return value


def _clean_number(value: Any, default: float, lowest: float, highest: float) -> float:
    """Return a usable measurement, falling back to ``default``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    if not lowest <= value <= highest:
        return default
    return float(value)


def _clean_count(value: Any, default: int, lowest: int, highest: int) -> int:
    """Return a usable whole number, falling back to ``default``."""
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    if not lowest <= value <= highest:
        return default
    return value


def _clean_flag(value: Any, default: bool) -> bool:
    """Return a usable yes-or-no answer, falling back to ``default``."""
    return value if isinstance(value, bool) else default


def _clean_text(value: Any, default: str) -> str:
    """Return text that must say something, falling back to ``default``.

    This is for the values a request cannot be made without, such as a
    model name. Blank text counts as missing rather than being kept,
    because a request sent with an empty model name fails in a way that is
    far harder to understand than one sent with the default.
    """
    if not isinstance(value, str) or not value.strip():
        return default
    return value.strip()


def _clean_optional_text(value: Any, default: str = "") -> str:
    """Return text that is allowed to say nothing.

    Empty is a real answer for these. An empty API key means the service is
    not set up yet, and an empty reasoning effort means the parameter is
    left out of the request altogether, so neither can be replaced by a
    default without changing what the user asked for.
    """
    if not isinstance(value, str):
        return default
    return value.strip()


def _clean_parameters(value: Any) -> dict[str, Any]:
    """Return free-form request parameters, or nothing if they are unusable.

    These dictionaries are passed to a service more or less as they are
    written, so the only thing worth insisting on is that they are honestly
    a JSON object: named values that will survive being written to the
    settings file and read back unchanged. Anything that would not survive
    that trip is thrown away whole rather than in part, because half a set
    of parameters is a request nobody meant to make.

    ``allow_nan`` is switched off deliberately. Python will happily write
    ``NaN`` into what is then no longer valid JSON, and the file would come
    back looking fine to us and broken to everything else.
    """
    if not isinstance(value, dict) or not all(isinstance(name, str) for name in value):
        return {}
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError):
        return {}


#: Characters Windows will not accept in a folder name. The separators
#: among them matter most: a suffix containing one would write the
#: transcript somewhere other than beside its recording.
_ILLEGAL_FOLDER_CHARACTERS = frozenset('<>:"/\\|?*')


def _clean_folder_suffix(value: Any, default: str) -> str:
    """Return a suffix that names a folder and nothing else.

    The suffix is joined onto a recording's name to make the folder written
    beside it, so it decides where files land. A hand-edited value holding
    a separator or a ``..`` would put those files in another folder
    entirely, and the user would have no idea where their transcripts had
    gone, so anything of that shape falls back to the default.
    """
    if not isinstance(value, str):
        return default
    suffix = value.strip()
    if not suffix or suffix in {".", ".."} or ".." in suffix:
        return default
    if any(character in _ILLEGAL_FOLDER_CHARACTERS for character in suffix):
        return default
    if any(ord(character) < 32 for character in suffix):
        return default
    return suffix


#: The nouns a credential field is named after. A field whose name ends
#: with one of these is a credential, wherever it sits.
#:
#: The test is "ends with" rather than "contains" on purpose, and the
#: difference matters. Credential fields are named for what they are, so
#: the noun lands at the end: ``api_key``, ``subscription_key``,
#: ``access_token``, ``client_secret``. A field that merely begins with the
#: same letters is usually something else entirely, and ``keyterms`` is the
#: case that proves it. Keyterms are the names and specialised vocabulary a
#: service was primed with, and the specification requires them in the
#: record of every request, so a rule that swept them up would quietly fail
#: a stated requirement while looking careful.
_SECRET_NAME_ENDINGS = ("key", "secret", "token", "password", "credential")

#: Matched anywhere in the name rather than only at the end, because a
#: field holding one key among several is still named for it:
#: ``api_key_for_billing`` is no less a credential than ``api_key``.
_SECRET_NAME_PART = "api_key"


def _is_secret_name(name: str) -> bool:
    lowered = name.lower()
    if _SECRET_NAME_PART in lowered:
        return True
    return lowered.endswith(_SECRET_NAME_ENDINGS)


def _without_secrets(value: Any) -> Any:
    """Return a copy of ``value`` with every secret-looking field removed.

    It walks the whole structure rather than only the top level, because a
    key can sit inside a free-form parameter dictionary just as easily as
    beside a model name, and a user who has put one there has no reason to
    expect us to publish it.
    """
    if isinstance(value, dict):
        return {
            name: _without_secrets(item)
            for name, item in value.items()
            if not _is_secret_name(str(name))
        }
    if isinstance(value, list):
        return [_without_secrets(item) for item in value]
    return value


class SettingsStore:
    """Reads and writes the settings file at a fixed location."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> Settings:
        """Return the saved settings, or the defaults if there are none usable."""
        data = read_json_object(self._path)
        if data is None:
            return Settings()
        return Settings.from_dict(data)

    def save(self, settings: Settings) -> bool:
        return write_json_object(self._path, settings.to_dict())
