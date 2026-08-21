"""Tests for the ElevenLabs adapter.

Nothing here touches the network and nothing needs an API key. A fake client
stands in for the library, which is why the adapter takes one through its
constructor: the interesting behaviour is all in what it sends and what it
makes of what comes back, and both can be examined exactly when the client
is a test double.

The tests that matter most are the ones about time. Everything downstream
trusts these numbers, so a chunk taken from the middle of a recording is
used throughout rather than a convenient offset of zero, because an adapter
that forgot to shift its times would still pass every test written against
an offset of zero.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from audio_transcriber.transcription import providers
from audio_transcriber.transcription.model import Language, Provider
from audio_transcriber.transcription.providers import elevenlabs as elevenlabs_module
from audio_transcriber.transcription.providers.base import (
    ProviderError,
    ProviderNotConfigured,
    ProviderUnavailable,
    TranscriptionRequest,
)
from audio_transcriber.transcription.providers.elevenlabs import (
    DEFAULT_MAXIMUM_KEYTERMS,
    KEYTERM_MINIMUM_CHARGE_THRESHOLD,
    MAXIMUM_KEYTERM_CHARACTERS,
    MAXIMUM_KEYTERM_WORDS,
    ElevenLabsForcedAligner,
    ElevenLabsProvider,
)

#: A key that is obviously not real, but is a distinctive string so a test
#: can prove it appears nowhere in what gets written to provenance.
API_KEY = "xi-secret-key-do-not-record"

#: The audio starts ten minutes and twelve and a half seconds into the
#: recording, so a missing offset shows up as a wrong answer rather than as
#: the right one by luck.
OFFSET = 612.5


# -- The fake client -----------------------------------------------------


class FakeSpeechToText:
    """Stands in for ``client.speech_to_text``, remembering what it was sent."""

    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def convert(self, **arguments):
        self.calls.append(arguments)
        if self.error is not None:
            raise self.error
        return self.response


class FakeForcedAlignment:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def create(self, **arguments):
        self.calls.append(arguments)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response=None, error: Exception | None = None, alignment=None) -> None:
        self.speech_to_text = FakeSpeechToText(response, error)
        self.forced_alignment = FakeForcedAlignment(alignment)


def word(text, kind="word", start=None, end=None, logprob=None, speaker=None):
    return SimpleNamespace(
        text=text, type=kind, start=start, end=end, logprob=logprob, speaker_id=speaker
    )


class FakeTranscription(SimpleNamespace):
    """Stands in for the library's response object.

    It reads like the real one, field by field, and it converts itself to a
    plain dictionary the way the real one does. That conversion is what the
    adapter uses to keep the answer, so a test double without it would not
    exercise the path that matters.
    """

    def __init__(self, payload: dict) -> None:
        super().__init__(**payload)
        self._payload = payload

    def model_dump(self) -> dict:
        return self._payload


def transcription(words, language_code="en", probability=0.99, identifier="tr-1", extra=None):
    payload = {
        "language_code": language_code,
        "language_probability": probability,
        "text": " ".join(entry.text for entry in words),
        "words": words,
        "transcription_id": identifier,
        "audio_duration_secs": 12.0,
    }
    payload.update(extra or {})
    return FakeTranscription(payload)


SAMPLE_WORDS = [
    word("Hello", start=0.10, end=0.50, logprob=-0.02, speaker="speaker_0"),
    word(" ", kind="spacing", start=0.50, end=0.52),
    word("world", start=0.52, end=0.90, logprob=-0.51, speaker="speaker_0"),
    word(",", start=0.90, end=0.95, logprob=-0.10, speaker="speaker_0"),
    word("(laughter)", kind="audio_event", start=1.00, end=2.00),
    word("again", start=2.00, end=2.40, logprob=-0.08, speaker="speaker_1"),
]


@pytest.fixture
def audio(tmp_path) -> Path:
    path = tmp_path / "meeting.wav"
    path.write_bytes(b"RIFF----WAVEfmt ")
    return path


@pytest.fixture
def request_for(audio):
    def build(**overrides) -> TranscriptionRequest:
        arguments = {"audio_path": audio, "canonical_offset": OFFSET, "duration": 12.0}
        arguments.update(overrides)
        return TranscriptionRequest(**arguments)

    return build


def build_provider(client=None, **overrides) -> ElevenLabsProvider:
    arguments = {"api_key": API_KEY, "client": client}
    arguments.update(overrides)
    return ElevenLabsProvider(**arguments)


# -- Times ---------------------------------------------------------------


def test_every_time_comes_back_on_the_canonical_timeline(request_for):
    """The offset is added to every start and every end, without exception."""
    client = FakeClient(transcription(SAMPLE_WORDS))
    result = build_provider(client).transcribe(request_for())

    assert result.succeeded
    assert [(token.start, token.end) for token in result.tokens] == [
        (OFFSET + 0.10, OFFSET + 0.50),
        (OFFSET + 0.50, OFFSET + 0.52),
        (OFFSET + 0.52, OFFSET + 0.90),
        (OFFSET + 0.90, OFFSET + 0.95),
        (OFFSET + 1.00, OFFSET + 2.00),
        (OFFSET + 2.00, OFFSET + 2.40),
    ]


def test_a_chunk_that_starts_at_the_beginning_is_left_where_it_is(request_for):
    """An offset of zero must change nothing, which is the other half of the rule."""
    client = FakeClient(transcription(SAMPLE_WORDS))
    result = build_provider(client).transcribe(request_for(canonical_offset=0.0))

    assert result.tokens[0].start == pytest.approx(0.10)
    assert result.tokens[-1].end == pytest.approx(2.40)


# -- What the service returns that is not a word -------------------------


def test_spacing_punctuation_and_sounds_are_kept_but_never_aligned(request_for):
    """All three keep their timing, and none of them is offered to alignment.

    They are worth keeping because the times they carry are real. None of
    them is a spoken word, so matching them against another service's text
    would line laughter up against whatever word happened to be near it.
    """
    client = FakeClient(transcription(SAMPLE_WORDS))
    result = build_provider(client).transcribe(request_for())

    assert [token.is_spoken_word for token in result.tokens] == [
        True,  # Hello
        False,  # the space
        True,  # world
        False,  # the comma
        False,  # the laughter
        True,  # again
    ]
    assert [token.text for token in result.word_tokens] == ["Hello", "world", "again"]


def test_a_sound_is_marked_as_a_sound_and_not_as_punctuation(request_for):
    """Laughter and a comma are both skipped, and they are not the same thing.

    A comma is a mark on the page that no service could disagree about. The
    laughter happened in the room, and an export or the review window may
    reasonably choose to show it. Folding the two together would leave
    nothing able to tell them apart afterwards.
    """
    client = FakeClient(transcription(SAMPLE_WORDS))
    result = build_provider(client).transcribe(request_for())

    laughter = result.tokens[4]
    assert laughter.text == "(laughter)"
    assert laughter.is_audio_event is True
    assert laughter.is_punctuation is False
    assert laughter.is_spoken_word is False
    assert laughter not in result.word_tokens
    # It keeps the span it was measured at, on the canonical timeline, which
    # is the whole reason for carrying it at all.
    assert laughter.start == pytest.approx(OFFSET + 1.00)
    assert laughter.end == pytest.approx(OFFSET + 2.00)

    comma = result.tokens[3]
    assert comma.is_punctuation is True
    assert comma.is_audio_event is False

    assert all(not token.is_audio_event for token in result.tokens if token is not laughter)


def test_the_words_are_numbered_in_the_order_the_service_gave_them(request_for):
    client = FakeClient(transcription(SAMPLE_WORDS))
    result = build_provider(client).transcribe(request_for(chunk_index=3))

    assert [token.index for token in result.tokens] == [0, 1, 2, 3, 4, 5]
    assert {token.chunk_index for token in result.tokens} == {3}


# -- Confidence, speakers and language -----------------------------------


def test_the_log_probability_is_carried_through_unchanged(request_for):
    """It stays a log probability. Turning it into a confidence would lose it.

    The two fields exist separately on the token because they are different
    measurements on different scales, and a number converted here could
    never be turned back.
    """
    client = FakeClient(transcription(SAMPLE_WORDS))
    result = build_provider(client).transcribe(request_for())

    assert result.tokens[0].log_probability == pytest.approx(-0.02)
    assert result.tokens[2].log_probability == pytest.approx(-0.51)
    assert all(token.confidence is None for token in result.tokens)


def test_the_speakers_are_reported_in_the_order_they_first_speak(request_for):
    client = FakeClient(transcription(SAMPLE_WORDS))
    result = build_provider(client).transcribe(request_for())

    assert result.speakers == ["speaker_0", "speaker_1"]
    assert result.tokens[0].speaker == "speaker_0"


def test_the_language_is_reported_for_the_file_and_not_for_each_word(request_for):
    """The service has no per-word language, so no word claims to have one."""
    client = FakeClient(transcription(SAMPLE_WORDS, language_code="de"))
    provider = build_provider(client)
    result = provider.transcribe(request_for())

    assert result.detected_language is Language.GERMAN
    assert all(token.language is Language.UNKNOWN for token in result.tokens)
    assert provider.capabilities.per_word_language is False
    assert provider.capabilities.language_detection is True


def test_the_declared_capabilities_match_what_the_service_actually_does():
    capabilities = ElevenLabsProvider().capabilities

    assert capabilities.word_timings
    assert capabilities.diarisation
    assert capabilities.speaker_count_hint
    assert capabilities.word_confidence
    assert capabilities.vocabulary_biasing
    assert capabilities.forced_alignment
    assert not capabilities.context_prompt
    assert not capabilities.time_window
    assert capabilities.supports(Language.AFRIKAANS)
    assert capabilities.maximum_file_bytes == 3 * 1024 * 1024 * 1024
    assert capabilities.maximum_duration_seconds == 10 * 60 * 60


# -- What gets sent ------------------------------------------------------


def test_the_key_terms_are_cut_before_the_minimum_charge_bites(request_for):
    """Above a hundred terms the request is billed 20 seconds, so a hundred go.

    The threshold is "more than a hundred", so a hundred is sent rather than
    ninety-nine. Cutting further would buy nothing: the separate 20 per cent
    keyterm surcharge is charged on any request carrying terms at all.
    """
    client = FakeClient(transcription(SAMPLE_WORDS))
    terms = tuple(f"term-{number}" for number in range(250))
    result = build_provider(client).transcribe(request_for(vocabulary_terms=terms))

    sent = client.speech_to_text.calls[0]["keyterms"]
    # Written out, because DEFAULT_MAXIMUM_KEYTERMS is defined as the
    # threshold and comparing the two against each other proves nothing.
    assert len(sent) == 100
    assert len(sent) == DEFAULT_MAXIMUM_KEYTERMS == KEYTERM_MINIMUM_CHARGE_THRESHOLD
    # The most specific terms come first, so the cut falls off the tail.
    assert sent[0] == "term-0"
    assert result.request.vocabulary_terms == terms[:DEFAULT_MAXIMUM_KEYTERMS]


def test_a_short_vocabulary_is_sent_whole(request_for):
    client = FakeClient(transcription(SAMPLE_WORDS))
    build_provider(client).transcribe(request_for(vocabulary_terms=("Bosch", "Suzanne")))

    assert client.speech_to_text.calls[0]["keyterms"] == ["Bosch", "Suzanne"]


def test_a_term_the_service_would_refuse_is_left_behind_rather_than_sent(request_for):
    """One bad term fails the whole request, so it never leaves this module.

    ElevenLabs rejects a key term of fifty characters or more, one of more
    than five words, and one containing any of a short list of characters.
    None of those refusals drops only the offending term: they refuse the
    request, and the chunk comes back with no transcript at all.
    """
    client = FakeClient(transcription(SAMPLE_WORDS))
    build_provider(client).transcribe(
        request_for(
            vocabulary_terms=(
                "Vermeulen",
                "V" * (MAXIMUM_KEYTERM_CHARACTERS + 1),
                " ".join(["word"] * (MAXIMUM_KEYTERM_WORDS + 1)),
                "Smith [Jr]",
                r"C:\Users",
                "Contoso <Pty>",
                "Schmidt",
            )
        )
    )

    assert client.speech_to_text.calls[0]["keyterms"] == ["Vermeulen", "Schmidt"]


def test_the_key_term_length_boundary_is_where_the_service_puts_it(request_for):
    """ElevenLabs says a key term must be "less than 50 characters".

    Both lengths are written out rather than taken from the constant under
    test. Deriving them would make the test agree with whatever the constant
    happens to say, including the value this pull request exists to correct.
    """
    client = FakeClient(transcription(SAMPLE_WORDS))
    build_provider(client).transcribe(
        request_for(vocabulary_terms=("V" * 49, "W" * 50))
    )

    assert client.speech_to_text.calls[0]["keyterms"] == ["V" * 49]


def test_the_key_term_word_boundary_is_where_the_service_puts_it(request_for):
    """ElevenLabs says a key term may hold "at most 5 words", so six is too many."""
    client = FakeClient(transcription(SAMPLE_WORDS))
    build_provider(client).transcribe(
        request_for(
            vocabulary_terms=(
                " ".join(["five"] * 5),
                " ".join(["six"] * 6),
            )
        )
    )

    assert client.speech_to_text.calls[0]["keyterms"] == [" ".join(["five"] * 5)]


def test_the_cap_can_be_raised_deliberately(request_for):
    """Going over the threshold is allowed, but only on purpose."""
    client = FakeClient(transcription(SAMPLE_WORDS))
    terms = tuple(f"term-{number}" for number in range(250))
    build_provider(client, maximum_keyterms=200).transcribe(
        request_for(vocabulary_terms=terms)
    )

    assert len(client.speech_to_text.calls[0]["keyterms"]) == 200


def test_the_tidy_up_switch_is_never_sent(request_for):
    """Our verbatim layer is authoritative, so no_verbatim never goes out."""
    client = FakeClient(transcription(SAMPLE_WORDS))
    build_provider(client, parameters={"no_verbatim": True}).transcribe(
        request_for(extra_parameters={"no_verbatim": True, "seed": 7})
    )

    call = client.speech_to_text.calls[0]
    body = call["request_options"]["additional_body_parameters"]
    assert "no_verbatim" not in body
    assert "no_verbatim" not in call
    assert body["seed"] == 7


def test_free_form_parameters_go_through_the_escape_hatch(request_for):
    """Settings extras and request extras both reach the pass-through body."""
    client = FakeClient(transcription(SAMPLE_WORDS))
    build_provider(client, parameters={"temperature": 0.0, "seed": 1}).transcribe(
        request_for(extra_parameters={"seed": 42, "some_new_flag": True})
    )

    body = client.speech_to_text.calls[0]["request_options"]["additional_body_parameters"]
    assert body == {"temperature": 0.0, "seed": 42, "some_new_flag": True}


def test_one_language_is_named_and_several_are_left_to_the_service(request_for):
    client = FakeClient(transcription(SAMPLE_WORDS))
    provider = build_provider(client)

    provider.transcribe(request_for(languages=(Language.GERMAN,)))
    assert client.speech_to_text.calls[0]["language_code"] == "de"

    provider.transcribe(request_for(languages=(Language.ENGLISH, Language.AFRIKAANS)))
    assert "language_code" not in client.speech_to_text.calls[1]


def test_the_speaker_count_is_only_sent_when_there_is_more_than_one(request_for):
    client = FakeClient(transcription(SAMPLE_WORDS))
    provider = build_provider(client)

    provider.transcribe(request_for(expected_speaker_count=1))
    assert "num_speakers" not in client.speech_to_text.calls[0]

    provider.transcribe(request_for(expected_speaker_count=4))
    assert client.speech_to_text.calls[1]["num_speakers"] == 4


def test_a_diarisation_field_in_the_extras_is_obeyed_but_never_passed_through(request_for):
    """The escape hatch still works, but it goes through the rules on the way.

    The client library spreads the extras over the named arguments rather
    than under them, so a field left in the body would replace whatever this
    adapter worked out and could land the request with a combination the
    service refuses. All three diarisation fields are therefore taken out of
    the body. Two of them are still read: the extras are, for now, the only
    place either the switch or the threshold can be set at all.
    """
    client = FakeClient(transcription(SAMPLE_WORDS))
    provider = build_provider(
        client,
        parameters={"diarize": False, "diarization_threshold": 0.9},
    )

    provider.transcribe(request_for(diarise=True, expected_speaker_count=1))

    call = client.speech_to_text.calls[0]
    assert call["request_options"]["additional_body_parameters"] == {}
    # Read out of the extras rather than passed through them.
    assert call["diarize"] is False
    # Refused on a request that does not diarise, whichever way it was set.
    assert "diarization_threshold" not in call


def test_a_threshold_from_the_extras_is_sent_when_the_service_will_take_it(request_for):
    """Nothing else can set the threshold today, so the extras have to reach it."""
    client = FakeClient(transcription(SAMPLE_WORDS))
    provider = build_provider(client, parameters={"diarization_threshold": 0.9})

    provider.transcribe(request_for(diarise=True, expected_speaker_count=1))

    call = client.speech_to_text.calls[0]
    assert call["diarization_threshold"] == 0.9
    assert "diarization_threshold" not in call["request_options"]["additional_body_parameters"]


def test_a_speaker_count_in_the_extras_is_dropped_rather_than_obeyed(request_for):
    """This one the adapter genuinely owns, and a second copy could only conflict.

    The count comes from the expected speaker count, which has its own
    setting and its own control, so a number written into the extras beside
    it has nothing to add and could arrive next to a threshold the service
    will not accept it with.
    """
    client = FakeClient(transcription(SAMPLE_WORDS))
    provider = build_provider(client, parameters={"num_speakers": 9})

    provider.transcribe(request_for(diarise=True, expected_speaker_count=1))

    call = client.speech_to_text.calls[0]
    assert "num_speakers" not in call
    assert call["request_options"]["additional_body_parameters"] == {}


def test_key_terms_written_into_the_request_still_go_through_the_rules(request_for):
    """The context layer's list is the one sent, and it is checked like any other.

    That layer hands the terms in as a keyterms parameter as well as a
    vocabulary, because every other service takes them as an ordinary
    parameter. Whichever door they come through, a term the service would
    refuse has to be caught, or one bracket costs the chunk its transcript.
    """
    client = FakeClient(transcription(SAMPLE_WORDS))
    build_provider(client).transcribe(
        request_for(
            vocabulary_terms=("unused",),
            extra_parameters={"keyterms": ["Vermeulen", "Smith [Jr]", "Schmidt"]},
        )
    )

    call = client.speech_to_text.calls[0]
    assert call["keyterms"] == ["Vermeulen", "Schmidt"]
    # Removed from the body, or it would replace the checked list on the way out.
    assert "keyterms" not in call["request_options"]["additional_body_parameters"]


def test_a_fixed_list_in_the_settings_gives_way_to_the_live_vocabulary(request_for):
    """A standing instruction is a fallback, not something that overrules the work.

    The escalation path gathers a vocabulary for one window and sends no
    request parameters at all. A list somebody put in the settings file once
    must not displace it, or every escalated window would be biased towards
    the wrong words.
    """
    client = FakeClient(transcription(SAMPLE_WORDS))
    provider = build_provider(client, parameters={"keyterms": ["Stale"]})

    provider.transcribe(request_for(vocabulary_terms=("Vermeulen", "Schmidt")))
    assert client.speech_to_text.calls[0]["keyterms"] == ["Vermeulen", "Schmidt"]

    # With no live vocabulary, the standing list is what is left to use.
    provider.transcribe(request_for(vocabulary_terms=()))
    assert client.speech_to_text.calls[1]["keyterms"] == ["Stale"]


def test_more_speakers_than_the_service_predicts_are_brought_down(request_for):
    """ElevenLabs predicts at most 32, and refuses a request asking for more."""
    client = FakeClient(transcription(SAMPLE_WORDS))
    build_provider(client).transcribe(
        request_for(diarise=True, expected_speaker_count=40)
    )

    assert client.speech_to_text.calls[0]["num_speakers"] == 32


def test_a_diarisation_value_that_cannot_be_read_is_reported_as_ignored(request_for, caplog):
    """A setting that did nothing has to say so, in the words of what happened.

    The parameter box is read as JSON, so a user who means false can write
    one. The string "false" is somebody who meant it but did not get it, and
    turning it into a boolean by guessing would change what the transcript is
    on the strength of a guess. It is refused and named in the log instead.
    """
    client = FakeClient(transcription(SAMPLE_WORDS))
    provider = build_provider(
        client, parameters={"diarize": "false", "diarization_threshold": "abc"}
    )

    with caplog.at_level("WARNING"):
        provider.transcribe(request_for(diarise=True))

    assert client.speech_to_text.calls[0]["diarize"] is True
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "diarize parameter was ignored" in messages
    assert "diarization_threshold parameter was ignored" in messages


def test_nothing_claims_a_superseded_list_was_sent(request_for, caplog):
    """The log said the terms went out. They did not, and it had to stop saying so."""
    client = FakeClient(transcription(SAMPLE_WORDS))
    provider = build_provider(client, parameters={"keyterms": ["Stale"]})

    with caplog.at_level("INFO"):
        provider.transcribe(request_for(vocabulary_terms=("Vermeulen",)))

    assert client.speech_to_text.calls[0]["keyterms"] == ["Vermeulen"]
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "were not sent to ElevenLabs" in messages
    assert "the settings" in messages


def test_a_keyterms_value_that_is_not_a_list_is_reported_as_ignored(request_for, caplog):
    """Its two neighbours warn when they cannot read a value, and so must this one.

    A bare string is the mistake people actually make. It cannot hold terms,
    so nothing is sent from it, and staying quiet about that would leave
    somebody with a vocabulary they believe is reaching the service.
    """
    client = FakeClient(transcription(SAMPLE_WORDS))
    provider = build_provider(client, parameters={"keyterms": "Bosch"})

    with caplog.at_level("WARNING"):
        provider.transcribe(request_for(vocabulary_terms=()))

    assert "keyterms" not in client.speech_to_text.calls[0]
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "keyterms value in the settings was ignored" in messages


def test_an_empty_list_in_the_request_does_not_erase_the_vocabulary(request_for):
    """An empty parameter names no terms, so it is not an instruction to send none."""
    client = FakeClient(transcription(SAMPLE_WORDS))
    build_provider(client).transcribe(
        request_for(vocabulary_terms=("Vermeulen",), extra_parameters={"keyterms": []})
    )

    assert client.speech_to_text.calls[0]["keyterms"] == ["Vermeulen"]


def test_the_diarisation_threshold_is_sent_when_the_service_will_take_it(request_for):
    client = FakeClient(transcription(SAMPLE_WORDS))
    provider = build_provider(client, diarisation_threshold=0.31)

    provider.transcribe(request_for(diarise=True, expected_speaker_count=1))

    assert client.speech_to_text.calls[0]["diarization_threshold"] == 0.31


def test_the_diarisation_threshold_gives_way_to_a_speaker_count(request_for):
    """The service takes one or the other and refuses a request carrying both.

    The count is the better of the two, because somebody listened to the
    recording and said how many voices were in it, so the threshold is what
    gets dropped.
    """
    client = FakeClient(transcription(SAMPLE_WORDS))
    provider = build_provider(client, diarisation_threshold=0.31)

    provider.transcribe(request_for(diarise=True, expected_speaker_count=4))

    call = client.speech_to_text.calls[0]
    assert call["num_speakers"] == 4
    assert "diarization_threshold" not in call


def test_the_diarisation_threshold_is_not_sent_without_diarisation(request_for):
    """The service rejects the threshold outright on a request that does not diarise."""
    client = FakeClient(transcription(SAMPLE_WORDS))
    provider = build_provider(client, diarisation_threshold=0.31)

    provider.transcribe(request_for(diarise=False, expected_speaker_count=1))

    assert "diarization_threshold" not in client.speech_to_text.calls[0]


def test_word_timings_and_audio_events_are_asked_for(request_for):
    client = FakeClient(transcription(SAMPLE_WORDS))
    build_provider(client).transcribe(request_for())

    call = client.speech_to_text.calls[0]
    assert call["model_id"] == "scribe_v2"
    assert call["timestamps_granularity"] == "word"
    assert call["tag_audio_events"] is True
    assert call["diarize"] is True


# -- Provenance ----------------------------------------------------------


def test_the_record_names_the_model_sent_and_the_transcription_returned(request_for):
    """The response has no model field, so the pair is the only handle there is."""
    client = FakeClient(transcription(SAMPLE_WORDS, identifier="tr-9988"))
    result = build_provider(client, model="scribe_v2").transcribe(request_for())

    record = result.request
    assert record is not None
    assert record.model_identifier == "scribe_v2"
    assert record.request_parameters["response_transcription_id"] == "tr-9988"
    assert record.request_parameters["response_language_probability"] == pytest.approx(0.99)
    assert record.succeeded is True
    assert record.provider is Provider.ELEVENLABS
    assert record.chunks[0].canonical_offset == pytest.approx(OFFSET)


def test_how_much_vocabulary_was_sent_survives_the_credential_filter(request_for):
    """The count is recorded, and under a name the filter does not eat.

    The filter that keeps credentials out of the record drops any parameter
    whose name contains "key". A count recorded as "keyterm_count" therefore
    passed that test and was thrown away before anything was written, so a
    folder said nothing at all about how much vocabulary the request carried.
    """
    client = FakeClient(transcription(SAMPLE_WORDS))
    result = build_provider(client).transcribe(
        request_for(vocabulary_terms=("Vermeulen", "Schmidt"))
    )

    assert result.request.request_parameters["vocabulary_term_count"] == 2


def test_the_whole_answer_is_carried_back_and_not_rebuilt(request_for):
    """The response is kept as the service sent it, not as we understood it.

    Only the original can tell a changed service from a changed application
    when a later run disagrees with this one. A copy assembled from the
    fields this adapter happens to read today would have dropped exactly the
    part that a future question turned out to be about, which is why the
    library's own conversion is used and the answer is checked for a field
    the adapter never looks at.
    """
    response = transcription(
        SAMPLE_WORDS,
        extra={"entities": [{"text": "Suzanne", "type": "person"}], "a_field_we_ignore": 42},
    )
    client = FakeClient(response)

    result = build_provider(client).transcribe(request_for())

    assert result.raw_response is response.model_dump()
    assert result.raw_response["a_field_we_ignore"] == 42
    assert result.raw_response["entities"] == [{"text": "Suzanne", "type": "person"}]
    assert result.raw_response["words"] is SAMPLE_WORDS


def test_the_body_of_a_refusal_is_kept_as_well(request_for):
    """What a service said when it said no is often the most useful thing there."""
    body = {"detail": {"code": "invalid_model", "message": "Unknown model", "status": 400}}
    failure = RuntimeError("Unknown model")
    failure.status_code = 400
    failure.body = body
    client = FakeClient(error=failure)

    result = build_provider(client).transcribe(request_for())

    assert not result.succeeded
    assert result.raw_response is body


def test_no_credential_ever_reaches_the_record(request_for):
    """A transcript folder is copied around, so a key in it would be a leak."""
    client = FakeClient(transcription(SAMPLE_WORDS))
    result = build_provider(client).transcribe(
        request_for(extra_parameters={"api_key": API_KEY, "auth_token": API_KEY, "seed": 3})
    )

    written = repr(result.request)
    assert API_KEY not in written
    assert "api_key" not in result.request.request_parameters["extra_parameters"]
    assert "auth_token" not in result.request.request_parameters["extra_parameters"]
    assert result.request.request_parameters["extra_parameters"]["seed"] == 3


# -- Failing well --------------------------------------------------------


def test_a_missing_key_is_reported_before_anything_is_sent(request_for):
    client = FakeClient(transcription(SAMPLE_WORDS))
    provider = build_provider(client, api_key="")

    assert provider.is_configured() is False
    assert provider.describe_configuration_problem() == "no API key has been entered"

    with pytest.raises(ProviderNotConfigured):
        provider._transcribe(request_for())

    result = provider.transcribe(request_for())
    assert not result.succeeded
    assert "Settings" in result.error
    assert client.speech_to_text.calls == []


def test_a_missing_package_disables_only_this_service(request_for, monkeypatch):
    """The library is optional, and its absence is a fact, not a crash."""
    monkeypatch.setitem(sys.modules, "elevenlabs", None)
    provider = build_provider(client=None)

    with pytest.raises(ProviderUnavailable):
        provider._transcribe(request_for())

    result = provider.transcribe(request_for())
    assert not result.succeeded
    assert "pip install elevenlabs" in result.error


def test_the_module_imports_with_no_package_installed(monkeypatch):
    """Importing must never need the library, or the application cannot start."""
    monkeypatch.setitem(sys.modules, "elevenlabs", None)
    reloaded = importlib.reload(elevenlabs_module)

    assert reloaded.DEFAULT_MODEL == "scribe_v2"
    assert reloaded.ElevenLabsProvider(api_key="k").model_identifier == "scribe_v2"


def test_a_service_error_becomes_a_failed_result(request_for):
    """Losing a service costs accuracy. It must never cost the transcript."""
    failure = RuntimeError("Too many concurrent requests")
    failure.status_code = 429
    failure.body = {"detail": {"code": "concurrent_limit_exceeded", "message": "Slow down"}}
    client = FakeClient(error=failure)

    result = build_provider(client).transcribe(request_for())

    assert not result.succeeded
    assert "429" in result.error
    assert "Slow down" in result.error
    assert result.tokens == []
    assert result.request is not None
    assert result.request.succeeded is False
    assert result.request.error == result.error


def test_a_file_over_the_limit_is_refused_without_being_uploaded(request_for, monkeypatch):
    """Three gigabytes is the safe reading of two disagreeing vendor pages.

    The size is faked rather than a four-gigabyte file being written, which
    would take longer than the rest of the suite put together.
    """
    monkeypatch.setattr(elevenlabs_module, "_file_size", lambda path: 4 * 1024 * 1024 * 1024)
    client = FakeClient(transcription(SAMPLE_WORDS))

    result = build_provider(client).transcribe(request_for())

    assert not result.succeeded
    assert "chunks" in result.error
    assert client.speech_to_text.calls == []


def test_the_run_can_be_stopped_before_the_request_goes(request_for):
    client = FakeClient(transcription(SAMPLE_WORDS))
    result = build_provider(client).transcribe(request_for(), cancelled=lambda: True)

    assert not result.succeeded
    assert client.speech_to_text.calls == []


# -- Forced alignment ----------------------------------------------------


def alignment(words, loss=0.12):
    return SimpleNamespace(words=words, loss=loss)


def aligned_word(text, start, end, loss=0.1):
    return SimpleNamespace(text=text, start=start, end=end, loss=loss)


def test_alignment_returns_words_on_the_canonical_timeline(audio):
    client = FakeClient(
        alignment=alignment(
            [aligned_word("Hello", 0.0, 0.4), aligned_word("world", 0.4, 0.9)]
        )
    )
    aligner = ElevenLabsForcedAligner(api_key=API_KEY, client=client)

    words = aligner.align(audio, "Hello world", Language.ENGLISH, canonical_offset=OFFSET)

    assert words == [
        ("Hello", OFFSET + 0.0, OFFSET + 0.4),
        ("world", OFFSET + 0.4, OFFSET + 0.9),
    ]
    sent = client.forced_alignment.calls[0]
    assert sent["text"] == "Hello world"
    # There is no language parameter on this endpoint at all; the text
    # carries the language.
    assert "language" not in sent and "language_code" not in sent


def test_afrikaans_is_refused_by_the_aligner(audio):
    """It is not one of the 29 languages, which is why the fallback exists."""
    client = FakeClient(alignment=alignment([aligned_word("Hallo", 0.0, 0.4)]))
    aligner = ElevenLabsForcedAligner(api_key=API_KEY, client=client)

    assert aligner.supports(Language.ENGLISH) is True
    assert aligner.supports(Language.GERMAN) is True
    assert aligner.supports(Language.AFRIKAANS) is False

    with pytest.raises(ProviderError):
        aligner.align(audio, "Hallo wêreld", Language.AFRIKAANS)
    assert client.forced_alignment.calls == []


def test_the_aligner_needs_a_key_of_its_own(audio):
    aligner = ElevenLabsForcedAligner(api_key="")

    assert aligner.is_configured() is False
    with pytest.raises(ProviderNotConfigured):
        aligner.align(audio, "Hello world", Language.ENGLISH)


def test_an_alignment_failure_is_raised_rather_than_guessed_at(audio):
    """The caller must keep the timing it had, not invent a new one."""
    client = FakeClient(alignment=alignment([]))
    aligner = ElevenLabsForcedAligner(api_key=API_KEY, client=client)

    with pytest.raises(ProviderError):
        aligner.align(audio, "Hello world", Language.ENGLISH)


def test_the_adapters_are_reachable_from_the_package():
    assert providers.__doc__  # the package explains itself
    assert ElevenLabsProvider(api_key="k").provider is Provider.ELEVENLABS
    assert ElevenLabsForcedAligner(api_key="k").provider is Provider.ELEVENLABS
