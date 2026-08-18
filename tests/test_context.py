"""Tests for the context package and the shape each service is sent."""

from __future__ import annotations

from audio_transcriber.transcription.context import (
    ASSEMBLYAI_MAXIMUM_KEYTERM_WORDS,
    ASSEMBLYAI_MAXIMUM_KEYTERMS_UNIVERSAL_2,
    ASSEMBLYAI_MAXIMUM_KEYTERMS_UNIVERSAL_3_5_PRO,
    ASSEMBLYAI_MAXIMUM_PROMPT_WORDS,
    ASSEMBLYAI_UNIVERSAL_2,
    ASSEMBLYAI_UNIVERSAL_3_5_PRO,
    DEEPGRAM_ESTIMATED_TOKENS_PER_WORD,
    DEEPGRAM_MAXIMUM_KEYTERM_TOKENS,
    ELEVENLABS_MAXIMUM_KEYTERM_CHARACTERS,
    ELEVENLABS_MAXIMUM_KEYTERM_WORDS,
    ELEVENLABS_MAXIMUM_KEYTERMS,
    LANGUAGES_WITH_AFRIKAANS,
    LANGUAGES_WITHOUT_AFRIKAANS,
    MICROSOFT_MAXIMUM_PHRASES,
    OPENAI_MAXIMUM_KEYWORDS,
    OPENAI_MAXIMUM_PROMPT_CHARACTERS,
    ContextPackage,
    adapt_for,
    build_context_package,
)
from audio_transcriber.transcription.model import Language, Provider, RecordingConfiguration
from audio_transcriber.transcription.providers.base import ProviderCapabilities
from audio_transcriber.transcription.vocabulary import VocabularyTerm


def terms(*texts: str) -> list[VocabularyTerm]:
    return [VocabularyTerm(text=text) for text in texts]


def many_terms(count: int, prefix: str = "Term") -> list[VocabularyTerm]:
    return [VocabularyTerm(text=f"{prefix}{number:05d}") for number in range(count)]


def configuration(**changes) -> RecordingConfiguration:
    settings = RecordingConfiguration(
        expected_speaker_count=2,
        known_speakers=["Suzanne", "Michael"],
        recording_context="A quarterly review with the client.",
    )
    for name, value in changes.items():
        setattr(settings, name, value)
    return settings


# -- Whether Afrikaans is enabled ----------------------------------------


def test_with_afrikaans_disabled_only_english_and_german_are_allowed():
    package = build_context_package(configuration(afrikaans_enabled=False))

    assert package.languages == LANGUAGES_WITHOUT_AFRIKAANS
    assert package.languages == (Language.ENGLISH, Language.GERMAN)
    assert package.language_codes == ("en", "de")
    assert package.afrikaans_enabled is False


def test_with_afrikaans_disabled_nothing_the_services_are_sent_mentions_it():
    package = build_context_package(
        configuration(afrikaans_enabled=False),
        terms("Kubernetes", "Acme Holdings"),
    )

    openai = adapt_for(package, Provider.OPENAI)
    assemblyai = adapt_for(package, Provider.ASSEMBLYAI, model=ASSEMBLYAI_UNIVERSAL_3_5_PRO)

    assert "afrikaans" not in package.language_sentence.lower()
    assert "afrikaans" not in openai.prompt.lower()
    assert "afrikaans" not in assemblyai.prompt.lower()
    assert openai.parameters["languages"] == ["en", "de"]
    assert Language.AFRIKAANS not in package.languages


def test_with_afrikaans_disabled_its_own_terms_are_left_out():
    """Sending an Afrikaans name is still asking a service to listen for Afrikaans."""
    package = build_context_package(
        configuration(afrikaans_enabled=False),
        [
            VocabularyTerm(text="Vermeulen", language=Language.AFRIKAANS),
            VocabularyTerm(text="Schmidt", language=Language.GERMAN),
            VocabularyTerm(text="Holdings"),
        ],
    )

    assert package.term_texts == ("Schmidt", "Holdings")


def test_with_afrikaans_enabled_all_three_languages_are_allowed():
    package = build_context_package(
        configuration(afrikaans_enabled=True),
        [VocabularyTerm(text="Vermeulen", language=Language.AFRIKAANS)],
    )

    assert package.languages == LANGUAGES_WITH_AFRIKAANS
    assert package.languages == (Language.ENGLISH, Language.GERMAN, Language.AFRIKAANS)
    assert package.language_codes == ("en", "de", "af")
    assert package.afrikaans_enabled is True
    assert package.term_texts == ("Vermeulen",)


def test_with_afrikaans_enabled_openai_is_given_the_third_code():
    package = build_context_package(configuration(afrikaans_enabled=True))

    written = adapt_for(package, Provider.OPENAI)

    assert written.parameters["languages"] == ["en", "de", "af"]
    assert written.languages == ("en", "de", "af")


def test_with_afrikaans_enabled_a_service_taking_prose_is_told_so():
    package = build_context_package(configuration(afrikaans_enabled=True))

    written = adapt_for(package, Provider.ASSEMBLYAI, model=ASSEMBLYAI_UNIVERSAL_3_5_PRO)

    assert "Afrikaans" in package.language_sentence
    assert "Afrikaans" in written.prompt


# -- What the package says -----------------------------------------------


def test_the_package_carries_what_the_user_told_us():
    package = build_context_package(configuration(), terms("Acme Holdings"))

    assert package.speaker_names == ("Suzanne", "Michael")
    assert package.expected_speaker_count == 2
    assert package.recording_context == "A quarterly review with the client."
    assert package.term_texts == ("Acme Holdings",)


def test_the_speakers_are_named_where_they_are_known():
    package = build_context_package(configuration())

    assert package.speaker_sentence == "The speakers are Suzanne and Michael."


def test_the_number_of_speakers_is_given_where_the_names_are_not():
    package = build_context_package(configuration(known_speakers=[], expected_speaker_count=3))

    assert package.speaker_sentence == "There are about 3 speakers."


def test_one_unnamed_speaker_needs_no_sentence_at_all():
    package = build_context_package(configuration(known_speakers=[], expected_speaker_count=1))

    assert package.speaker_sentence == ""


# -- The shape each service is sent --------------------------------------


def test_elevenlabs_is_sent_keyterms_and_no_prose():
    package = build_context_package(configuration(), terms("Vermeulen", "Acme Holdings"))

    written = adapt_for(package, Provider.ELEVENLABS)

    assert written.provider is Provider.ELEVENLABS
    assert written.parameters == {"keyterms": ["Vermeulen", "Acme Holdings"]}
    assert written.terms == ("Vermeulen", "Acme Holdings")
    assert written.prompt == ""


def test_openai_is_sent_its_three_channels_separately():
    """The terms have a parameter of their own now, so they stay out of the prose."""
    package = build_context_package(configuration(), terms("Vermeulen"))

    written = adapt_for(package, Provider.OPENAI)

    assert sorted(written.parameters) == ["keywords", "languages", "prompt"]
    assert written.parameters["keywords"] == ["Vermeulen"]
    assert written.parameters["languages"] == ["en", "de"]
    assert written.parameters["prompt"] == written.prompt
    assert "The speakers are Suzanne and Michael." in written.prompt
    assert "A quarterly review with the client." in written.prompt
    assert "Vermeulen" not in written.prompt


def test_the_openai_prompt_leaves_the_languages_to_their_own_parameter():
    package = build_context_package(configuration())

    written = adapt_for(package, Provider.OPENAI)

    assert "The recording is in" not in written.prompt


def test_microsoft_is_sent_a_phrase_list_and_never_any_prose():
    """Prompt tuning is unsupported there, so a prompt would be refused, not ignored."""
    package = build_context_package(configuration(), terms("Vermeulen"))

    written = adapt_for(package, Provider.MICROSOFT)

    assert written.parameters == {"phrases": ["Vermeulen"]}
    assert written.prompt == ""


def test_assemblyai_sets_keyterms_prompt():
    package = build_context_package(configuration(), terms("Vermeulen"))

    written = adapt_for(package, Provider.ASSEMBLYAI)

    assert written.parameters["keyterms_prompt"] == ["Vermeulen"]
    assert "word_boost" not in written.parameters
    assert written.terms == ("Vermeulen",)


def test_deepgram_is_sent_keyterms_under_its_own_name_and_no_prose():
    package = build_context_package(configuration(), terms("Vermeulen"))

    written = adapt_for(package, Provider.DEEPGRAM)

    assert written.parameters == {"keyterm": ["Vermeulen"]}
    assert written.prompt == ""


def test_a_package_built_by_hand_needs_only_its_terms():
    """The pipeline is not the only caller; a bare package must still adapt."""
    package = ContextPackage(terms=tuple(terms("Vermeulen")))

    written = adapt_for(package, Provider.ELEVENLABS)

    assert written.terms == ("Vermeulen",)
    assert written.parameters == {"keyterms": ["Vermeulen"]}


# -- The limits ----------------------------------------------------------


def test_elevenlabs_stops_below_the_surcharge_threshold():
    """The cap is a cost decision, not a capability one, and it has to bite."""
    package = build_context_package(configuration(), many_terms(ELEVENLABS_MAXIMUM_KEYTERMS + 25))

    written = adapt_for(package, Provider.ELEVENLABS)

    assert len(written.terms) == ELEVENLABS_MAXIMUM_KEYTERMS
    assert written.dropped_term_count == 25


def test_the_terms_that_are_cut_are_the_ones_at_the_end():
    """The list arrives most valuable first, so the tail is what may go."""
    wanted = many_terms(ELEVENLABS_MAXIMUM_KEYTERMS + 5)
    package = build_context_package(configuration(), wanted)

    written = adapt_for(package, Provider.ELEVENLABS)

    assert list(written.terms) == [found.text for found in wanted[:ELEVENLABS_MAXIMUM_KEYTERMS]]


def test_a_term_too_long_for_a_service_is_dropped_whole():
    """Half a surname is a different word, not a shorter version of that surname."""
    long_term = "V" * (ELEVENLABS_MAXIMUM_KEYTERM_CHARACTERS + 1)
    package = build_context_package(configuration(), terms("Vermeulen", long_term, "Schmidt"))

    written = adapt_for(package, Provider.ELEVENLABS)

    assert written.terms == ("Vermeulen", "Schmidt")
    assert written.dropped_term_count == 1


def test_a_phrase_of_too_many_words_is_dropped_rather_than_shortened():
    wordy = " ".join(["word"] * (ELEVENLABS_MAXIMUM_KEYTERM_WORDS + 1))
    just_short_enough = " ".join(["word"] * ELEVENLABS_MAXIMUM_KEYTERM_WORDS)
    package = build_context_package(configuration(), terms("Vermeulen", wordy, just_short_enough))

    written = adapt_for(package, Provider.ELEVENLABS)

    assert written.terms == ("Vermeulen", just_short_enough)
    assert written.dropped_term_count == 1


def test_openai_takes_only_as_many_keywords_as_we_will_send_it():
    package = build_context_package(configuration(), many_terms(OPENAI_MAXIMUM_KEYWORDS + 30))

    written = adapt_for(package, Provider.OPENAI)

    assert len(written.parameters["keywords"]) == OPENAI_MAXIMUM_KEYWORDS
    assert written.dropped_term_count == 30


def test_an_enormous_description_is_cut_at_a_word_boundary():
    package = build_context_package(configuration(recording_context="elephant " * 400))

    written = adapt_for(package, Provider.OPENAI)

    assert len(written.prompt) <= OPENAI_MAXIMUM_PROMPT_CHARACTERS
    assert written.prompt.endswith("elephant")


def test_a_long_vocabulary_no_longer_crowds_out_the_description():
    """This is what the three separate channels bought us."""
    package = build_context_package(configuration(), many_terms(2000))

    written = adapt_for(package, Provider.OPENAI)

    assert "A quarterly review with the client." in written.prompt
    assert len(written.parameters["keywords"]) == OPENAI_MAXIMUM_KEYWORDS


def test_microsoft_stops_at_its_phrase_limit():
    package = build_context_package(configuration(), many_terms(MICROSOFT_MAXIMUM_PHRASES + 10))

    written = adapt_for(package, Provider.MICROSOFT)

    assert len(written.parameters["phrases"]) == MICROSOFT_MAXIMUM_PHRASES
    assert written.dropped_term_count == 10


def test_assemblyai_uses_the_smaller_cap_when_the_model_is_not_said():
    """Being refused for sending too much is worse than sending less than we could."""
    package = build_context_package(
        configuration(), many_terms(ASSEMBLYAI_MAXIMUM_KEYTERMS_UNIVERSAL_3_5_PRO + 10)
    )

    written = adapt_for(package, Provider.ASSEMBLYAI)

    assert len(written.parameters["keyterms_prompt"]) == ASSEMBLYAI_MAXIMUM_KEYTERMS_UNIVERSAL_2
    assert written.prompt == ""
    assert "prompt" not in written.parameters


def test_assemblyai_uses_the_larger_cap_on_the_newer_model():
    package = build_context_package(
        configuration(), many_terms(ASSEMBLYAI_MAXIMUM_KEYTERMS_UNIVERSAL_3_5_PRO + 10)
    )

    written = adapt_for(package, Provider.ASSEMBLYAI, model=ASSEMBLYAI_UNIVERSAL_3_5_PRO)

    assert (
        len(written.parameters["keyterms_prompt"])
        == ASSEMBLYAI_MAXIMUM_KEYTERMS_UNIVERSAL_3_5_PRO
    )


def test_assemblyai_keeps_the_smaller_cap_on_the_afrikaans_model():
    package = build_context_package(
        configuration(afrikaans_enabled=True),
        many_terms(ASSEMBLYAI_MAXIMUM_KEYTERMS_UNIVERSAL_3_5_PRO + 10),
    )

    written = adapt_for(package, Provider.ASSEMBLYAI, model=ASSEMBLYAI_UNIVERSAL_2)

    assert len(written.parameters["keyterms_prompt"]) == ASSEMBLYAI_MAXIMUM_KEYTERMS_UNIVERSAL_2
    assert written.prompt == ""


def test_an_assemblyai_model_name_is_recognised_however_it_is_spelled():
    package = build_context_package(
        configuration(), many_terms(ASSEMBLYAI_MAXIMUM_KEYTERMS_UNIVERSAL_2 + 5)
    )

    for spelling in ("Universal-3.5-Pro", "universal_3_5_pro", "universal 3.5 pro"):
        written = adapt_for(package, Provider.ASSEMBLYAI, model=spelling)
        assert len(written.terms) > ASSEMBLYAI_MAXIMUM_KEYTERMS_UNIVERSAL_2

    unknown = adapt_for(package, Provider.ASSEMBLYAI, model="something-else")
    assert len(unknown.terms) == ASSEMBLYAI_MAXIMUM_KEYTERMS_UNIVERSAL_2


def test_the_assemblyai_prompt_is_cut_by_words_rather_than_characters():
    package = build_context_package(configuration(recording_context="elephant " * 3000))

    written = adapt_for(package, Provider.ASSEMBLYAI, model=ASSEMBLYAI_UNIVERSAL_3_5_PRO)

    assert len(written.prompt.split()) == ASSEMBLYAI_MAXIMUM_PROMPT_WORDS
    assert written.prompt.endswith("elephant")


def test_an_assemblyai_phrase_of_too_many_words_is_dropped():
    wordy = " ".join(["word"] * (ASSEMBLYAI_MAXIMUM_KEYTERM_WORDS + 1))
    package = build_context_package(configuration(), terms("Vermeulen", wordy))

    written = adapt_for(package, Provider.ASSEMBLYAI)

    assert written.parameters["keyterms_prompt"] == ["Vermeulen"]
    assert written.dropped_term_count == 1


def test_deepgram_is_filled_to_a_token_budget_rather_than_a_count():
    """Going over the budget is an error there, not a quiet truncation."""
    affordable = DEEPGRAM_MAXIMUM_KEYTERM_TOKENS // DEEPGRAM_ESTIMATED_TOKENS_PER_WORD
    package = build_context_package(configuration(), many_terms(affordable + 20))

    written = adapt_for(package, Provider.DEEPGRAM)

    assert len(written.terms) == affordable
    assert written.dropped_term_count == 20


def test_a_deepgram_budget_of_long_phrases_runs_out_sooner_than_one_of_names():
    names = build_context_package(configuration(), many_terms(200))
    phrases = build_context_package(
        configuration(), [VocabularyTerm(text=f"a longer phrase {number}") for number in range(200)]
    )

    with_names = adapt_for(names, Provider.DEEPGRAM)
    with_phrases = adapt_for(phrases, Provider.DEEPGRAM)

    assert len(with_phrases.terms) < len(with_names.terms)
    assert with_phrases.dropped_term_count > with_names.dropped_term_count


def test_a_deepgram_list_that_fits_is_left_alone():
    package = build_context_package(configuration(), terms("Vermeulen", "Schmidt"))

    written = adapt_for(package, Provider.DEEPGRAM)

    assert written.terms == ("Vermeulen", "Schmidt")
    assert written.dropped_term_count == 0


# -- What a service says it can do ---------------------------------------


def test_a_service_that_takes_no_vocabulary_is_sent_none():
    package = build_context_package(configuration(), terms("Vermeulen", "Schmidt"))
    capabilities = ProviderCapabilities(vocabulary_biasing=False, context_prompt=True)

    written = adapt_for(package, Provider.OPENAI, capabilities)

    assert written.terms == ()
    assert written.parameters["keywords"] == []
    assert written.dropped_term_count == 2
    assert "A quarterly review with the client." in written.prompt


def test_a_service_that_takes_no_prompt_is_sent_none():
    package = build_context_package(configuration(), terms("Vermeulen"))
    capabilities = ProviderCapabilities(vocabulary_biasing=True, context_prompt=False)

    written = adapt_for(package, Provider.OPENAI, capabilities)

    assert written.prompt == ""
    assert "prompt" not in written.parameters
    assert written.parameters["keywords"] == ["Vermeulen"]
    assert written.parameters["languages"] == ["en", "de"]


def test_a_service_that_takes_both_is_sent_both():
    package = build_context_package(configuration(), terms("Vermeulen"))
    capabilities = ProviderCapabilities(vocabulary_biasing=True, context_prompt=True)

    written = adapt_for(package, Provider.ELEVENLABS, capabilities)

    assert written.parameters == {"keyterms": ["Vermeulen"]}
