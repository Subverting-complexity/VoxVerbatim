"""Tests for working out the language of a span from weak evidence.

The tables and results here are built by hand rather than fetched, so the
tests say exactly what each service reported and nothing is hidden inside a
fixture. The most important of them is the one proving that no Afrikaans
reasoning happens at all when the user has said the recording contains none.
"""

from __future__ import annotations

from vox_verbatim.transcription.alignment import build_aligned_table
from vox_verbatim.transcription.language import (
    CONFIDENT_LANGUAGE_THRESHOLD,
    MICROSOFT_MINIMUM_WEIGHT,
    LanguageWeights,
    SpanLanguage,
    allowed_languages,
    lexical_language,
    provider_language_weight,
    read_languages,
)
from vox_verbatim.transcription.model import (
    AudioSpan,
    Language,
    LanguageEvidence,
    Provider,
    ProviderResult,
    ProviderToken,
    RecordingConfiguration,
)


def words(
    provider: Provider,
    texts: list[str],
    language: Language = Language.UNKNOWN,
    languages: list[Language] | None = None,
    start: float = 0.0,
    step: float = 0.5,
) -> list[ProviderToken]:
    """One service's words, timed one after another."""
    built: list[ProviderToken] = []
    for index, text in enumerate(texts):
        built.append(
            ProviderToken(
                provider=provider,
                index=index,
                text=text,
                start=start + index * step,
                end=start + index * step + step * 0.8,
                language=languages[index] if languages else language,
            )
        )
    return built


def result(
    provider: Provider,
    texts: list[str],
    detected: Language = Language.UNKNOWN,
    languages: list[Language] | None = None,
) -> ProviderResult:
    return ProviderResult(
        provider=provider,
        tokens=words(provider, texts, detected, languages),
        detected_language=detected,
    )


ENGLISH_SENTENCE = ["we", "should", "talk", "about", "the", "contract", "today"]


def test_afrikaans_is_not_allowed_unless_the_user_enabled_it() -> None:
    assert allowed_languages(RecordingConfiguration()) == (Language.ENGLISH, Language.GERMAN)
    enabled = RecordingConfiguration(afrikaans_enabled=True)
    assert Language.AFRIKAANS in allowed_languages(enabled)


def test_no_afrikaans_score_is_produced_when_afrikaans_is_disabled() -> None:
    """The requirement from section 3, tested as a requirement rather than a filter.

    The recording is full of words the Afrikaans lexicon would recognise and
    two services claim to have detected Afrikaans. None of it may produce an
    Afrikaans score anywhere, because the user has said there is no Afrikaans
    in this recording and the point is to remove the guess, not to make it
    and then discard it.
    """
    afrikaans_words = ["ek", "dink", "ons", "moet", "nou", "praat"]
    backbone = result(Provider.ELEVENLABS, afrikaans_words, Language.AFRIKAANS)
    others = [
        result(Provider.OPENAI, afrikaans_words, Language.AFRIKAANS),
        result(Provider.MICROSOFT, afrikaans_words, Language.GERMAN,
               languages=[Language.GERMAN] * len(afrikaans_words)),
    ]
    table = build_aligned_table(backbone, others)

    reading = read_languages(table, [backbone, *others], RecordingConfiguration())

    assert Language.AFRIKAANS not in reading.allowed
    assert Language.AFRIKAANS not in reading.prior.scores
    for evidence in reading.by_position:
        assert Language.AFRIKAANS not in evidence.scores
    for position in range(len(afrikaans_words)):
        assert reading.language_at(position) is not Language.AFRIKAANS
        assert not reading.is_uncertain(position)


def test_the_afrikaans_lexicon_is_not_consulted_when_afrikaans_is_disabled() -> None:
    without = (Language.ENGLISH, Language.GERMAN)
    with_afrikaans = (Language.ENGLISH, Language.GERMAN, Language.AFRIKAANS)
    assert lexical_language("ek", without) is None
    assert lexical_language("ek", with_afrikaans) is Language.AFRIKAANS


def test_number_words_are_kept_out_of_the_lexicon() -> None:
    # "een" is Afrikaans for one and compares equal to the digit, so a
    # lexicon holding it would call every numeral in the recording Afrikaans.
    every = (Language.ENGLISH, Language.GERMAN, Language.AFRIKAANS)
    assert lexical_language("een", every) is None
    assert lexical_language("1", every) is None


def test_the_whole_file_readings_become_the_prior() -> None:
    backbone = result(Provider.ELEVENLABS, ENGLISH_SENTENCE, Language.ENGLISH)
    others = [result(Provider.OPENAI, ENGLISH_SENTENCE, Language.ENGLISH)]
    table = build_aligned_table(backbone, others)

    reading = read_languages(table, [backbone, *others], RecordingConfiguration())

    assert reading.prior.best is Language.ENGLISH
    assert reading.language_at(0) is Language.ENGLISH


def test_a_detected_language_outside_the_allowed_set_is_ignored() -> None:
    backbone = result(Provider.ELEVENLABS, ENGLISH_SENTENCE, Language.ENGLISH)
    strange = result(Provider.OPENAI, ENGLISH_SENTENCE, Language.UNKNOWN)
    table = build_aligned_table(backbone, [strange])

    reading = read_languages(table, [backbone, strange], RecordingConfiguration())

    assert reading.prior.score_for(Language.ENGLISH) == 1.0


def test_a_microsoft_locale_carries_a_span_against_the_prior() -> None:
    """Microsoft is the only per-phrase language signal, and it is weighted as such."""
    texts = ["and", "then", "kein", "problem", "for", "us"]
    backbone = result(Provider.ELEVENLABS, texts, Language.ENGLISH)
    microsoft = result(
        Provider.MICROSOFT,
        texts,
        Language.ENGLISH,
        languages=[
            Language.ENGLISH,
            Language.ENGLISH,
            Language.GERMAN,
            Language.GERMAN,
            Language.ENGLISH,
            Language.ENGLISH,
        ],
    )
    table = build_aligned_table(backbone, [microsoft])

    reading = read_languages(table, [backbone, microsoft], RecordingConfiguration())

    assert reading.language_at(3) is Language.GERMAN
    assert reading.language_at(0) is Language.ENGLISH


def test_a_word_only_one_language_has_is_evidence_for_that_language() -> None:
    texts = ["das", "ist", "wirklich", "nicht", "einfach"]
    backbone = result(Provider.ELEVENLABS, texts)
    table = build_aligned_table(backbone, [])

    reading = read_languages(table, [backbone], RecordingConfiguration())

    assert reading.language_at(3) is Language.GERMAN


def test_evidence_reaches_the_words_around_it() -> None:
    texts = ["hallo", "wirklich", "gut", "so", "ja"]
    backbone = result(Provider.ELEVENLABS, texts)
    table = build_aligned_table(backbone, [])

    reading = read_languages(table, [backbone], RecordingConfiguration())

    # Nothing at the last word says anything, but the German a few words
    # earlier does, because a language does not stop between two words.
    assert reading.language_at(4) is Language.GERMAN


def test_an_escalation_reading_applies_to_the_window_it_was_taken_from() -> None:
    texts = ["so", "then", "we", "spoke", "and", "toe", "sê", "hy", "yes", "fine"]
    backbone = result(Provider.ELEVENLABS, texts, Language.ENGLISH)
    table = build_aligned_table(backbone, [])
    # AssemblyAI was sent the window around the fifth and sixth words and
    # reported what it heard there, which is a reading about that region
    # rather than about the recording.
    window = SpanLanguage(
        span=AudioSpan(2.4, 3.4),
        language=Language.AFRIKAANS,
        provider=Provider.ASSEMBLYAI,
    )

    reading = read_languages(
        table,
        [backbone],
        RecordingConfiguration(afrikaans_enabled=True),
        escalation_readings=[window],
    )

    assert reading.language_at(5) is Language.AFRIKAANS
    assert reading.language_at(0) is Language.ENGLISH
    inside = reading.for_position(5).score_for(Language.AFRIKAANS)
    outside = reading.for_position(0).score_for(Language.AFRIKAANS)
    assert inside > outside


# -- Who may weigh in on which span --------------------------------------


def test_microsoft_is_excluded_from_a_confidently_afrikaans_span() -> None:
    evidence = LanguageEvidence({Language.AFRIKAANS: 0.9, Language.ENGLISH: 0.1})
    assert evidence.is_confidently(Language.AFRIKAANS, CONFIDENT_LANGUAGE_THRESHOLD)
    assert provider_language_weight(Provider.MICROSOFT, evidence, True) == 0.0


def test_microsoft_is_only_reduced_where_the_language_is_uncertain() -> None:
    evidence = LanguageEvidence({Language.ENGLISH: 0.5, Language.AFRIKAANS: 0.5})
    weight = provider_language_weight(Provider.MICROSOFT, evidence, True)
    assert 0.0 < weight < 1.0
    assert weight >= MICROSOFT_MINIMUM_WEIGHT


def test_microsofts_weight_falls_gradually_rather_than_at_a_step() -> None:
    weights = [
        provider_language_weight(
            Provider.MICROSOFT,
            LanguageEvidence({Language.ENGLISH: 1.0 - share, Language.AFRIKAANS: share}),
            True,
        )
        for share in (0.1, 0.3, 0.5, 0.7)
    ]
    assert weights == sorted(weights, reverse=True)
    assert len(set(weights)) == len(weights)


def test_microsoft_keeps_full_weight_when_afrikaans_is_disabled() -> None:
    # Section 3: with Afrikaans disabled, Microsoft is valid wherever
    # English or German evidence is relevant, which is everywhere.
    evidence = LanguageEvidence({Language.ENGLISH: 0.5, Language.GERMAN: 0.5})
    assert provider_language_weight(Provider.MICROSOFT, evidence, False) == 1.0


def test_the_other_services_are_never_reduced_for_language() -> None:
    evidence = LanguageEvidence({Language.AFRIKAANS: 1.0})
    for provider in (Provider.ELEVENLABS, Provider.OPENAI, Provider.ASSEMBLYAI):
        assert provider_language_weight(provider, evidence, True) == 1.0


# -- Uncertainty as a review reason --------------------------------------


def test_an_even_split_is_uncertain_only_when_afrikaans_is_enabled() -> None:
    texts = ["yes", "ja", "okay", "so"]
    backbone = result(Provider.ELEVENLABS, texts)
    table = build_aligned_table(backbone, [])

    disabled = read_languages(table, [backbone], RecordingConfiguration())
    enabled = read_languages(
        table, [backbone], RecordingConfiguration(afrikaans_enabled=True)
    )

    assert not disabled.is_uncertain(0)
    assert enabled.is_uncertain(0)


def test_weights_can_be_replaced_without_touching_the_module() -> None:
    texts = ["and", "kein", "problem"]
    backbone = result(Provider.ELEVENLABS, texts, Language.ENGLISH)
    microsoft = result(
        Provider.MICROSOFT,
        texts,
        Language.ENGLISH,
        languages=[Language.ENGLISH, Language.GERMAN, Language.GERMAN],
    )
    table = build_aligned_table(backbone, [microsoft])
    silenced = LanguageWeights(microsoft_locale=0.0, lexical=0.0)

    reading = read_languages(
        table, [backbone, microsoft], RecordingConfiguration(), weights=silenced
    )

    assert reading.language_at(1) is Language.ENGLISH


def test_a_position_outside_the_recording_falls_back_rather_than_raising() -> None:
    backbone = result(Provider.ELEVENLABS, ["one"], Language.ENGLISH)
    table = build_aligned_table(backbone, [])

    reading = read_languages(table, [backbone], RecordingConfiguration())

    assert reading.for_position(-4).best is Language.ENGLISH
    assert reading.for_position(99).best is Language.ENGLISH
