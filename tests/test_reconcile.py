"""Tests for deciding what was actually said.

Every table here is built by hand from provider results, so each test states
exactly what each service reported. Nothing reaches the network and nothing
reads a recording.

The tests that matter most are the ones that pin down behaviour the
specification is emphatic about: that formatting differences are not
disagreements, that the vocabulary is real evidence in both directions, that
scoring never collapses into counting heads, that a high-risk value in
dispute is never settled, and that no Afrikaans reasoning happens when
Afrikaans is disabled.
"""

from __future__ import annotations

import dataclasses

from vox_verbatim.transcription.alignment import build_aligned_table
from vox_verbatim.transcription.confidence import (
    HIGH_CONFIDENCE_THRESHOLD,
    category_for,
)
from vox_verbatim.transcription.model import (
    AlignmentStatus,
    AudioSpan,
    Confidence,
    Language,
    Provider,
    ProviderResult,
    ProviderToken,
    RecordingConfiguration,
    ReviewReason,
    ReviewStatus,
    RiskCategory,
    TimingStatus,
)
from vox_verbatim.transcription.reconcile import (
    DEFAULT_OPTIONS,
    DEFAULT_PROVIDER_RELIABILITY,
    ReconciliationOptions,
    ReliabilityWeights,
    WordFacts,
    decide_timing,
    reconcile,
)
from vox_verbatim.transcription.vocabulary import (
    TermCategory,
    VocabularyIndex,
    VocabularyTerm,
)
from vox_verbatim.transcription.normalise import EquivalenceKind

SENTENCE = ["we", "should", "sign", "the", "contract", "tomorrow"]


def spoken(
    provider: Provider,
    texts: list[str],
    start: float = 0.0,
    step: float = 1.0,
    timed: bool = True,
    speakers: list[str] | None = None,
    log_probabilities: list[float] | None = None,
    confidences: list[float] | None = None,
    languages: list[Language] | None = None,
) -> ProviderResult:
    """One service's whole answer for a recording, timed word by word."""
    tokens: list[ProviderToken] = []
    for index, text in enumerate(texts):
        tokens.append(
            ProviderToken(
                provider=provider,
                index=index,
                text=text,
                start=start + index * step if timed else None,
                end=start + index * step + step * 0.8 if timed else None,
                speaker=speakers[index] if speakers else None,
                log_probability=log_probabilities[index] if log_probabilities else None,
                confidence=confidences[index] if confidences else None,
                language=languages[index] if languages else Language.UNKNOWN,
            )
        )
    return ProviderResult(
        provider=provider,
        tokens=tokens,
        detected_language=Language.ENGLISH,
    )


def table_of(backbone: ProviderResult, *others: ProviderResult):
    return build_aligned_table(backbone, list(others))


def texts_of(tokens) -> list[str]:
    return [token.text for token in tokens]


def token_at(tokens, text: str):
    for token in tokens:
        if token.text == text:
            return token
    raise AssertionError(f"no token said {text!r}: {texts_of(tokens)}")


# -- The three agreement cases -------------------------------------------


def test_exact_agreement_is_accepted_with_high_confidence() -> None:
    table = table_of(
        spoken(Provider.ELEVENLABS, SENTENCE),
        spoken(Provider.OPENAI, SENTENCE, timed=False),
        spoken(Provider.MICROSOFT, SENTENCE, timed=False),
    )

    tokens = reconcile(table)

    assert texts_of(tokens) == SENTENCE
    assert all(token.text_confidence is Confidence.HIGH for token in tokens)
    assert all(token.alignment_status is AlignmentStatus.EXACT for token in tokens)
    assert all(not token.review_reasons for token in tokens)


def test_every_settled_word_carries_the_strength_behind_its_category() -> None:
    """The two saved answers about a word must never contradict each other.

    This case is the one that can break it. Where every service says the
    same thing, the word is accepted as high confidence by rule rather than
    by arithmetic, because two of the three services report no confidence
    figure at all and running unanimity back through the arithmetic would
    mark the word down for that alone. The strength saved beside the
    category has to follow it up, or the review window would leave the word
    alone while the low-confidence sweep pulled the same word up as weak.
    """
    table = table_of(
        spoken(Provider.ELEVENLABS, SENTENCE),
        spoken(Provider.OPENAI, SENTENCE, timed=False),
        spoken(Provider.MICROSOFT, SENTENCE, timed=False),
    )

    tokens = reconcile(table)

    assert all(token.text_confidence is Confidence.HIGH for token in tokens)
    for token in tokens:
        assert token.confidence_strength is not None
        assert category_for(token.confidence_strength) is token.text_confidence


def test_a_disputed_word_saves_the_strength_the_evidence_actually_reached() -> None:
    """The promotion above is a floor, not a habit of writing down the floor.

    A word the services argued about must keep its own number, because the
    whole use of the number is to tell one weak word from another weak word
    and decide which to look at first.
    """
    backbone = spoken(Provider.ELEVENLABS, SENTENCE, confidences=[0.4] * len(SENTENCE))
    other = spoken(
        Provider.OPENAI,
        ["we", "should", "sign", "the", "contact", "tomorrow"],
        timed=False,
    )
    table = table_of(backbone, other)

    disputed = reconcile(table)[4]

    assert disputed.text_confidence is not Confidence.HIGH
    assert disputed.confidence_strength is not None
    assert disputed.confidence_strength < HIGH_CONFIDENCE_THRESHOLD
    assert category_for(disputed.confidence_strength) is disputed.text_confidence


def test_an_exactly_agreed_word_keeps_the_measured_time() -> None:
    table = table_of(
        spoken(Provider.ELEVENLABS, SENTENCE),
        spoken(Provider.OPENAI, SENTENCE, timed=False),
    )

    tokens = reconcile(table)

    assert tokens[0].timing_status is TimingStatus.EXACT_PROVIDER_TIME
    assert tokens[0].timing_source is Provider.ELEVENLABS
    assert tokens[0].start == 0.0


def test_a_formatting_difference_is_not_a_disagreement() -> None:
    backbone = spoken(Provider.ELEVENLABS, ["it", "costs", "twenty", "five", "each"])
    openai = spoken(Provider.OPENAI, ["it", "costs", "25", "each"], timed=False)
    table = table_of(backbone, openai)

    tokens = reconcile(table)

    assert not any(
        ReviewReason.PROVIDER_DISAGREEMENT in token.review_reasons for token in tokens
    )
    assert all(token.text_confidence is Confidence.HIGH for token in tokens)


def test_a_formatting_difference_keeps_the_span_it_was_measured_in() -> None:
    backbone = spoken(Provider.ELEVENLABS, ["ask", "Jurgen", "about", "it"])
    openai = spoken(Provider.OPENAI, ["ask", "Jürgen", "about", "it"], timed=False)
    table = table_of(backbone, openai)

    tokens = reconcile(table)

    # The same sounds are underneath the word, so the measured span still
    # fits and the corrected spelling inherits it.
    chosen = tokens[1]
    assert chosen.text == "Jürgen"
    assert chosen.timing_status is TimingStatus.MAPPED_FORMAT_EQUIVALENT
    assert chosen.start == backbone.tokens[1].start


def test_a_material_disagreement_is_reported_rather_than_hidden() -> None:
    backbone = spoken(Provider.ELEVENLABS, SENTENCE, confidences=[0.4] * len(SENTENCE))
    other = spoken(
        Provider.OPENAI,
        ["we", "should", "sign", "the", "contact", "tomorrow"],
        timed=False,
    )
    table = table_of(backbone, other)

    tokens = reconcile(table)

    disputed = tokens[4]
    assert ReviewReason.PROVIDER_DISAGREEMENT in disputed.review_reasons
    assert disputed.text_confidence is not Confidence.HIGH
    assert {candidate.text for candidate in disputed.candidates} == {"contract", "contact"}


# -- What the user knows ---------------------------------------------------


def _index(*terms: VocabularyTerm) -> VocabularyIndex:
    return VocabularyIndex(terms)


def test_a_known_term_beats_the_services_that_outnumber_it() -> None:
    backbone = spoken(Provider.ELEVENLABS, ["call", "Van", "Niekerk", "back"])
    openai = spoken(Provider.OPENAI, ["call", "van", "Nieker", "back"], timed=False)
    microsoft = spoken(Provider.MICROSOFT, ["call", "van", "Nieker", "back"], timed=False)
    table = table_of(backbone, openai, microsoft)
    vocabulary = _index(VocabularyTerm(text="Niekerk", category=TermCategory.PERSON))

    tokens = reconcile(table, vocabulary=vocabulary)

    assert tokens[2].text == "Niekerk"
    assert tokens[2].candidates[0].in_vocabulary


def test_a_known_misrecognition_is_evidence_against_a_candidate() -> None:
    backbone = spoken(Provider.ELEVENLABS, ["ask", "Mueller", "again"])
    openai = spoken(Provider.OPENAI, ["ask", "Müller", "again"], timed=False)
    table = table_of(backbone, openai)
    vocabulary = _index(
        VocabularyTerm(
            text="Müller",
            category=TermCategory.PERSON,
            common_misrecognitions=("Mueller",),
        )
    )

    tokens = reconcile(table, vocabulary=vocabulary)

    assert tokens[1].text == "Müller"


def test_a_misrecognition_is_scored_below_an_unknown_word() -> None:
    backbone = spoken(Provider.ELEVENLABS, ["the", "Bosh", "system"])
    openai = spoken(Provider.OPENAI, ["the", "Bosch", "system"], timed=False)
    table = table_of(backbone, openai)
    vocabulary = _index(
        VocabularyTerm(text="Bosch", common_misrecognitions=("Bosh",))
    )

    tokens = reconcile(table, vocabulary=vocabulary)
    chosen = tokens[1]
    scores = {candidate.text: candidate.score for candidate in chosen.candidates}

    assert chosen.text == "Bosch"
    assert scores["Bosch"] > scores["Bosh"]


# -- Scoring, and the ban on counting heads --------------------------------


def test_two_weak_services_cannot_outvote_one_strong_one() -> None:
    """The rule section 12 states twice, tested as arithmetic.

    Two services agree with each other against one, and both of the usual
    shortcuts would hand them the word: counting heads gives them two to one,
    and adding the scores up gives them more than the single service has. The
    combination used here does neither, because agreement removes what is
    left of the doubt rather than adding to a total, so it approaches
    certainty without ever passing a strong reading.
    """
    backbone = spoken(
        Provider.ELEVENLABS,
        SENTENCE,
        log_probabilities=[-0.05] * len(SENTENCE),
    )
    weak = ["we", "should", "sign", "the", "contact", "tomorrow"]
    deepgram = spoken(Provider.DEEPGRAM, weak, timed=False, confidences=[0.85] * len(weak))
    assemblyai = spoken(
        Provider.ASSEMBLYAI, weak, timed=False, confidences=[0.85] * len(weak)
    )
    table = table_of(backbone, deepgram, assemblyai)

    tokens = reconcile(table)
    disputed = token_at(tokens, "contract")
    winner = disputed.candidates[0]
    loser = disputed.candidates[1]

    assert winner.text == "contract"
    # The losing reading had more services behind it, and the sum of their
    # individual scores was larger. Both of the naive rules would have
    # chosen it.
    assert len(loser.providers) > len(winner.providers)
    assert _sum_of_provider_scores(loser) > _sum_of_provider_scores(winner)


def _sum_of_provider_scores(candidate) -> float:
    return sum(
        value for name, value in candidate.components.items() if name.endswith(".score")
    )


def test_every_factor_that_made_the_score_is_kept_beside_it() -> None:
    table = table_of(
        spoken(Provider.ELEVENLABS, SENTENCE, log_probabilities=[-0.1] * len(SENTENCE)),
        spoken(Provider.OPENAI, SENTENCE, timed=False),
    )

    tokens = reconcile(table)
    components = tokens[0].candidates[0].components

    for factor in (
        "provider_reliability",
        "language_reliability",
        "acoustic_confidence",
        "vocabulary_evidence",
        "alignment_quality",
        "historical_performance",
    ):
        assert f"elevenlabs.{factor}" in components
    assert "combined" in components


def test_the_reliability_weights_are_a_parameter_rather_than_a_rule() -> None:
    backbone = spoken(Provider.ELEVENLABS, ["the", "contract", "stands"])
    openai = spoken(Provider.OPENAI, ["the", "contact", "stands"], timed=False)
    table = table_of(backbone, openai)
    trusting = ReliabilityWeights(by_provider={Provider.OPENAI: 1.0, Provider.ELEVENLABS: 0.05})

    tokens = reconcile(table, weights=trusting, options=ReconciliationOptions(
        minimum_decision_margin=0.0
    ))

    assert tokens[1].text == "contact"


def test_the_default_weights_match_the_specifications_ordering() -> None:
    weights = DEFAULT_PROVIDER_RELIABILITY
    assert weights[Provider.OPENAI] >= weights[Provider.ELEVENLABS]
    assert weights[Provider.ELEVENLABS] > weights[Provider.MICROSOFT]
    assert weights[Provider.MICROSOFT] > weights[Provider.ASSEMBLYAI]
    assert weights[Provider.ASSEMBLYAI] > weights[Provider.DEEPGRAM]


def test_a_history_of_getting_this_wrong_can_be_supplied() -> None:
    backbone = spoken(Provider.ELEVENLABS, ["the", "contract", "stands"])
    openai = spoken(Provider.OPENAI, ["the", "contact", "stands"], timed=False)
    table = table_of(backbone, openai)
    seen: list[WordFacts] = []

    def history(facts: WordFacts) -> float:
        seen.append(facts)
        return 0.1 if facts.provider is Provider.ELEVENLABS else 1.0

    tokens = reconcile(
        table,
        historical=history,
        options=ReconciliationOptions(minimum_decision_margin=0.0),
    )

    assert seen and any(facts.provider is Provider.OPENAI for facts in seen)
    assert tokens[1].text == "contact"


# -- Values that must never be guessed -------------------------------------


def test_a_disputed_amount_is_left_unresolved_with_both_readings() -> None:
    backbone = spoken(
        Provider.ELEVENLABS, ["the", "deposit", "was", "15,000", "rand", "in", "total"]
    )
    openai = spoken(
        Provider.OPENAI,
        ["the", "deposit", "was", "50,000", "rand", "in", "total"],
        timed=False,
    )
    microsoft = spoken(
        Provider.MICROSOFT,
        ["the", "deposit", "was", "50,000", "rand", "in", "total"],
        timed=False,
    )
    table = table_of(backbone, openai, microsoft)

    tokens = reconcile(table)
    amount = tokens[3]

    assert amount.text_confidence is Confidence.UNRESOLVED
    assert set(amount.is_uncertain_between()) == {"15,000", "50,000"}
    assert ReviewReason.HIGH_RISK_ENTITY in amount.review_reasons
    assert ReviewReason.NUMERIC_DISAGREEMENT in amount.review_reasons
    assert RiskCategory.MONEY in amount.risk_categories


def test_a_two_to_one_result_on_an_amount_is_still_not_enough() -> None:
    # Section 13 says the two-to-one result is meaningful evidence and still
    # not a reason to write the amount down as though it were known.
    backbone = spoken(Provider.ELEVENLABS, ["pay", "15,000", "rand"])
    openai = spoken(Provider.OPENAI, ["pay", "50,000", "rand"], timed=False)
    microsoft = spoken(Provider.MICROSOFT, ["pay", "50,000", "rand"], timed=False)
    table = table_of(backbone, openai, microsoft)

    tokens = reconcile(table)

    assert tokens[1].text_confidence is Confidence.UNRESOLVED


def test_an_agreed_amount_is_not_dragged_into_review() -> None:
    words = ["pay", "15,000", "rand"]
    table = table_of(
        spoken(Provider.ELEVENLABS, words),
        spoken(Provider.OPENAI, words, timed=False),
    )

    tokens = reconcile(table)

    assert tokens[1].text_confidence is Confidence.HIGH


def test_a_decimal_comma_read_two_ways_is_never_settled_quietly() -> None:
    backbone = spoken(Provider.ELEVENLABS, ["it", "was", "1,500", "exactly"])
    openai = spoken(Provider.OPENAI, ["it", "was", "1.500", "exactly"], timed=False)
    table = table_of(backbone, openai)

    tokens = reconcile(table)

    assert tokens[2].text_confidence is Confidence.UNRESOLVED


# -- Afrikaans, only when the user asked for it ----------------------------


AFRIKAANS_WORDS = ["ek", "dink", "ons", "moet", "nou", "praat"]


def test_no_afrikaans_reasoning_happens_when_afrikaans_is_disabled() -> None:
    backbone = spoken(Provider.ELEVENLABS, AFRIKAANS_WORDS)
    backbone.detected_language = Language.AFRIKAANS
    openai = spoken(Provider.OPENAI, AFRIKAANS_WORDS, timed=False)
    openai.detected_language = Language.AFRIKAANS
    table = table_of(backbone, openai)

    tokens = reconcile(
        table,
        configuration=RecordingConfiguration(afrikaans_enabled=False),
        results=[backbone, openai],
    )

    for token in tokens:
        assert token.language is not Language.AFRIKAANS
        assert Language.AFRIKAANS not in token.language_evidence.scores
        assert ReviewReason.LANGUAGE_UNCERTAIN not in token.review_reasons


def test_microsoft_carries_no_weight_on_a_confidently_afrikaans_span() -> None:
    backbone = spoken(Provider.ELEVENLABS, AFRIKAANS_WORDS)
    backbone.detected_language = Language.AFRIKAANS
    openai = spoken(Provider.OPENAI, AFRIKAANS_WORDS, timed=False)
    openai.detected_language = Language.AFRIKAANS
    wrong = ["ek", "dink", "ons", "moet", "nou", "praag"]
    microsoft = spoken(Provider.MICROSOFT, wrong, timed=False)
    microsoft.detected_language = Language.AFRIKAANS
    table = table_of(backbone, openai, microsoft)

    tokens = reconcile(
        table,
        configuration=RecordingConfiguration(afrikaans_enabled=True),
        results=[backbone, openai, microsoft],
    )

    disputed = tokens[5]
    assert disputed.language is Language.AFRIKAANS
    losing = [
        candidate for candidate in disputed.candidates if candidate.text == "praag"
    ]
    assert losing and losing[0].components["microsoft.language_reliability"] == 0.0
    assert losing[0].score == 0.0
    assert disputed.text == "praat"


def test_microsoft_is_only_reduced_where_the_language_is_uncertain() -> None:
    """A code-switch boundary, where nothing is confidently anything.

    One service read the recording as English and another as Afrikaans, and
    no word in the disputed stretch belongs to only one language. Microsoft
    must not be excluded, because the span is not known to be Afrikaans, and
    must not keep its full weight either, because it might be.
    """
    mixed = ["so", "toe", "ja", "nee", "mmm", "reg"]
    backbone = spoken(Provider.ELEVENLABS, mixed)
    backbone.detected_language = Language.ENGLISH
    microsoft = spoken(
        Provider.MICROSOFT, ["so", "toe", "ja", "nee", "mmm", "rek"], timed=False
    )
    microsoft.detected_language = Language.ENGLISH
    afrikaans_reader = spoken(Provider.OPENAI, mixed, timed=False)
    afrikaans_reader.detected_language = Language.AFRIKAANS
    table = table_of(backbone, microsoft, afrikaans_reader)

    tokens = reconcile(
        table,
        configuration=RecordingConfiguration(afrikaans_enabled=True),
        results=[backbone, microsoft, afrikaans_reader],
    )

    disputed = tokens[5]
    losing = [candidate for candidate in disputed.candidates if candidate.text == "rek"]
    assert losing, [candidate.text for candidate in disputed.candidates]
    weight = losing[0].components["microsoft.language_reliability"]
    assert 0.0 < weight < 1.0


def test_an_uncertain_language_is_a_review_reason_only_when_afrikaans_is_enabled() -> None:
    words = ["yes", "ja", "okay", "so"]
    backbone = spoken(Provider.ELEVENLABS, words)
    backbone.detected_language = Language.UNKNOWN
    table = table_of(backbone)

    enabled = reconcile(
        table,
        configuration=RecordingConfiguration(afrikaans_enabled=True),
        results=[backbone],
    )
    disabled = reconcile(table, results=[backbone])

    assert all(ReviewReason.LANGUAGE_UNCERTAIN in token.review_reasons for token in enabled)
    assert all(
        ReviewReason.LANGUAGE_UNCERTAIN not in token.review_reasons for token in disabled
    )


# -- Timing, said separately from text -------------------------------------


def test_a_split_shares_one_span_and_invents_no_boundary() -> None:
    decision = decide_timing(
        backbone_tokens=(
            ProviderToken(Provider.ELEVENLABS, 0, "cannot", start=1.0, end=1.8),
        ),
        equivalence=EquivalenceKind.COMPOUND,
        shared=True,
    )

    assert decision.status is TimingStatus.SHARED_PHRASE_SPAN
    assert decision.start == 1.0
    assert decision.end == 1.8


def test_a_merge_covers_the_combined_span() -> None:
    decision = decide_timing(
        backbone_tokens=(
            ProviderToken(Provider.ELEVENLABS, 0, "data", start=1.0, end=1.4),
            ProviderToken(Provider.ELEVENLABS, 1, "base", start=1.4, end=1.9),
        ),
        equivalence=EquivalenceKind.COMPOUND,
    )

    assert decision.start == 1.0
    assert decision.end == 1.9
    assert decision.status is TimingStatus.MAPPED_FORMAT_EQUIVALENT


def test_a_substitution_inherits_the_span_and_says_so() -> None:
    decision = decide_timing(
        backbone_tokens=(
            ProviderToken(Provider.ELEVENLABS, 0, "contact", start=2.0, end=2.6),
        ),
        equivalence=EquivalenceKind.DIFFERENT,
    )

    assert decision.status is TimingStatus.MAPPED_SUBSTITUTION


def test_forced_alignment_replaces_the_span_when_it_can() -> None:
    decision = decide_timing(
        backbone_tokens=(
            ProviderToken(Provider.ELEVENLABS, 0, "contact", start=2.0, end=2.6),
        ),
        equivalence=EquivalenceKind.DIFFERENT,
        forced_alignment=lambda text, span: AudioSpan(2.05, 2.7),
        text="contract",
    )

    assert decision.status is TimingStatus.FORCED_ALIGNED
    assert decision.start == 2.05


def test_forced_alignment_declining_is_reported_rather_than_hidden() -> None:
    decision = decide_timing(
        backbone_tokens=(
            ProviderToken(Provider.ELEVENLABS, 0, "contact", start=2.0, end=2.6),
        ),
        equivalence=EquivalenceKind.DIFFERENT,
        forced_alignment=lambda text, span: None,
        text="contract",
    )

    assert decision.status is TimingStatus.MAPPED_SUBSTITUTION
    assert ReviewReason.FORCED_ALIGNMENT_FAILED in decision.reasons


def test_a_joined_spelling_covers_the_span_of_the_words_it_replaced() -> None:
    backbone = spoken(Provider.ELEVENLABS, ["the", "data", "base", "is", "down"])
    openai = spoken(Provider.OPENAI, ["the", "database", "is", "down"], timed=False)
    table = table_of(backbone, openai)
    vocabulary = _index(VocabularyTerm(text="database"))

    tokens = reconcile(table, vocabulary=vocabulary)
    joined = token_at(tokens, "database")

    assert texts_of(tokens) == ["the", "database", "is", "down"]
    assert joined.alignment_status is AlignmentStatus.MERGE
    assert joined.timing_status is TimingStatus.MAPPED_FORMAT_EQUIVALENT
    assert joined.start == backbone.tokens[1].start
    assert joined.end == backbone.tokens[2].end


def test_a_split_spelling_shares_one_span_and_invents_no_join() -> None:
    backbone = spoken(Provider.ELEVENLABS, ["we", "cannot", "sign"])
    openai = spoken(Provider.OPENAI, ["we", "can", "not", "sign"], timed=False)
    table = table_of(backbone, openai)
    vocabulary = _index(VocabularyTerm(text="can not"))

    tokens = reconcile(table, vocabulary=vocabulary)

    assert texts_of(tokens) == ["we", "can", "not", "sign"]
    can, cannot = tokens[1], tokens[2]
    assert can.timing_status is TimingStatus.SHARED_PHRASE_SPAN
    assert cannot.timing_status is TimingStatus.SHARED_PHRASE_SPAN
    assert (can.start, can.end) == (cannot.start, cannot.end)
    assert can.start == backbone.tokens[1].start


def test_a_different_word_over_a_merged_span_is_a_substitution_not_an_exact_time() -> None:
    """The span a merged winner inherits is the backbone's, so the comparison is
    with what the backbone wrote there. "databus" over "data base" is a
    different word in the same place, and must say so; it used to be
    compared with its own reading and reported as an exact provider time.
    """
    backbone = spoken(Provider.ELEVENLABS, ["the", "data", "base", "is", "down"])
    openai = spoken(Provider.OPENAI, ["the", "databus", "is", "down"], timed=False)
    table = table_of(backbone, openai)
    vocabulary = _index(VocabularyTerm(text="databus"))

    tokens = reconcile(table, vocabulary=vocabulary)
    chosen = token_at(tokens, "databus")

    assert chosen.timing_status is TimingStatus.MAPPED_SUBSTITUTION
    assert chosen.alignment_status is not AlignmentStatus.EXACT


def test_a_joined_spelling_is_not_forced_on_a_recording_that_did_not_ask() -> None:
    # Without a reason to prefer it, the backbone's own word count stands,
    # and both of its words keep the times measured for them.
    backbone = spoken(Provider.ELEVENLABS, ["the", "data", "base", "is", "down"])
    openai = spoken(Provider.OPENAI, ["the", "database", "is", "down"], timed=False)

    tokens = reconcile(table_of(backbone, openai))

    assert texts_of(tokens) == ["the", "data", "base", "is", "down"]
    assert tokens[1].timing_status is TimingStatus.EXACT_PROVIDER_TIME


def test_the_losing_readings_are_kept_beside_the_winner() -> None:
    backbone = spoken(Provider.ELEVENLABS, ["the", "contract", "stands"])
    openai = spoken(Provider.OPENAI, ["the", "contact", "stands"], timed=False)

    tokens = reconcile(table_of(backbone, openai))
    disputed = tokens[1]

    assert len(disputed.candidates) == 2
    assert disputed.source_tokens
    assert all(candidate.source_tokens for candidate in disputed.candidates)


def test_a_word_no_service_timed_is_unaligned() -> None:
    decision = decide_timing(
        backbone_tokens=(ProviderToken(Provider.OPENAI, 0, "the"),),
        equivalence=EquivalenceKind.IDENTICAL,
    )

    assert decision.status is TimingStatus.UNALIGNED
    assert decision.start is None


def test_a_word_only_one_service_heard_has_no_acoustic_evidence() -> None:
    backbone = spoken(Provider.ELEVENLABS, ["we", "sign", "tomorrow"])
    openai = spoken(
        Provider.OPENAI, ["we", "will", "sign", "tomorrow"], timed=False
    )
    table = table_of(backbone, openai)

    tokens = reconcile(table)
    inserted = [token for token in tokens if token.text == "will"]

    assert inserted
    assert inserted[0].timing_status is TimingStatus.UNALIGNED
    assert ReviewReason.UNALIGNED_WORD in inserted[0].review_reasons
    assert inserted[0].source_audio_span is not None


def test_a_weakly_supported_insertion_is_kept_out_of_the_transcript() -> None:
    backbone = spoken(Provider.ELEVENLABS, ["we", "sign", "tomorrow"])
    deepgram = spoken(
        Provider.DEEPGRAM,
        ["we", "will", "sign", "tomorrow"],
        timed=False,
        confidences=[0.2, 0.2, 0.2, 0.2],
    )
    table = table_of(backbone, deepgram)

    tokens = reconcile(table)

    assert "will" not in texts_of(tokens)
    assert any(token.rejected_tokens for token in tokens)


def test_a_word_two_services_both_inserted_appears_once_with_both_as_support() -> None:
    """Two services hearing the word the backbone missed are one word.

    Each service's insertion used to be judged on its own, and both cleared
    the bar, so the transcript said "it was very very good today".
    """
    backbone = spoken(Provider.ELEVENLABS, ["it", "was", "good", "today"])
    heard = ["it", "was", "very", "good", "today"]
    openai = spoken(Provider.OPENAI, heard, timed=False)
    microsoft = spoken(Provider.MICROSOFT, heard, timed=False)

    tokens = reconcile(table_of(backbone, openai, microsoft))

    assert texts_of(tokens) == heard
    very = token_at(tokens, "very")
    assert very.alignment_status is AlignmentStatus.INSERTION
    assert set(very.candidates[0].providers) == {Provider.OPENAI, Provider.MICROSOFT}
    assert {reference.provider for reference in very.source_tokens} == {
        Provider.OPENAI,
        Provider.MICROSOFT,
    }
    components = very.candidates[0].components
    assert components["supporting_providers"] == 2.0
    # Agreement adds evidence, so the shared word scores above either
    # service on its own.
    assert components["combined"] > components["openai.score"]
    assert components["combined"] > components["microsoft.score"]


# -- Words the backbone may have imagined ----------------------------------


def _long_recording(count: int = 60) -> list[str]:
    return [f"word{index}" for index in range(count)]


def test_a_word_two_services_missed_while_the_backbone_was_unsure_is_removed() -> None:
    words = _long_recording()
    backbone = spoken(Provider.ELEVENLABS, words, confidences=[0.3] * len(words))
    missing_one = [*words[:15], *words[16:]]
    openai = spoken(Provider.OPENAI, missing_one, timed=False)
    microsoft = spoken(Provider.MICROSOFT, missing_one, timed=False)

    tokens = reconcile(table_of(backbone, openai, microsoft))

    assert "word15" not in texts_of(tokens)
    assert len(tokens) == len(words) - 1


def test_a_service_that_stopped_half_way_does_not_delete_the_rest_of_the_recording() -> None:
    """Silence after the point where a service stopped is not a deletion.

    A chunk that failed, or a request that timed out after partial output,
    leaves a service with words for the first part of the recording and
    nothing for the rest. It is not in the table's list of missing services,
    because it did say something. Counting every later word as one it
    failed to hear let two such services, with an unsure backbone, remove
    the whole second half of a recording.
    """
    words = _long_recording()
    backbone = spoken(Provider.ELEVENLABS, words, confidences=[0.3] * len(words))
    openai = spoken(Provider.OPENAI, words[:20], timed=False)
    microsoft = spoken(Provider.MICROSOFT, words[:20], timed=False)

    tokens = reconcile(table_of(backbone, openai, microsoft))

    assert texts_of(tokens) == words


def test_a_service_that_started_late_does_not_delete_the_start_of_the_recording() -> None:
    words = _long_recording()
    backbone = spoken(Provider.ELEVENLABS, words, confidences=[0.3] * len(words))
    openai = spoken(Provider.OPENAI, words[40:], timed=False)
    microsoft = spoken(Provider.MICROSOFT, words[40:], timed=False)

    tokens = reconcile(table_of(backbone, openai, microsoft))

    assert texts_of(tokens) == words


def test_a_service_absent_from_the_whole_neighbourhood_is_not_a_deletion() -> None:
    """Inside a long gap a service is as absent as one that never answered.

    Near the edges of the gap its silence still counts, because a service
    that dropped a word or two beside words it did hear is exactly the
    case the hallucination rule is for.
    """
    words = _long_recording(120)
    backbone = spoken(Provider.ELEVENLABS, words, confidences=[0.3] * len(words))
    with_gap = [*words[:10], *words[110:]]
    openai = spoken(Provider.OPENAI, with_gap, timed=False)
    microsoft = spoken(Provider.MICROSOFT, with_gap, timed=False)

    tokens = reconcile(table_of(backbone, openai, microsoft))
    kept = texts_of(tokens)

    reach = DEFAULT_OPTIONS.absence_reach
    assert "word60" in kept
    assert all(words[index] in kept for index in range(10 + reach, 110 - reach))
    assert "word10" not in kept
    assert "word109" not in kept


# -- The risk window ----------------------------------------------------------


def test_risk_is_judged_from_the_words_within_reach_and_no_further() -> None:
    """The windowed copy must answer exactly as the whole recording did."""
    from vox_verbatim.transcription.reconcile import _risk_for
    from vox_verbatim.transcription.risk import RISK_WINDOW, risk_at

    words = ["pay", "fifteen", "thousand", "rand", *(["word"] * 40), "on", "the", "fifteenth"]

    class _Candidate:
        def __init__(self, display: str) -> None:
            self.display = display

    def by_copying_everything(scope, groups):
        found = []
        for group in groups:
            copy = list(words)
            copy[scope[0] : scope[0] + len(scope)] = [group.display]
            for category in risk_at(copy, scope[0]):
                if category not in found:
                    found.append(category)
        return tuple(found)

    for position in range(len(words)):
        for scope in ((position,), (position, position + 1)):
            if scope[-1] >= len(words):
                continue
            groups = [_Candidate("fifty"), _Candidate("fifteen"), _Candidate("two words")]
            assert _risk_for(scope, words, groups) == by_copying_everything(scope, groups)
    assert RISK_WINDOW < 40


# -- Speakers, said separately again ---------------------------------------


def test_the_speaker_comes_from_the_backbone_and_the_text_does_not_have_to() -> None:
    backbone = spoken(
        Provider.ELEVENLABS,
        ["yes", "the", "contract"],
        speakers=["A", "A", "A"],
    )
    openai = spoken(Provider.OPENAI, ["yes", "the", "contact"], timed=False)
    table = table_of(backbone, openai)

    tokens = reconcile(table)

    assert all(token.speaker == "A" for token in tokens)
    assert all(token.speaker_source is Provider.ELEVENLABS for token in tokens)


def test_an_undiarised_recording_is_not_flagged_for_a_question_nobody_asked() -> None:
    table = table_of(
        spoken(Provider.ELEVENLABS, SENTENCE),
        spoken(Provider.OPENAI, SENTENCE, timed=False),
    )

    tokens = reconcile(table)

    assert all(token.speaker is None for token in tokens)
    assert all(
        ReviewReason.SPEAKER_UNCERTAIN not in token.review_reasons for token in tokens
    )


def test_a_service_hearing_a_change_of_voice_where_the_backbone_does_not_is_a_dispute() -> None:
    backbone = spoken(
        Provider.ELEVENLABS,
        ["yes", "we", "should", "sign", "it", "today"],
        speakers=["A"] * 6,
    )
    other = spoken(
        Provider.ASSEMBLYAI,
        ["yes", "we", "should", "sign", "it", "today"],
        timed=False,
        speakers=["one", "one", "two", "two", "two", "two"],
    )
    table = table_of(backbone, other)

    tokens = reconcile(table)

    assert any(
        ReviewReason.SPEAKER_UNCERTAIN in token.review_reasons for token in tokens
    )


def test_people_talking_over_each_other_is_reported() -> None:
    tokens = [
        ProviderToken(Provider.ELEVENLABS, 0, "yes", start=0.0, end=1.0, speaker="A"),
        ProviderToken(Provider.ELEVENLABS, 1, "no", start=0.8, end=1.6, speaker="B"),
        ProviderToken(Provider.ELEVENLABS, 2, "fine", start=1.7, end=2.2, speaker="B"),
    ]
    backbone = ProviderResult(provider=Provider.ELEVENLABS, tokens=tokens)
    table = table_of(backbone)

    settled = reconcile(table)

    assert any(
        ReviewReason.OVERLAPPING_SPEECH in token.review_reasons for token in settled
    )


# -- What the second opinion and the model add ------------------------------


def test_a_position_that_escalation_did_not_settle_says_so() -> None:
    backbone = spoken(Provider.ELEVENLABS, ["the", "15,000", "rand"])
    openai = spoken(Provider.OPENAI, ["the", "50,000", "rand"], timed=False)
    table = table_of(backbone, openai)

    tokens = reconcile(table, escalated={1})

    assert ReviewReason.ESCALATION_UNRESOLVED in tokens[1].review_reasons


def test_the_model_may_choose_between_the_readings_on_offer() -> None:
    backbone = spoken(Provider.ELEVENLABS, ["the", "contract", "stands"])
    openai = spoken(Provider.OPENAI, ["the", "contact", "stands"], timed=False)
    table = table_of(backbone, openai)

    tokens = reconcile(table, adjudications={1: "contact"})

    assert tokens[1].text == "contact"
    assert tokens[1].llm_decision == "contact"


def test_the_model_may_not_introduce_a_reading_of_its_own() -> None:
    backbone = spoken(Provider.ELEVENLABS, ["the", "contract", "stands"])
    openai = spoken(Provider.OPENAI, ["the", "contact", "stands"], timed=False)
    table = table_of(backbone, openai)

    tokens = reconcile(table, adjudications={1: "agreement"})

    assert tokens[1].text in {"contract", "contact"}
    assert ReviewReason.ADJUDICATION_DECLINED in tokens[1].review_reasons


def test_a_model_that_declined_is_reported_as_having_declined() -> None:
    backbone = spoken(Provider.ELEVENLABS, ["the", "contract", "stands"])
    openai = spoken(Provider.OPENAI, ["the", "contact", "stands"], timed=False)
    table = table_of(backbone, openai)

    tokens = reconcile(table, adjudications={1: None})

    assert ReviewReason.ADJUDICATION_DECLINED in tokens[1].review_reasons
    assert tokens[1].text_confidence is Confidence.UNRESOLVED


# -- Every reason is raised by something ------------------------------------


def test_every_review_reason_can_actually_be_raised() -> None:
    """No reason in the model is decorative.

    Each of the thirteen reasons is a filter in the review window, and a
    filter that nothing can ever raise is worse than no filter: it tells a
    person there is nothing of that kind in their transcript when nothing
    was ever looking.
    """
    raised: set[ReviewReason] = set()

    for tokens in _scenarios():
        for token in tokens:
            raised.update(token.review_reasons)

    assert raised == set(ReviewReason)


def _scenarios() -> list[list]:
    """A recording for each kind of trouble, gathered in one place."""
    scenarios: list[list] = []

    # Disagreement over an amount, a name and a number, with the backbone
    # unsure of what it heard and a second opinion that did not settle it.
    backbone = spoken(
        Provider.ELEVENLABS,
        ["pay", "Niekerk", "15,000", "rand"],
        log_probabilities=[-1.6, -1.6, -1.6, -1.6],
    )
    openai = spoken(Provider.OPENAI, ["pay", "Nieker", "50,000", "rand"], timed=False)
    scenarios.append(
        reconcile(table_of(backbone, openai), escalated={1, 2}, adjudications={1: None})
    )

    # An inserted word, and a substitution that wins against a backbone which
    # was unsure of itself, so the span has to be measured again and the
    # forced aligner refuses.
    short = spoken(
        Provider.ELEVENLABS,
        ["we", "sign", "tomorrow"],
        log_probabilities=[-1.8, -1.8, -1.8],
    )
    long = spoken(Provider.OPENAI, ["we", "will", "sign", "today"], timed=False)
    scenarios.append(
        reconcile(
            table_of(short, long),
            forced_alignment=lambda text, span: None,
            options=ReconciliationOptions(minimum_decision_margin=0.0),
        )
    )

    # Overlapping speech and a disputed speaker boundary.
    overlapping = ProviderResult(
        provider=Provider.ELEVENLABS,
        tokens=[
            ProviderToken(Provider.ELEVENLABS, 0, "yes", start=0.0, end=1.0, speaker="A"),
            ProviderToken(Provider.ELEVENLABS, 1, "no", start=0.8, end=1.6, speaker="B"),
        ],
    )
    heard_otherwise = ProviderResult(
        provider=Provider.ASSEMBLYAI,
        tokens=[
            ProviderToken(Provider.ASSEMBLYAI, 0, "yes", speaker="one"),
            ProviderToken(Provider.ASSEMBLYAI, 1, "no", speaker="one"),
        ],
    )
    scenarios.append(reconcile(table_of(overlapping, heard_otherwise)))

    # A stretch whose language nobody can pin down, with Afrikaans enabled.
    unclear = spoken(Provider.ELEVENLABS, ["ja", "okay", "so", "nou"])
    unclear.detected_language = Language.UNKNOWN
    scenarios.append(
        reconcile(
            table_of(unclear),
            configuration=RecordingConfiguration(afrikaans_enabled=True),
            results=[unclear],
        )
    )

    # A stretch where the alignment itself lost its place, so the columns
    # carry a low quality of their own.
    muddled = spoken(
        Provider.ELEVENLABS, ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    )
    different = spoken(
        Provider.OPENAI,
        ["alpha", "quebec", "romeo", "sierra", "tango", "foxtrot"],
        timed=False,
    )
    scenarios.append(reconcile(table_of(muddled, different)))

    return scenarios


def test_the_options_are_all_reachable_from_one_place() -> None:
    assert DEFAULT_OPTIONS.minimum_decision_margin > 0.0
    assert 0.0 < DEFAULT_OPTIONS.agreement_independence < 1.0
    assert DEFAULT_OPTIONS.vocabulary_misrecognition < 1.0 < DEFAULT_OPTIONS.vocabulary_exact


# -- Audio events ----------------------------------------------------------


def _laughing(texts: list[str]) -> ProviderResult:
    """ElevenLabs' answer with "(laughter)" marked as an audio event."""
    result = spoken(Provider.ELEVENLABS, texts)
    tokens = [
        dataclasses.replace(token, is_audio_event=True) if token.text == "(laughter)" else token
        for token in result.tokens
    ]
    return dataclasses.replace(result, tokens=tokens)


def test_an_audio_event_stays_in_the_transcript_unflagged() -> None:
    backbone = _laughing(["so", "then", "(laughter)", "we", "left"])
    table = table_of(
        backbone,
        spoken(Provider.OPENAI, ["so", "then", "we", "left"], timed=False),
        spoken(Provider.DEEPGRAM, ["so", "then", "we", "left"], timed=False),
    )

    tokens = reconcile(table)

    assert texts_of(tokens) == ["so", "then", "(laughter)", "we", "left"]
    event = token_at(tokens, "(laughter)")
    source = backbone.tokens[2]
    assert event.review_status is ReviewStatus.SETTLED
    assert not event.needs_review
    assert ReviewReason.PROVIDER_DISAGREEMENT not in event.review_reasons
    assert (event.start, event.end) == (source.start, source.end)
    assert event.timing_status is TimingStatus.EXACT_PROVIDER_TIME
    assert all(not token.needs_review for token in tokens)


def test_an_audio_event_before_the_first_word_comes_first() -> None:
    table = table_of(
        _laughing(["(laughter)", "so", "then"]),
        spoken(Provider.OPENAI, ["so", "then"], timed=False),
        spoken(Provider.DEEPGRAM, ["so", "then"], timed=False),
    )

    tokens = reconcile(table)

    assert texts_of(tokens) == ["(laughter)", "so", "then"]
    assert not tokens[0].needs_review
