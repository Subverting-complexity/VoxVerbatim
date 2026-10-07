"""Tests for the vocabulary levels, the learned corrections and how they are kept."""

from __future__ import annotations

import json

from vox_verbatim.transcription.model import Language, Provider
from vox_verbatim.transcription.normalise import EquivalenceKind, normalise
from vox_verbatim.transcription.project import LearnedName
from vox_verbatim.transcription.vocabulary import (
    LearnedCorrection,
    LearnedCorrections,
    TermCategory,
    Vocabulary,
    VocabularyIndex,
    VocabularyLevel,
    VocabularyProfile,
    VocabularyStore,
    VocabularyTerm,
    folder_terms_first,
    resolve_terms,
    terms_from_learned_names,
)


def term(text: str, confirmations: int = 0, **extra) -> VocabularyTerm:
    return VocabularyTerm(text=text, confirmation_count=confirmations, **extra)


def four_level_vocabulary() -> Vocabulary:
    """One profile at each level, with a term shared between two of them."""
    return Vocabulary(
        profiles=[
            VocabularyProfile(
                id="global",
                level=VocabularyLevel.GLOBAL,
                name="Everything",
                terms=[term("Kubernetes"), term("Bosch", confirmations=1)],
            ),
            VocabularyProfile(
                id="acme",
                level=VocabularyLevel.CLIENT,
                name="Acme",
                terms=[term("Acme Holdings"), term("Bosch", confirmations=2)],
            ),
            VocabularyProfile(
                id="atlas",
                level=VocabularyLevel.PROJECT,
                name="Project Atlas",
                terms=[term("Atlas")],
            ),
            VocabularyProfile(
                id="suzanne",
                level=VocabularyLevel.SPEAKER,
                name="Suzanne",
                terms=[term("Vermeulen")],
            ),
        ]
    )


def test_all_four_levels_combine_into_one_list():
    vocabulary = four_level_vocabulary()

    texts = [
        found.text for found in resolve_terms(vocabulary, ["global", "acme", "atlas", "suzanne"])
    ]

    assert set(texts) == {"Kubernetes", "Bosch", "Acme Holdings", "Atlas", "Vermeulen"}


def test_only_the_chosen_profiles_are_used():
    vocabulary = four_level_vocabulary()

    texts = [found.text for found in resolve_terms(vocabulary, ["global"])]

    assert texts == ["Bosch", "Kubernetes"]


def test_the_narrowest_levels_come_first():
    """A service that will only take a few terms should get the specific ones."""
    vocabulary = four_level_vocabulary()

    texts = [
        found.text for found in resolve_terms(vocabulary, ["global", "acme", "atlas", "suzanne"])
    ]

    assert texts.index("Vermeulen") < texts.index("Atlas")
    assert texts.index("Atlas") < texts.index("Acme Holdings")
    assert texts.index("Acme Holdings") < texts.index("Kubernetes")


def test_within_one_level_the_confirmed_terms_come_first():
    vocabulary = Vocabulary(
        profiles=[
            VocabularyProfile(
                id="global",
                level=VocabularyLevel.GLOBAL,
                terms=[
                    term("Never seen", confirmations=0),
                    term("Seen twice", confirmations=2),
                    term("Seen often", confirmations=9),
                ],
            )
        ]
    )

    texts = [found.text for found in resolve_terms(vocabulary, ["global"])]

    assert texts == ["Seen often", "Seen twice", "Never seen"]


def test_a_repeated_term_appears_once_and_keeps_its_narrowest_place():
    vocabulary = four_level_vocabulary()

    resolved = resolve_terms(vocabulary, ["global", "acme"])
    texts = [found.text for found in resolved]

    assert texts.count("Bosch") == 1
    assert texts.index("Bosch") < texts.index("Kubernetes")


def test_a_repeated_term_adds_up_its_confirmations():
    """A term the user wrote down twice is a stronger signal, not the same one."""
    vocabulary = four_level_vocabulary()

    resolved = resolve_terms(vocabulary, ["global", "acme"])
    bosch = next(found for found in resolved if found.text == "Bosch")

    assert bosch.confirmation_count == 3


def test_a_repeated_term_gathers_what_the_other_copy_knew():
    vocabulary = Vocabulary(
        profiles=[
            VocabularyProfile(
                id="global",
                level=VocabularyLevel.GLOBAL,
                terms=[
                    VocabularyTerm(
                        text="Bosch",
                        category=TermCategory.PERSON,
                        common_misrecognitions=("Bosh",),
                    )
                ],
            ),
            VocabularyProfile(
                id="acme",
                level=VocabularyLevel.CLIENT,
                terms=[VocabularyTerm(text="Bosch", common_misrecognitions=("Bosque",))],
            ),
        ]
    )

    bosch = resolve_terms(vocabulary, ["global", "acme"])[0]

    assert bosch.common_misrecognitions == ("Bosque", "Bosh")
    assert bosch.category is TermCategory.PERSON


def test_two_spellings_of_one_name_become_one_term():
    """ElevenLabs is sent fewer than a hundred terms; one name must not cost two."""
    vocabulary = Vocabulary(
        profiles=[
            VocabularyProfile(
                id="global", level=VocabularyLevel.GLOBAL, terms=[term("Muller")]
            ),
            VocabularyProfile(
                id="acme", level=VocabularyLevel.CLIENT, terms=[term("Müller")]
            ),
        ]
    )

    resolved = resolve_terms(vocabulary, ["global", "acme"])

    assert len(resolved) == 1
    assert resolved[0].text == "Müller"


def test_a_number_written_two_ways_is_one_term():
    vocabulary = Vocabulary(
        profiles=[
            VocabularyProfile(
                id="global",
                level=VocabularyLevel.GLOBAL,
                terms=[term("Platform twenty five"), term("Platform 25")],
            )
        ]
    )

    assert len(resolve_terms(vocabulary, ["global"])) == 1


def test_a_profile_that_was_never_saved_is_simply_skipped():
    vocabulary = four_level_vocabulary()

    texts = [found.text for found in resolve_terms(vocabulary, ["global", "not-a-profile"])]

    assert texts == ["Bosch", "Kubernetes"]


# -- Learned corrections -------------------------------------------------


def test_the_same_correction_twice_is_counted_rather_than_listed_twice():
    corrections = LearnedCorrections()

    corrections.record("Bosh", "Bosch", Provider.OPENAI, when="2026-01-01T09:00:00")
    corrections.record("Bosh", "Bosch", Provider.OPENAI, when="2026-02-01T09:00:00")

    assert len(corrections.corrections) == 1
    assert corrections.corrections[0].occurrences == 2
    assert corrections.corrections[0].last_seen == "2026-02-01T09:00:00"


def test_the_same_word_misheard_by_two_services_stays_two_pieces_of_evidence():
    corrections = LearnedCorrections()

    corrections.record("Bosh", "Bosch", Provider.OPENAI)
    corrections.record("Bosh", "Bosch", Provider.DEEPGRAM)

    assert len(corrections.corrections) == 2
    assert corrections.mistakes_by_provider() == {Provider.OPENAI: 1, Provider.DEEPGRAM: 1}


def test_a_correction_that_changed_nothing_is_not_recorded():
    corrections = LearnedCorrections()

    assert corrections.record("Bosch", "  Bosch  ", Provider.OPENAI) is None
    assert corrections.record("", "Bosch") is None
    assert corrections.corrections == []


def test_a_change_of_capitals_or_spelling_alone_is_still_a_correction():
    """The two spellings sound the same, and which one to print is the point."""
    corrections = LearnedCorrections()

    assert corrections.record("bosch", "Bosch", Provider.OPENAI) is not None
    assert corrections.record("Mueller", "Müller", Provider.OPENAI) is not None
    assert len(corrections.corrections) == 2


def test_a_correction_made_the_other_way_round_is_a_different_correction():
    """Counting the reversal as another sighting would say the opposite of what happened."""
    corrections = LearnedCorrections()

    corrections.record("Mueller", "Müller", Provider.OPENAI)
    corrections.record("Müller", "Mueller", Provider.OPENAI)

    assert len(corrections.corrections) == 2
    assert all(correction.occurrences == 1 for correction in corrections.corrections)


def test_the_same_mistake_in_different_capitals_is_the_same_mistake():
    corrections = LearnedCorrections()

    corrections.record("Bosh", "Bosch", Provider.OPENAI)
    corrections.record("bosh", "Bosch", Provider.OPENAI)

    assert len(corrections.corrections) == 1
    assert corrections.corrections[0].occurrences == 2


def test_recurring_corrections_become_terms_and_one_off_ones_do_not():
    corrections = LearnedCorrections()
    corrections.record("Bosh", "Bosch", Provider.OPENAI)
    corrections.record("Bosh", "Bosch", Provider.OPENAI)
    corrections.record("Fermilab", "Vermeulen", Provider.MICROSOFT)

    terms = corrections.as_terms()

    assert [found.text for found in terms] == ["Bosch"]
    assert terms[0].common_misrecognitions == ("Bosh",)
    assert terms[0].confirmation_count == 2


def test_corrections_from_two_services_merge_into_one_term():
    corrections = LearnedCorrections()
    corrections.record("Bosh", "Bosch", Provider.OPENAI)
    corrections.record("Bosh", "Bosch", Provider.OPENAI)
    corrections.record("Bosque", "Bosch", Provider.DEEPGRAM)
    corrections.record("Bosque", "Bosch", Provider.DEEPGRAM)

    terms = corrections.as_terms()

    assert len(terms) == 1
    assert terms[0].common_misrecognitions == ("Bosh", "Bosque")
    assert terms[0].confirmation_count == 4


def test_corrections_are_offered_ahead_of_every_typed_in_list():
    vocabulary = four_level_vocabulary()
    vocabulary.corrections.record("Fermilab", "Vermeulen", Provider.MICROSOFT)
    vocabulary.corrections.record("Fermilab", "Vermeulen", Provider.MICROSOFT)

    texts = [
        found.text for found in resolve_terms(vocabulary, ["global", "acme", "atlas", "suzanne"])
    ]

    assert texts[0] == "Vermeulen"
    assert texts.count("Vermeulen") == 1


def test_corrections_can_be_left_out_of_a_resolved_list():
    vocabulary = four_level_vocabulary()
    vocabulary.corrections.record("Aker", "Acme Holdings", Provider.OPENAI)
    vocabulary.corrections.record("Aker", "Acme Holdings", Provider.OPENAI)

    resolved = resolve_terms(vocabulary, ["global"], include_corrections=False)

    assert [found.text for found in resolved] == ["Bosch", "Kubernetes"]


def test_corrections_count_up_the_mistakes_each_service_made():
    corrections = LearnedCorrections()
    corrections.record("Bosh", "Bosch", Provider.OPENAI)
    corrections.record("Bosh", "Bosch", Provider.OPENAI)
    corrections.record("fifteen", "fifty", Provider.OPENAI)
    corrections.record("Fermilab", "Vermeulen", Provider.MICROSOFT)
    corrections.record("Ada", "Adrienne")

    counts = corrections.mistakes_by_provider()

    assert counts == {Provider.OPENAI: 3, Provider.MICROSOFT: 1}


def test_corrections_can_be_looked_up_by_what_was_wrong():
    corrections = LearnedCorrections()
    corrections.record("Bosh", "Bosch", Provider.OPENAI)
    corrections.record("Bosh", "Bosch", Provider.DEEPGRAM)

    found = corrections.corrections_for("BOSH")

    assert len(found) == 2
    assert {correction.provider for correction in found} == {Provider.OPENAI, Provider.DEEPGRAM}


# -- The file on disk ----------------------------------------------------


def test_a_saved_vocabulary_comes_back_unchanged(tmp_path):
    store = VocabularyStore(tmp_path / "vocabulary.json")
    vocabulary = four_level_vocabulary()
    vocabulary.corrections.record("Bosh", "Bosch", Provider.OPENAI, when="2026-01-01T09:00:00")

    assert store.save(vocabulary)

    assert store.load() == vocabulary


def test_a_term_keeps_all_of_its_details_across_a_save(tmp_path):
    store = VocabularyStore(tmp_path / "vocabulary.json")
    vocabulary = Vocabulary(
        profiles=[
            VocabularyProfile(
                id="global",
                level=VocabularyLevel.GLOBAL,
                terms=[
                    VocabularyTerm(
                        text="Vermeulen",
                        category=TermCategory.PERSON,
                        language=Language.AFRIKAANS,
                        pronunciation_hints=("fer-MER-len",),
                        common_misrecognitions=("Fermilab",),
                        confirmation_count=4,
                    )
                ],
            )
        ]
    )

    assert store.save(vocabulary)

    assert store.load().profiles[0].terms[0] == vocabulary.profiles[0].terms[0]


def test_no_vocabulary_file_gives_an_empty_vocabulary(tmp_path):
    loaded = VocabularyStore(tmp_path / "not-there.json").load()

    assert loaded == Vocabulary()
    assert loaded.profiles == []


def test_a_damaged_vocabulary_file_gives_an_empty_vocabulary(tmp_path):
    path = tmp_path / "vocabulary.json"
    path.write_text("{ not json at all", encoding="utf-8")

    assert VocabularyStore(path).load() == Vocabulary()


def test_a_vocabulary_file_that_is_not_text_gives_an_empty_vocabulary(tmp_path):
    path = tmp_path / "vocabulary.json"
    path.write_bytes(b'{"profiles": \xff\xfe}')

    assert VocabularyStore(path).load() == Vocabulary()


def test_one_unusable_profile_does_not_lose_the_others(tmp_path):
    """Losing one client's names is a small loss; refusing to load is a large one."""
    path = tmp_path / "vocabulary.json"
    path.write_text(
        json.dumps(
            {
                "profiles": [
                    {"level": "client", "terms": [{"text": "Nameless"}]},
                    "not a profile at all",
                    {"id": "global", "level": "global", "terms": [{"text": "Kubernetes"}]},
                ]
            }
        ),
        encoding="utf-8",
    )

    loaded = VocabularyStore(path).load()

    assert [profile.id for profile in loaded.profiles] == ["global"]
    assert [found.text for found in loaded.profiles[0].terms] == ["Kubernetes"]


def test_nonsense_inside_a_term_falls_back_only_for_that_part(tmp_path):
    path = tmp_path / "vocabulary.json"
    path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "id": "global",
                        "level": "made up level",
                        "terms": [
                            {"text": "  Bosch  ", "category": "wizard", "language": "klingon"},
                            {"text": "   "},
                            {"confirmation_count": 3},
                            {
                                "text": "Vermeulen",
                                "category": "person",
                                "language": "af",
                                "confirmation_count": -2,
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    profile = VocabularyStore(path).load().profiles[0]

    assert profile.level is VocabularyLevel.GLOBAL
    assert [found.text for found in profile.terms] == ["Bosch", "Vermeulen"]
    assert profile.terms[0].category is None
    assert profile.terms[0].language is None
    assert profile.terms[1].category is TermCategory.PERSON
    assert profile.terms[1].language is Language.AFRIKAANS
    assert profile.terms[1].confirmation_count == 0


def test_a_correction_naming_an_unknown_service_still_counts(tmp_path):
    path = tmp_path / "vocabulary.json"
    path.write_text(
        json.dumps(
            {
                "learned": {
                    "corrections": [
                        {"wrong_text": "Bosh", "right_text": "Bosch", "provider": "yodel"},
                        {"wrong_text": "", "right_text": "Bosch"},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    corrections = VocabularyStore(path).load().corrections.corrections

    assert corrections == [
        LearnedCorrection(wrong_text="Bosh", right_text="Bosch", occurrences=1, last_seen="")
    ]


# -- Looking a word up ---------------------------------------------------


def test_a_known_term_is_recognised_whatever_its_case():
    index = VocabularyIndex([term("Van Der Merwe"), term("Kubernetes")])

    assert index.knows("van der merwe")
    assert index.knows("KUBERNETES")
    assert "Kubernetes" in index


def test_surrounding_space_does_not_hide_a_known_term():
    index = VocabularyIndex([term("Bosch")])

    assert index.match("  bosch  ") is not None
    assert index.match("  ") is None
    assert index.match("") is None


def test_an_unknown_word_is_not_claimed_as_a_term():
    index = VocabularyIndex([term("Bosch")])

    assert not index.knows("Boston")
    assert index.match("Boston") is None


def test_the_matched_term_carries_everything_known_about_it():
    index = VocabularyIndex([VocabularyTerm(text="Bosch", category=TermCategory.PERSON)])

    found = index.match("bosch")

    assert found is not None
    assert found.term.category is TermCategory.PERSON


def test_a_name_spelled_without_its_umlaut_still_finds_the_term():
    """This is what the real comparison module buys us, and it is worth having."""
    index = VocabularyIndex([term("Müller")])

    assert index.knows("Muller")
    assert index.knows("Mueller")
    assert index.match("Muller").term.text == "Müller"


def test_the_match_says_how_closely_it_matched():
    """Reconciliation weighs an exact hit differently from a formatting coincidence."""
    index = VocabularyIndex([term("Müller"), term("25 Main Street")])

    assert index.match("Müller").kind is EquivalenceKind.IDENTICAL
    assert index.match("müller").kind is EquivalenceKind.CASE_ONLY
    assert index.match("Muller").kind is EquivalenceKind.SPELLING_VARIANT
    assert index.match("twenty five Main Street").kind is EquivalenceKind.NUMBER_FORMAT

    assert index.match("Müller").is_exact
    assert index.match("müller").is_exact
    assert not index.match("Muller").is_exact


def test_a_known_mistake_is_not_treated_as_a_known_term():
    """The wrong spelling must not be handed the support the right one earns."""
    index = VocabularyIndex([VocabularyTerm(text="Bosch", common_misrecognitions=("Bosh",))])

    assert not index.knows("Bosh")
    assert index.misrecognition_of("bosh") is not None
    assert index.misrecognition_of("bosh").text == "Bosch"
    assert index.misrecognition_of("Boston") is None


def test_a_mistake_that_sounds_like_its_own_correction_is_still_a_mistake():
    """The case that would collapse silently if mistakes were matched on sound.

    "Mueller" and "Müller" are the same spoken evidence, so a mistake matched
    the way a term is matched would be found as the term it is a mistake for,
    and the user's knowledge that one of the two is wrong would be lost.
    """
    assert normalise("Mueller") == normalise("Müller")

    index = VocabularyIndex([VocabularyTerm(text="Müller", common_misrecognitions=("Mueller",))])

    assert index.knows("Müller")
    assert not index.knows("Mueller")
    assert index.misrecognition_of("Mueller").text == "Müller"


def test_a_name_that_is_somebody_elses_misspelling_is_still_a_name():
    """An exact term wins over a mistake recorded against a different term."""
    index = VocabularyIndex(
        [
            VocabularyTerm(text="Müller", common_misrecognitions=("Mueller",)),
            term("Mueller"),
        ]
    )

    assert index.knows("Mueller")
    assert index.match("Mueller").term.text == "Mueller"



def _match_by_walking(terms: list[VocabularyTerm], text: str) -> VocabularyTerm | None:
    """The lookup as it was first written, for terms with no recorded mistakes.

    A term as the user wrote it won outright, then the comparison form was
    looked up, and a miss walked the terms asking ``are_equivalent`` of
    each. The index answers the whole thing
    by key now, because reconciliation asks it for every word of a
    transcript, and the walk made a large vocabulary cost more than it was
    worth. The keys have to give this answer, first term included, and
    this is what holds them to it.
    """
    from vox_verbatim.transcription.normalise import are_equivalent

    by_plain: dict[str, VocabularyTerm] = {}
    by_form: dict[str, VocabularyTerm] = {}
    for candidate in terms:
        by_plain.setdefault(candidate.text.strip().casefold(), candidate)
        by_form.setdefault(normalise(candidate.text), candidate)
    found = by_plain.get(text.strip().casefold())
    if found is not None:
        return found
    found = by_form.get(normalise(text))
    if found is not None:
        return found
    for candidate in terms:
        if are_equivalent(candidate.text, text):
            return candidate
    return None


def test_the_keyed_lookup_finds_exactly_the_term_the_walk_would():
    """Every way two texts can be equivalent, and the order of the terms.

    The umlaut is the delicate case in both directions: "Muller" reaches
    "Müller" because the term has the German letter, "Müller" reaches
    "Mueller" because the text has it, and "Muller" never reaches "Mueller"
    because neither does. Two terms that share a key have to come back as
    the earlier of the two, which is what "first term" meant in the walk.
    """
    terms = [
        term("Müller"),
        term("Mueller"),
        term("Muller"),
        term("25 Main Street"),
        term("data base"),
        term("Jürgen Bosch"),
        term("Bosch"),
        term("fünf"),
        term("five"),
        term("Straße"),
        term("I'm"),
        term("up-to-date"),
    ]
    texts = [candidate.text for candidate in terms] + [
        "muller", "MÜLLER", "mueller", "Mühller", "twenty five Main Street",
        "twenty-five main street", "database", "Data-Base", "Jurgen Bosch", "Juergen Bosch",
        "bosch,", "funf", "fuenf", "5", "Strasse", "strasse", "I am", "im", "up to date",
        "upto date", "nothing here", "", "   ", "Bosh",
    ]
    index = VocabularyIndex(terms)

    for text in texts:
        cleaned = text.strip()
        expected = _match_by_walking(terms, cleaned) if cleaned else None

        found = index.match(text)

        assert (found.term if found else None) == expected, text


# -- A folder's learned names ---------------------------------------------


def test_a_learned_name_becomes_a_confirmed_term_with_its_wrong_forms():
    [found] = terms_from_learned_names([LearnedName("Bosch", "de", ["Bosh", "Bush"])])

    assert found.text == "Bosch"
    assert found.language is Language.GERMAN
    assert found.common_misrecognitions == ("Bosh", "Bush")
    assert found.confirmation_count == 1


def test_a_learned_name_of_unknown_language_belongs_to_every_language():
    [unknown, invalid] = terms_from_learned_names(
        [LearnedName("Bosch", "unknown"), LearnedName("Smit", "klingon")]
    )

    assert unknown.language is None
    assert invalid.language is None


def test_the_folder_names_come_before_the_profile_terms():
    ordered = folder_terms_first([term("Bosch"), term("Smit")], [term("Acme"), term("Rollout")])

    assert [found.text for found in ordered] == ["Bosch", "Smit", "Acme", "Rollout"]


def test_a_profile_term_that_is_also_a_folder_name_is_sent_once_in_the_folder_place():
    ordered = folder_terms_first(
        [term("Bosch", 1, common_misrecognitions=("Bosh",))],
        [term("Acme"), term("bosch", 3, common_misrecognitions=("Bush",))],
    )

    assert [found.text for found in ordered] == ["Bosch", "Acme"]
    assert ordered[0].common_misrecognitions == ("Bosh", "Bush")
    assert ordered[0].confirmation_count == 4


def test_a_folder_name_and_a_profile_term_of_another_language_are_sent_once_for_every_language():
    # A profile term with no language belongs to every language. Keeping an
    # Afrikaans folder name's language would make a run with Afrikaans off
    # drop a term it sends today, and sending both would spend a place twice.
    for folder_language in (Language.AFRIKAANS, Language.ENGLISH):
        ordered = folder_terms_first(
            [term("Bosch", 1, language=folder_language)],
            [term("Bosch", 2)],
        )

        assert [(found.text, found.language) for found in ordered] == [("Bosch", None)]
        assert ordered[0].confirmation_count == 3


def test_a_folder_name_and_a_profile_term_of_the_same_language_keep_it():
    ordered = folder_terms_first(
        [term("Bosch", 1, language=Language.AFRIKAANS)],
        [term("Bosch", 2, language=Language.AFRIKAANS)],
    )

    assert [(found.text, found.language) for found in ordered] == [
        ("Bosch", Language.AFRIKAANS)
    ]
