"""Turning what we know about a recording into what each service will accept.

The user says one thing: these are the languages, these are the speakers,
this is roughly what the recording is about, and these are the words that
matter. Every service then wants that same knowledge in a different shape.
ElevenLabs takes a list of keyterms. OpenAI takes three separate things: some
prose about the recording, a list of keywords, and the candidate language
codes. Microsoft takes a phrase list and will not accept prose at all.
AssemblyAI takes a keyterms prompt whose size depends on which model is
running, and still tolerates its older boosted-word list. Deepgram takes
keyterms measured in tokens rather than counted.

This module is the only place that knows any of that. The adapters ask it for
their own shape and send whatever comes back. The alternative, where every
adapter works out its own list from the vocabulary, spreads five different
sets of somebody else's limits through five different files, and every one of
them goes stale silently the day a service changes its mind.

Three things follow from the limits and are worth stating plainly. Terms are
cut from the end of the list, never from the front, because the list arrives
already ordered with the most valuable terms first. A term that breaks a
service's rule about length is dropped whole rather than shortened, because
half a surname is not a shorter version of that surname; it is a different
word, and asking a service to listen for it makes things worse. And where a
service errors on too much rather than quietly ignoring the excess, the
estimate of how much has been used is deliberately pessimistic, because a
request refused outright costs the user their transcript.

Nothing here depends on Qt or on any provider library, so it can be tested on
its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from audio_transcriber.transcription.model import (
    Language,
    Provider,
    RecordingConfiguration,
)
from audio_transcriber.transcription.providers.base import ProviderCapabilities
from audio_transcriber.transcription.vocabulary import VocabularyTerm

# -- Provider limits -----------------------------------------------------
#
# Every number below is a limit rather than a preference, and they are
# gathered here, and only here, because services change them without telling
# anyone: when one moves, exactly one line in this file has to be corrected
# and every adapter follows. No number like this belongs anywhere else in the
# application.
#
# Each one says whose limit it is. Some are the service's own and must be
# obeyed. Others are ours, chosen conservatively where a service documents no
# limit at all, and a later measurement may safely raise them.

#: ElevenLabs Scribe accepts a list of keyterms.
#:
#: Theirs: the API takes up to 1000 keyterms, each of fewer than 50 characters
#: and at most 5 words. A term breaking any of those rules is rejected, not
#: trimmed, and it takes the whole request down with it rather than being
#: quietly left out, so a term that cannot be sent has to be dropped here.
#:
#: The character rule is written as "less than 50", so the longest term that
#: is actually accepted is 49 characters. The limit and the longest acceptable
#: term are named separately below, because the two numbers are one apart and
#: reading the wrong one costs a request.
ELEVENLABS_KEYTERM_API_LIMIT = 1000
ELEVENLABS_KEYTERM_CHARACTER_LIMIT = 50
ELEVENLABS_MAXIMUM_KEYTERM_CHARACTERS = ELEVENLABS_KEYTERM_CHARACTER_LIMIT - 1
ELEVENLABS_MAXIMUM_KEYTERM_WORDS = 5

#: Theirs: these characters may not appear inside a keyterm at all. A term
#: containing one is refused, and the refusal is of the request rather than of
#: the term, so anything holding a bracket or a backslash has to be left
#: behind. Stripping the character out instead would send a word the user
#: never asked for, which is worse than sending nothing.
ELEVENLABS_UNSUPPORTED_KEYTERM_CHARACTERS = frozenset("<>{}[]\\")

#: Theirs: sending more than this many keyterms bills the request a minimum of
#: 20 seconds of audio, however short the recording really is. That is the
#: charge a limit can avoid, and 100 terms is the largest list that avoids it.
#:
#: The 20 per cent surcharge that ElevenLabs adds for keyterms is a separate
#: matter and is not avoidable by sending fewer: it applies to any request
#: that carries keyterms at all. Nothing is gained by cutting the list below
#: the threshold, which is why the limit sits exactly on it rather than under.
ELEVENLABS_KEYTERM_MINIMUM_CHARGE_THRESHOLD = 100

#: Ours, and a cost decision rather than a capability one. A user with a long
#: vocabulary and a reason to pay the 20-second minimum could reasonably want
#: this raised, which is exactly why it is one named number.
ELEVENLABS_MAXIMUM_KEYTERMS = ELEVENLABS_KEYTERM_MINIMUM_CHARGE_THRESHOLD

#: OpenAI ``gpt-transcribe`` takes context through three separate parameters,
#: so its terms no longer have to be smuggled into the prose.
#:
#: Theirs: the prompt is capped by length.
OPENAI_MAXIMUM_PROMPT_CHARACTERS = 1024

#: Ours: no cap on ``keywords`` is documented. This is a conservative guess,
#: not a measured limit, and should be raised once somebody has established
#: what the service actually refuses.
OPENAI_MAXIMUM_KEYWORDS = 200

#: Theirs: a keyword may not contain any of these, and the documentation is
#: explicit that the refusal is of the whole request rather than of the one
#: keyword. A single term like ``<inaudible>`` in somebody's vocabulary would
#: otherwise cost the recording every OpenAI chunk rather than that one word,
#: so anything holding one has to be left behind here.
#:
#: The two line endings are in the set because the same rule requires each
#: keyword to stay on one line. Stripping the offending character out instead
#: would send a word the user never asked us to listen for, which is worse
#: than sending nothing.
OPENAI_UNSUPPORTED_KEYWORD_CHARACTERS = frozenset("<>\r\n")

#: Microsoft accepts a ``phraseList`` for what its documentation calls entity
#: biasing.
#:
#: Ours: no cap is documented, so this is a conservative choice of our own.
MICROSOFT_MAXIMUM_PHRASES = 1024

#: AssemblyAI's keyterms prompt is capped by the model that is running, and
#: the model is chosen per span rather than per recording.
#:
#: Theirs: 200 terms on Universal-2, 1000 on Universal-3.5 Pro, at most 6
#: words per phrase. Universal-3.5 Pro also accepts a prompt of up to 1500
#: words; Universal-2 does not accept one at all.
ASSEMBLYAI_MAXIMUM_KEYTERMS_UNIVERSAL_2 = 200
ASSEMBLYAI_MAXIMUM_KEYTERMS_UNIVERSAL_3_5_PRO = 1000
ASSEMBLYAI_MAXIMUM_KEYTERM_WORDS = 6
ASSEMBLYAI_MAXIMUM_PROMPT_WORDS = 1500

#: The AssemblyAI model names this module recognises. The adapter may pass
#: whatever string settings hold; anything unrecognised falls back to the
#: smaller cap, which is the safe direction to be wrong in.
ASSEMBLYAI_UNIVERSAL_2 = "universal-2"
ASSEMBLYAI_UNIVERSAL_3_5_PRO = "universal-3-5-pro"

#: Deepgram measures its keyterms in tokens across the whole request rather
#: than counting them, and a request over the budget is refused outright
#: rather than silently truncated.
#:
#: Theirs: 500 tokens per request.
DEEPGRAM_MAXIMUM_KEYTERM_TOKENS = 500

#: Ours, and deliberately pessimistic. Deepgram's own guidance puts 500 tokens
#: at roughly 100 words, which is the ratio used here. We cannot run their
#: tokeniser, and the cost of guessing low is a refused request rather than a
#: shorter list, so the estimate errs upwards.
DEEPGRAM_ESTIMATED_TOKENS_PER_WORD = 5

#: The languages allowed when the user has not enabled Afrikaans, and when
#: they have. These are tuples rather than a set so that the order in which
#: they are named to a service is always the same.
LANGUAGES_WITHOUT_AFRIKAANS: tuple[Language, ...] = (Language.ENGLISH, Language.GERMAN)
LANGUAGES_WITH_AFRIKAANS: tuple[Language, ...] = (
    Language.ENGLISH,
    Language.GERMAN,
    Language.AFRIKAANS,
)


@dataclass(frozen=True)
class ContextPackage:
    """Everything worth telling a service about a recording, before shaping.

    Built once per recording and then adapted for each service in turn. It is
    kept separate from the per-service shapes so that the question "what do we
    know?" is answered in one place and the question "what will this service
    take?" in another.
    """

    languages: tuple[Language, ...] = LANGUAGES_WITHOUT_AFRIKAANS
    """The languages the recording may contain, and the only ones any service
    is told about."""

    afrikaans_enabled: bool = False
    terms: tuple[VocabularyTerm, ...] = ()
    """The resolved vocabulary, most valuable first. Nothing here re-orders
    it; the limits cut from the end and rely on that order."""

    speaker_names: tuple[str, ...] = ()
    expected_speaker_count: int = 1
    recording_context: str = ""
    """The sentence or two the user wrote about the recording, in their own
    words."""

    @property
    def term_texts(self) -> tuple[str, ...]:
        return tuple(term.text for term in self.terms)

    @property
    def language_codes(self) -> tuple[str, ...]:
        """The allowed languages as ISO 639-1 codes, for the services that take them.

        This is where the Afrikaans switch becomes visible to a service: two
        codes when it is off and three when it is on.
        """
        return tuple(language.value for language in self.languages)

    @property
    def language_sentence(self) -> str:
        """How the allowed languages are named to a service that takes prose."""
        names = [language.display_name for language in self.languages]
        if not names:
            return ""
        if len(names) == 1:
            spoken = names[0]
        else:
            spoken = f"{', '.join(names[:-1])} or {names[-1]}"
        return f"The recording is in {spoken}."

    @property
    def speaker_sentence(self) -> str:
        """How the speakers are named, or how many there are if they are not."""
        if self.speaker_names:
            names = list(self.speaker_names)
            if len(names) == 1:
                return f"The speaker is {names[0]}."
            joined = f"{', '.join(names[:-1])} and {names[-1]}"
            return f"The speakers are {joined}."
        if self.expected_speaker_count > 1:
            return f"There are about {self.expected_speaker_count} speakers."
        return ""


def build_context_package(
    configuration: RecordingConfiguration,
    terms: Sequence[VocabularyTerm] = (),
) -> ContextPackage:
    """Gather what is known about one recording into a package.

    ``terms`` is the already-resolved vocabulary, in the order
    :func:`~audio_transcriber.transcription.vocabulary.resolve_terms` put it.

    Whether Afrikaans is enabled decides more than which languages are listed.
    With it off, no part of the package mentions Afrikaans and terms the user
    marked as Afrikaans are left out altogether. That is deliberate and it is
    what the specification asks for: switching Afrikaans off is meant to
    remove the Afrikaans handling rather than run it and discard the answer,
    because a service told that Afrikaans is possible will occasionally decide
    it heard some, in a recording that never contained a word of it.

    The user's own sentence about the recording is passed through as they
    wrote it. Their prose is not rewritten, even to remove a word: what
    switching Afrikaans off removes is everything the application would
    otherwise add or infer.
    """
    if configuration.afrikaans_enabled:
        languages = LANGUAGES_WITH_AFRIKAANS
    else:
        languages = LANGUAGES_WITHOUT_AFRIKAANS
    allowed = [
        term
        for term in terms
        if configuration.afrikaans_enabled or term.language is not Language.AFRIKAANS
    ]
    names = tuple(name.strip() for name in configuration.known_speakers if name.strip())
    return ContextPackage(
        languages=languages,
        afrikaans_enabled=configuration.afrikaans_enabled,
        terms=tuple(allowed),
        speaker_names=names,
        expected_speaker_count=max(1, configuration.expected_speaker_count),
        recording_context=configuration.recording_context.strip(),
    )


@dataclass(frozen=True)
class ProviderContext:
    """The context package in the shape one service will accept.

    ``parameters`` is what the adapter passes to its client library, already
    under the names that library uses. The other fields are the same content
    in a form the rest of the application can read without knowing what a
    given service happens to call things, which is what the request record
    needs when it writes down what was sent.
    """

    provider: Provider
    terms: tuple[str, ...] = ()
    """The terms that fitted, in order. Never a re-ordering of the input."""

    prompt: str = ""
    languages: tuple[str, ...] = ()
    """The ISO 639-1 codes sent, where the service takes a language list."""

    parameters: dict[str, Any] = field(default_factory=dict)
    dropped_term_count: int = 0
    """How many terms this service would not take.

    Worth knowing rather than hiding: a user whose list has outgrown a
    service should be told that some of it is not reaching them.
    """


def adapt_for(
    package: ContextPackage,
    provider: Provider,
    capabilities: ProviderCapabilities | None = None,
    model: str | None = None,
) -> ProviderContext:
    """Shape ``package`` for one service, respecting its limits.

    ``model`` matters only for AssemblyAI, where both the number of keyterms
    and whether a prompt is accepted at all depend on which model will run.
    Left unsaid, the smaller limits are used, because a request refused for
    being too large costs more than a few terms left behind.

    Where ``capabilities`` is given, it is obeyed: a service that says it does
    not accept vocabulary is shaped from a package with no terms in it, and
    one that says it takes no prose has its prompt removed.
    """
    accepts_terms = capabilities is None or capabilities.vocabulary_biasing
    accepts_prompt = capabilities is None or capabilities.context_prompt

    effective = package if accepts_terms else replace(package, terms=())
    context = _shape(effective, provider, model)

    if not accepts_terms:
        context = replace(context, dropped_term_count=len(package.terms))
    if not accepts_prompt and context.prompt:
        parameters = dict(context.parameters)
        parameters.pop("prompt", None)
        context = replace(context, prompt="", parameters=parameters)
    return context


def _shape(package: ContextPackage, provider: Provider, model: str | None) -> ProviderContext:
    if provider is Provider.ELEVENLABS:
        return _for_elevenlabs(package)
    if provider is Provider.OPENAI:
        return _for_openai(package)
    if provider is Provider.MICROSOFT:
        return _for_microsoft(package)
    if provider is Provider.ASSEMBLYAI:
        return _for_assemblyai(package, model)
    if provider is Provider.DEEPGRAM:
        return _for_deepgram(package)
    # A service added to the enum but not yet given a mechanism here is sent
    # nothing, which is the harmless outcome rather than a guessed one.
    return ProviderContext(provider=provider)


def _for_elevenlabs(package: ContextPackage) -> ProviderContext:
    """ElevenLabs takes keyterms, and Scribe accepts no prose at all."""
    terms, dropped = _fit_terms(
        package.term_texts,
        ELEVENLABS_MAXIMUM_KEYTERMS,
        maximum_characters=ELEVENLABS_MAXIMUM_KEYTERM_CHARACTERS,
        maximum_words=ELEVENLABS_MAXIMUM_KEYTERM_WORDS,
        forbidden_characters=ELEVENLABS_UNSUPPORTED_KEYTERM_CHARACTERS,
    )
    return ProviderContext(
        provider=Provider.ELEVENLABS,
        terms=terms,
        parameters={"keyterms": list(terms)},
        dropped_term_count=dropped,
    )


def _for_openai(package: ContextPackage) -> ProviderContext:
    """OpenAI takes its context through three parameters rather than one.

    The prompt carries only what cannot be said any other way: who is speaking
    and what the recording is about. The terms travel as their own list and
    the languages as their own codes, so a long vocabulary no longer competes
    with the description of the recording for room in a single string.

    A term holding a character the service refuses is dropped here rather than
    sent, because that refusal takes the whole request with it.
    """
    prompt = _trim_at_word_boundary(
        _describe(package, include_languages=False),
        OPENAI_MAXIMUM_PROMPT_CHARACTERS,
    )
    keywords, dropped = _fit_terms(
        package.term_texts,
        OPENAI_MAXIMUM_KEYWORDS,
        forbidden_characters=OPENAI_UNSUPPORTED_KEYWORD_CHARACTERS,
    )
    return ProviderContext(
        provider=Provider.OPENAI,
        terms=keywords,
        prompt=prompt,
        languages=package.language_codes,
        parameters={
            "prompt": prompt,
            "keywords": list(keywords),
            "languages": list(package.language_codes),
        },
        dropped_term_count=dropped,
    )


def _for_microsoft(package: ContextPackage) -> ProviderContext:
    """Microsoft biases on a phrase list, and takes no prose whatsoever.

    Prompt tuning is unsupported on this model, so everything worth saying has
    to be said as phrases. There is deliberately no prompt here to be removed
    later: sending one would be rejected rather than ignored.
    """
    terms, dropped = _fit_terms(package.term_texts, MICROSOFT_MAXIMUM_PHRASES)
    return ProviderContext(
        provider=Provider.MICROSOFT,
        terms=terms,
        parameters={"phrases": list(terms)},
        dropped_term_count=dropped,
    )


def _for_assemblyai(package: ContextPackage, model: str | None) -> ProviderContext:
    """AssemblyAI is shaped around its keyterms prompt.

    How many keyterms it will take depends on the model, and so does whether
    it accepts prose at all, but the model is chosen per span rather than per
    recording: Universal-2 handles the Afrikaans spans and Universal-3.5 Pro
    the rest. When the caller has not said which is running, the smaller
    limits apply and no prompt is sent, because being refused for sending too
    much is a worse outcome than sending less than we could have.
    """
    resolved = _assemblyai_model(model)
    keyterms, dropped = _fit_terms(
        package.term_texts,
        _assemblyai_keyterm_limit(resolved),
        maximum_words=ASSEMBLYAI_MAXIMUM_KEYTERM_WORDS,
    )
    parameters: dict[str, Any] = {
        "keyterms_prompt": list(keyterms),
    }
    prompt = ""
    if resolved == ASSEMBLYAI_UNIVERSAL_3_5_PRO:
        prompt = _trim_to_words(
            _describe(package, include_languages=True),
            ASSEMBLYAI_MAXIMUM_PROMPT_WORDS,
        )
        if prompt:
            parameters["prompt"] = prompt
    return ProviderContext(
        provider=Provider.ASSEMBLYAI,
        terms=keyterms,
        prompt=prompt,
        parameters=parameters,
        dropped_term_count=dropped,
    )


def _assemblyai_model(model: str | None) -> str | None:
    """Recognise an AssemblyAI model name however it happens to be spelled."""
    if not model:
        return None
    text = model.strip().casefold().replace("_", "-").replace(" ", "-")
    if text.startswith(ASSEMBLYAI_UNIVERSAL_2):
        return ASSEMBLYAI_UNIVERSAL_2
    if "universal-3" in text and "pro" in text:
        return ASSEMBLYAI_UNIVERSAL_3_5_PRO
    return None


def _assemblyai_keyterm_limit(resolved_model: str | None) -> int:
    if resolved_model == ASSEMBLYAI_UNIVERSAL_3_5_PRO:
        return ASSEMBLYAI_MAXIMUM_KEYTERMS_UNIVERSAL_3_5_PRO
    return ASSEMBLYAI_MAXIMUM_KEYTERMS_UNIVERSAL_2


def _for_deepgram(package: ContextPackage) -> ProviderContext:
    """Deepgram counts its keyterms in tokens, and refuses a request over budget.

    That is why the terms are added one at a time against an estimated cost
    rather than simply counted: there is no number of terms that is always
    safe, because a handful of long phrases can cost more than many short
    names. Deepgram takes no prompt.
    """
    terms, dropped = _fit_terms_within_tokens(
        package.term_texts, DEEPGRAM_MAXIMUM_KEYTERM_TOKENS
    )
    return ProviderContext(
        provider=Provider.DEEPGRAM,
        terms=terms,
        parameters={"keyterm": list(terms)},
        dropped_term_count=dropped,
    )


def _fit_terms(
    texts: Sequence[str],
    maximum_terms: int,
    maximum_characters: int | None = None,
    maximum_words: int | None = None,
    forbidden_characters: frozenset[str] | None = None,
) -> tuple[tuple[str, ...], int]:
    """Return as many terms as a service will take, and how many were left out.

    A term that breaks a rule about itself is dropped and the next one
    considered, because it was never going to be accepted whatever room there
    was. That covers being too long, having too many words, and containing a
    character the service refuses. Running out of room is different:
    everything after that point goes, because the list is ordered by value and
    letting a later, shorter term jump the queue would quietly reverse that
    ranking.
    """
    kept: list[str] = []
    dropped = 0
    for position, text in enumerate(texts):
        if len(kept) >= maximum_terms:
            dropped += len(texts) - position
            break
        if maximum_characters is not None and len(text) > maximum_characters:
            dropped += 1
            continue
        if maximum_words is not None and _word_count(text) > maximum_words:
            dropped += 1
            continue
        if forbidden_characters and any(
            character in forbidden_characters for character in text
        ):
            dropped += 1
            continue
        kept.append(text)
    return tuple(kept), dropped


def _fit_terms_within_tokens(
    texts: Sequence[str],
    token_budget: int,
) -> tuple[tuple[str, ...], int]:
    """Fill a token budget from the front of the list, stopping before it is exceeded."""
    kept: list[str] = []
    spent = 0
    for position, text in enumerate(texts):
        cost = _estimated_tokens(text)
        if spent + cost > token_budget:
            return tuple(kept), len(texts) - position
        kept.append(text)
        spent += cost
    return tuple(kept), 0


def _estimated_tokens(text: str) -> int:
    """A deliberately generous guess at what a term costs in tokens."""
    return max(1, _word_count(text)) * DEEPGRAM_ESTIMATED_TOKENS_PER_WORD


def _word_count(text: str) -> int:
    return len(text.split())


def _describe(package: ContextPackage, include_languages: bool) -> str:
    """Write the recording out as prose, for the services that take prose.

    The languages are left out where the service has a parameter of its own
    for them, since saying the same thing twice in two forms invites the two
    to disagree the day one of them is changed and the other forgotten.
    """
    lines = [
        line
        for line in (
            package.language_sentence if include_languages else "",
            package.speaker_sentence,
            package.recording_context,
        )
        if line
    ]
    return "\n".join(lines)


def _trim_at_word_boundary(text: str, maximum_characters: int) -> str:
    """Cut ``text`` to length without leaving half a word at the end.

    A sentence stopping mid-word reads as nonsense to a model being asked to
    make sense of it, which is the opposite of what a prompt is for.
    """
    if len(text) <= maximum_characters:
        return text
    cut = text[:maximum_characters]
    space = cut.rfind(" ")
    return cut[:space].rstrip() if space > 0 else cut.rstrip()


def _trim_to_words(text: str, maximum_words: int) -> str:
    """Cut ``text`` to a number of words, for the services that count them."""
    words = text.split()
    if len(words) <= maximum_words:
        return text
    return " ".join(words[:maximum_words])
