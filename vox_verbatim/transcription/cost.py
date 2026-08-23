"""Telling the user what a run is about to cost, before it starts.

Running four speech-to-text services over the same recording costs four
times what running one of them does, and the person pressing the button is
paying for it. A two-hour recording that quietly turns into several pounds
is not a pleasant surprise, and the only fair moment to mention it is before
the run rather than in next month's invoice.

So this module works out an estimate from the rates in Settings and hands
back two things: a breakdown, one line per service, for anyone who wants to
see where the money goes, and one plain sentence saying the total, for the
status line and for a screen reader to read out.

The estimate is honest about being an estimate, and honest about what it
leaves out. The full passes over the recording can be worked out exactly:
the recording is a known length and each service charges by the minute.
Escalation and adjudication cannot, because how many passages need a second
opinion depends on how much the services disagree, and how much they
disagree is not knowable until they have answered. A guess dressed up as a
figure would be worse than no figure, so those are named as what they are
and, where the user asks for them, estimated from a number of requests the
caller supplies rather than one invented here.

A service with no rate set is reported as having no rate rather than as
costing nothing. Zero is a claim about the price; silence is the truth.

The rates come from the settings module, but this module does not depend on
it. Pass a plain mapping of service to rate, or an object that exposes them,
and the import is done lazily only when nothing was passed. That keeps cost
estimation testable on its own and unblocked by whatever shape the settings
happen to take.

Nothing here depends on Qt, so it can be tested on its own.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from vox_verbatim.transcription.model import Provider

_log = logging.getLogger(__name__)

#: What the amounts are in when nobody has said otherwise. It is only ever a
#: label: no conversion happens anywhere in this module, and mixing rates in
#: different currencies would simply add up to a wrong number.
DEFAULT_CURRENCY = "USD"

#: What ElevenLabs adds to a request that carries keyterms, as a fraction of
#: the per-minute price. It applies to any request with keyterms at all,
#: however few, so every run with a vocabulary pays it. The figure is theirs
#: and is also recorded in :mod:`vox_verbatim.transcription.context`,
#: where the keyterm limits live.
ELEVENLABS_KEYTERM_SURCHARGE = 0.20


@dataclass(frozen=True)
class ProviderCost:
    """What one service is expected to charge for one recording."""

    provider: Provider
    minutes: float
    rate_per_minute: float | None
    """``None`` where no rate has been set for this service."""

    surcharge: float = 0.0
    """What this service adds on top of its per-minute price for this run,
    as a fraction. Zero for nearly everything; see
    :data:`ELEVENLABS_KEYTERM_SURCHARGE` for the one case that is not."""

    surcharge_reason: str = ""
    """Why the surcharge applies, in words a person can read."""

    @property
    def amount(self) -> float | None:
        if self.rate_per_minute is None:
            return None
        return self.minutes * self.rate_per_minute * (1.0 + self.surcharge)

    @property
    def is_surcharged(self) -> bool:
        return self.surcharge > 0.0

    @property
    def is_priced(self) -> bool:
        return self.rate_per_minute is not None


@dataclass(frozen=True)
class CostEstimate:
    """What a whole run is expected to cost, and how that was arrived at."""

    duration_seconds: float
    per_provider: tuple[ProviderCost, ...]
    adjudication_requests: int
    adjudication_rate_per_request: float
    currency: str

    @property
    def summary(self) -> str:
        """The whole estimate as sentences, to be read aloud as well as seen."""
        return describe_estimate(self)

    @property
    def transcription_total(self) -> float:
        return sum(cost.amount or 0.0 for cost in self.per_provider)

    @property
    def adjudication_total(self) -> float:
        return self.adjudication_requests * self.adjudication_rate_per_request

    @property
    def total(self) -> float:
        return self.transcription_total + self.adjudication_total

    @property
    def unpriced_providers(self) -> tuple[Provider, ...]:
        """The services that will be called but whose rates are not set.

        The total leaves these out, so anything in here is money the estimate
        does not account for, and the summary says so.
        """
        return tuple(cost.provider for cost in self.per_provider if not cost.is_priced)

    @property
    def surcharged(self) -> tuple[ProviderCost, ...]:
        """The priced services whose figure includes a surcharge for this run."""
        return tuple(
            cost for cost in self.per_provider if cost.is_priced and cost.is_surcharged
        )


def estimate_cost(
    duration_seconds: float,
    providers: Iterable[Provider],
    rates: Mapping[Provider | str, float] | Any | None = None,
    adjudication_requests: int = 0,
    adjudication_rate_per_request: float | None = None,
    currency: str = DEFAULT_CURRENCY,
    vocabulary_terms_sent: bool = False,
) -> CostEstimate:
    """Estimate what transcribing one recording with these services will cost.

    Every service is charged for the whole recording, because every service
    is sent the whole recording. Chunking does not change that: a recording
    cut into eleven pieces for OpenAI is still one recording's worth of audio
    and is charged as such, and the small overlap between the pieces is far
    too little to be worth pretending to account for.

    ``adjudication_requests`` is a number the caller believes, not one this
    module guesses. Left at zero, adjudication is left out of the total and
    named in the summary as something that will be charged and cannot be
    predicted. The price of one request is read from the same settings the
    per-minute rates came from unless it is given here.

    ``vocabulary_terms_sent`` says whether the run will send the services a
    list of terms to listen for. ElevenLabs charges a fifth more for any
    request that carries them, and its line is raised by that much so that
    the estimate matches the invoice. Left false, which is the default, the
    estimate is the one for a run with no vocabulary.
    """
    minutes = max(0.0, duration_seconds) / 60.0
    table = rates_mapping(rates)
    if adjudication_rate_per_request is None:
        adjudication_rate_per_request = adjudication_rate(rates)
    costs = tuple(
        _provider_cost(provider, minutes, table.get(provider), vocabulary_terms_sent)
        for provider in providers
    )
    return CostEstimate(
        duration_seconds=max(0.0, duration_seconds),
        per_provider=costs,
        adjudication_requests=max(0, adjudication_requests),
        adjudication_rate_per_request=max(0.0, adjudication_rate_per_request),
        currency=currency,
    )


def _provider_cost(
    provider: Provider,
    minutes: float,
    rate_per_minute: float | None,
    vocabulary_terms_sent: bool,
) -> ProviderCost:
    """One service's line, with whatever this run makes it charge extra."""
    if provider is Provider.ELEVENLABS and vocabulary_terms_sent:
        return ProviderCost(
            provider=provider,
            minutes=minutes,
            rate_per_minute=rate_per_minute,
            surcharge=ELEVENLABS_KEYTERM_SURCHARGE,
            surcharge_reason=(
                "ElevenLabs charges that much more for a request that carries vocabulary terms"
            ),
        )
    return ProviderCost(provider=provider, minutes=minutes, rate_per_minute=rate_per_minute)


def describe_estimate(estimate: CostEstimate) -> str:
    """Write the estimate out as sentences a person can hear and understand.

    It says what is being transcribed, by whom, and for how much; then that
    the figure is an estimate; then what is not in it. The last of those is
    the part that matters most, because a total that looked complete and was
    not would be worse than no total at all.
    """
    names = [cost.provider.display_name for cost in estimate.per_provider]
    length = describe_duration(estimate.duration_seconds)

    if not names:
        return f"No services are switched on, so {length} of audio would not be transcribed."

    who = _list_in_words(names)
    if not any(cost.is_priced for cost in estimate.per_provider):
        return (
            f"Transcribing {length} of audio with {who} cannot be costed, because no "
            "per-minute rates have been entered in Settings."
        )

    sentences = [
        f"Transcribing {length} of audio with {who} is estimated to cost about "
        f"{describe_money(estimate.transcription_total, estimate.currency)}."
    ]

    for cost in estimate.surcharged:
        sentences.append(
            f"The {cost.provider.display_name} figure is {_describe_fraction(cost.surcharge)} "
            f"higher than its per-minute rate alone, because {cost.surcharge_reason}."
        )

    missing = estimate.unpriced_providers
    if missing:
        unpriced = _list_in_words([provider.display_name for provider in missing])
        ending = "no rate has been entered for it" if len(missing) == 1 else (
            "no rates have been entered for them"
        )
        sentences.append(f"That leaves out {unpriced}, because {ending}.")

    if estimate.adjudication_requests:
        sentences.append(
            f"A further {describe_money(estimate.adjudication_total, estimate.currency)} is "
            f"allowed for {estimate.adjudication_requests} adjudication requests, giving "
            f"about {describe_money(estimate.total, estimate.currency)} in all."
        )

    sentences.append(
        "This is an estimate from the rates in Settings. Second opinions on disputed "
        "passages, and asking a language model to decide between them, are charged on top "
        "of it and cannot be worked out in advance, because how much of the recording "
        "needs them depends on how much the services turn out to disagree."
    )
    return " ".join(sentences)


def _describe_fraction(fraction: float) -> str:
    """Say a fraction as people say it: "20 per cent"."""
    percent = fraction * 100.0
    if percent == int(percent):
        return f"{int(percent)} per cent"
    return f"{percent:.1f} per cent"


def describe_duration(seconds: float) -> str:
    """Say a length of time the way a person would say it."""
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, plain_seconds = divmod(remainder, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours} hour" if hours == 1 else f"{hours} hours")
    if minutes:
        parts.append(f"{minutes} minute" if minutes == 1 else f"{minutes} minutes")
    if plain_seconds or not parts:
        unit = "second" if plain_seconds == 1 else "seconds"
        parts.append(f"{plain_seconds} {unit}")
    return _list_in_words(parts)


def describe_money(amount: float, currency: str = DEFAULT_CURRENCY) -> str:
    """Say an amount of money in a form a screen reader reads sensibly.

    The currency follows the number as a word rather than preceding it as a
    symbol, because a screen reader reads a leading symbol unreliably and
    because the currency here is whatever the user typed in Settings rather
    than one of a fixed few with symbols of their own.
    """
    return f"{amount:,.2f} {currency}"


def rates_mapping(source: Mapping[Provider | str, float] | Any | None) -> dict[Provider, float]:
    """Turn whatever the caller has into a rate per service.

    Three shapes are accepted, so that this module is not waiting on the
    exact form the settings take. A plain mapping is used as it is, keyed
    either by :class:`Provider` or by the strings behind it. An object with a
    ``rate_per_minute`` method is asked for each service in turn. An object
    with a ``per_minute_rates`` mapping has that mapping read.

    Nothing is passed and nothing is found, and the answer is an empty
    mapping, which reports every service as unpriced rather than as free.
    """
    if source is None:
        source = _settings_cost_defaults()
    if source is None:
        return {}

    if isinstance(source, Mapping):
        return _from_mapping(source)

    for method_name in _RATE_METHOD_NAMES:
        rate_of = getattr(source, method_name, None)
        if not callable(rate_of):
            continue
        rates: dict[Provider, float] = {}
        for provider in Provider:
            try:
                value = rate_of(provider)
            except Exception:  # a settings object must not break an estimate
                _log.debug("No rate could be read for %s.", provider, exc_info=True)
                continue
            if value is not None:
                rates[provider] = float(value)
        return rates

    table = getattr(source, "per_minute_rates", None)
    if isinstance(table, Mapping):
        return _from_mapping(table)

    _log.debug("Cost rates of type %s were not understood.", type(source).__name__)
    return {}


#: The names a settings object may use for "what does this service charge a
#: minute". More than one is listed so that this module is not broken by the
#: settings being renamed, which is a poor reason to lose a cost estimate.
_RATE_METHOD_NAMES = ("per_minute_for", "rate_per_minute")

#: Where a settings object keeps the price of one adjudication request.
_ADJUDICATION_ATTRIBUTE_NAMES = ("adjudication_per_request", "adjudication_rate_per_request")


def adjudication_rate(source: Any | None) -> float:
    """What one adjudication request costs, according to these settings.

    Zero where the settings do not say, which leaves adjudication out of the
    total rather than guessing at it.
    """
    if source is None:
        source = _settings_cost_defaults()
    if source is None:
        return 0.0
    if isinstance(source, Mapping):
        for name in _ADJUDICATION_ATTRIBUTE_NAMES:
            if name in source:
                return float(source[name] or 0.0)
        return 0.0
    for name in _ADJUDICATION_ATTRIBUTE_NAMES:
        value = getattr(source, name, None)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _from_mapping(source: Mapping[Any, Any]) -> dict[Provider, float]:
    rates: dict[Provider, float] = {}
    for key, value in source.items():
        if value is None:
            continue
        try:
            provider = key if isinstance(key, Provider) else Provider(str(key))
        except ValueError:
            _log.debug("The cost rates name a service this application has no adapter for: %r", key)
            continue
        rates[provider] = float(value)
    return rates


def _settings_cost_defaults() -> Any | None:
    """Fetch the cost settings, if that class exists yet.

    Imported here rather than at the top of the file, so that this module
    neither depends on the settings module nor waits for it. A version of the
    application without cost settings simply reports every service as
    unpriced.
    """
    try:
        from vox_verbatim.settings import CostSettings  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        return None
    try:
        return CostSettings()
    except Exception:  # pragma: no cover - a settings class that will not build
        _log.debug("The cost settings could not be built.", exc_info=True)
        return None


def _list_in_words(items: list[str]) -> str:
    """Join names the way they are spoken: a, b and c."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"
