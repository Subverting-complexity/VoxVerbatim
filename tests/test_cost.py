"""Tests for telling the user what a run will cost before it starts.

There is no audio here and no network. The arithmetic is simple enough to
check by hand, so most of these tests are about the two things that are
easier to get wrong than the sum: that a service with no rate is reported as
unpriced rather than as free, and that the sentence read out to the user
does not sound more certain than the estimate really is.
"""

from __future__ import annotations

import pytest

from audio_transcriber.transcription.cost import (
    adjudication_rate,
    describe_duration,
    describe_money,
    estimate_cost,
    rates_mapping,
)
from audio_transcriber.transcription.model import Provider

#: A minute of audio at these rates, chosen so the sums are easy to follow.
RATES = {
    Provider.ELEVENLABS: 0.010,
    Provider.OPENAI: 0.006,
    Provider.MICROSOFT: 0.004,
}


def test_each_service_is_charged_for_the_whole_recording():
    estimate = estimate_cost(600.0, [Provider.ELEVENLABS, Provider.OPENAI], RATES)

    assert [cost.provider for cost in estimate.per_provider] == [
        Provider.ELEVENLABS,
        Provider.OPENAI,
    ]
    assert all(cost.minutes == pytest.approx(10.0) for cost in estimate.per_provider)
    assert estimate.per_provider[0].amount == pytest.approx(0.10)
    assert estimate.per_provider[1].amount == pytest.approx(0.06)
    assert estimate.transcription_total == pytest.approx(0.16)
    assert estimate.total == pytest.approx(0.16)


def test_chunking_does_not_multiply_what_a_service_charges():
    """A recording cut into eleven pieces is still one recording of audio.

    OpenAI's chunking is an upload constraint, not a repeat of the work, and
    an estimate that charged per chunk would be several times too high.
    """
    whole = estimate_cost(3600.0, [Provider.OPENAI], RATES)

    assert whole.per_provider[0].minutes == pytest.approx(60.0)
    assert whole.total == pytest.approx(0.36)


def test_a_service_with_no_rate_is_unpriced_rather_than_free():
    """Zero is a claim about the price. Silence is the truth."""
    estimate = estimate_cost(600.0, [Provider.ELEVENLABS, Provider.ASSEMBLYAI], RATES)

    assemblyai = estimate.per_provider[1]
    assert assemblyai.rate_per_minute is None
    assert assemblyai.amount is None
    assert not assemblyai.is_priced
    assert estimate.unpriced_providers == (Provider.ASSEMBLYAI,)
    assert estimate.total == pytest.approx(0.10)
    assert "AssemblyAI" in estimate.summary
    assert "no rate has been entered" in estimate.summary


def test_adjudication_is_left_out_unless_a_number_of_requests_is_given():
    without = estimate_cost(600.0, [Provider.OPENAI], RATES)

    assert without.adjudication_total == 0.0
    assert without.total == pytest.approx(0.06)


def test_adjudication_is_added_when_the_caller_knows_how_many_requests():
    with_adjudication = estimate_cost(
        600.0,
        [Provider.OPENAI],
        RATES,
        adjudication_requests=20,
        adjudication_rate_per_request=0.002,
    )

    assert with_adjudication.adjudication_total == pytest.approx(0.04)
    assert with_adjudication.total == pytest.approx(0.10)
    assert "20 adjudication requests" in with_adjudication.summary


def test_the_summary_says_what_it_costs_and_admits_what_it_cannot_know():
    estimate = estimate_cost(750.0, [Provider.ELEVENLABS, Provider.OPENAI], RATES)

    summary = estimate.summary
    assert "12 minutes and 30 seconds" in summary
    assert "ElevenLabs Scribe and OpenAI" in summary
    assert "0.20 USD" in summary
    assert "estimate" in summary
    # The caveat is the part that matters. A total that looked complete and
    # was not would be worse than no total at all.
    assert "cannot be worked out in advance" in summary
    assert "disagree" in summary


def test_the_summary_says_so_when_no_rates_have_been_entered():
    estimate = estimate_cost(600.0, [Provider.OPENAI], {})

    assert "cannot be costed" in estimate.summary
    assert "Settings" in estimate.summary
    assert estimate.total == 0.0


def test_an_estimate_with_no_services_says_that_too():
    estimate = estimate_cost(600.0, [], RATES)

    assert estimate.per_provider == ()
    assert "No services are switched on" in estimate.summary


def test_a_recording_of_no_length_costs_nothing():
    estimate = estimate_cost(0.0, [Provider.OPENAI], RATES)

    assert estimate.total == 0.0
    assert "0 seconds" in estimate.summary


# -- Where the rates come from -------------------------------------------


def test_rates_may_be_given_by_the_names_used_in_settings():
    """The settings file holds strings, not enumeration members."""
    table = rates_mapping({"openai": 0.006, "elevenlabs": 0.01})

    assert table == {Provider.OPENAI: 0.006, Provider.ELEVENLABS: 0.01}


def test_a_rate_for_a_service_this_application_has_no_adapter_for_is_ignored():
    """A settings file from a later version must not break an estimate."""
    table = rates_mapping({"openai": 0.006, "some-new-service": 0.02})

    assert table == {Provider.OPENAI: 0.006}


def test_rates_may_come_from_a_settings_object_that_answers_per_service():
    class FakeCostSettings:
        def rate_per_minute(self, provider: Provider) -> float | None:
            return 0.005 if provider is Provider.OPENAI else None

    table = rates_mapping(FakeCostSettings())

    assert table == {Provider.OPENAI: 0.005}


def test_the_real_cost_settings_are_understood_as_they_stand():
    """The settings module is the one that actually holds these rates.

    It is imported here rather than in the module under test, so that
    estimating a cost never depends on the shape the settings happen to have
    today. This test is what keeps the two in step.
    """
    from audio_transcriber.settings import CostSettings

    settings = CostSettings()
    table = rates_mapping(settings)

    assert set(table) == set(Provider)
    assert table[Provider.OPENAI] == settings.openai_transcription_per_minute
    assert adjudication_rate(settings) == settings.adjudication_per_request


def test_the_price_of_an_adjudication_comes_from_the_same_settings():
    from audio_transcriber.settings import CostSettings

    settings = CostSettings()

    estimate = estimate_cost(600.0, [Provider.OPENAI], settings, adjudication_requests=5)

    assert estimate.adjudication_rate_per_request == settings.adjudication_per_request
    assert estimate.adjudication_total == pytest.approx(5 * settings.adjudication_per_request)


def test_rates_may_come_from_a_settings_object_holding_a_mapping():
    class FakeCostSettings:
        per_minute_rates = {"microsoft": 0.004}

    table = rates_mapping(FakeCostSettings())

    assert table == {Provider.MICROSOFT: 0.004}


def test_something_that_holds_no_rates_at_all_reports_none_rather_than_zero():
    assert rates_mapping(object()) == {}


# -- Saying numbers out loud ---------------------------------------------


def test_a_length_of_time_is_said_the_way_a_person_would_say_it():
    assert describe_duration(0) == "0 seconds"
    assert describe_duration(1) == "1 second"
    assert describe_duration(90) == "1 minute and 30 seconds"
    assert describe_duration(3600) == "1 hour"
    assert describe_duration(3725) == "1 hour, 2 minutes and 5 seconds"


def test_money_is_said_with_the_currency_after_the_number():
    """A screen reader reads a leading currency symbol unreliably."""
    assert describe_money(1234.5, "USD") == "1,234.50 USD"
    assert describe_money(0.0) == "0.00 USD"
