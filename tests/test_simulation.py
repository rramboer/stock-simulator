"""Unit tests for the simulation model."""

import math
import random
from statistics import mean

import pytest

from stock_simulator import simulation as sim_module
from stock_simulator.simulation import (
    BUY,
    FOLLOW_NARRATIVE_WEIGHT,
    FOMO_SENSITIVITY,
    HOLD,
    PANIC_REVERSAL,
    SELL,
    SENTIMENT_NOISE,
    SENTIMENT_SKEWS,
    Investor,
    Simulation,
    SimulationConfig,
    count_states,
    percent_change,
    state_for_sentiment,
)


class ScriptedRandom(random.Random):
    """A ``Random`` whose ``random()`` returns a fixed sequence, for rule-level tests.

    ``update_status`` draws, in order: noise (via ``uniform``), FOMO, panic, and
    then the narrative draw only when panic did not fire. A draw of 0.5 for the
    noise means zero drift; 0.0 always passes a probability check and 1.0 never does.
    """

    def __init__(self, values: list[float]) -> None:
        super().__init__(0)
        self._values = list(values)

    def random(self) -> float:  # type: ignore[override]
        return self._values.pop(0)


NO_NOISE = 0.5
FIRE = 0.0
SKIP = 1.0
ALWAYS = SimulationConfig(fomo_threshold=1.0, panic_prob=1.0)


class TestPercentChange:
    def test_gain(self):
        assert percent_change(100.0, 110.0) == pytest.approx(10.0)

    def test_loss(self):
        assert percent_change(100.0, 90.0) == pytest.approx(-10.0)

    def test_no_change(self):
        assert percent_change(100.0, 100.0) == pytest.approx(0.0)

    def test_zero_previous_price_rejected(self):
        with pytest.raises(ValueError, match="nonzero"):
            percent_change(0.0, 50.0)


class TestStateForSentiment:
    @pytest.mark.parametrize(
        ("sentiment", "expected"),
        [
            (-1.0, SELL),
            (-0.334, SELL),
            (-0.333, HOLD),
            (0.0, HOLD),
            (0.333, HOLD),
            (0.334, BUY),
            (1.0, BUY),
        ],
    )
    def test_thresholds(self, sentiment, expected):
        assert state_for_sentiment(sentiment) == expected


class TestSimulationConfig:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"num_investors": 0},
            {"max_shares": 0},
            {"start_price": 0.0},
            {"start_price": -5.0},
            {"num_trades": -1},
            {"num_trades": 100},  # 100 investors have only 99 counterparties
            {"fomo_threshold": 1.5},
            {"fomo_threshold": -0.1},
            {"panic_prob": 2.0},
            {"shock_prob": -0.5},
            {"sentiment_skew": 1.5},
            {"narrative_half_life": 0.0},
            {"narrative_half_life": -10.0},
            {"value_pull": -0.1},
            {"event_step": -1},
        ],
    )
    def test_invalid_values_rejected(self, kwargs):
        with pytest.raises(ValueError):
            SimulationConfig(**kwargs)

    def test_edge_values_accepted(self):
        SimulationConfig(num_trades=0)
        SimulationConfig(num_investors=1, num_trades=0)
        SimulationConfig(num_trades=99)
        SimulationConfig(fomo_threshold=0.0, panic_prob=1.0, shock_prob=1.0)
        SimulationConfig(narrative_half_life=math.inf)
        SimulationConfig(event_step=0)
        SimulationConfig(event_step=10_000)


class TestInvestor:
    def test_state_is_derived_from_sentiment(self):
        investor = Investor(sentiment=0.9, holdings=10.0)
        assert investor.state == BUY
        investor.sentiment = -0.9
        assert investor.state == SELL
        investor.sentiment = 0.1
        assert investor.state == HOLD

    def test_trade_moves_mean_eagerness_fraction_of_seller_holdings(self):
        buyer = Investor(sentiment=0.9, holdings=100.0)
        seller = Investor(sentiment=-0.5, holdings=200.0)
        volume = buyer.trade(seller)
        assert volume == pytest.approx(140.0)  # mean eagerness (0.9 + 0.5) / 2 = 0.7
        assert buyer.holdings == pytest.approx(240.0)
        assert seller.holdings == pytest.approx(60.0)

    def test_trade_direction_does_not_depend_on_initiator(self):
        buyer = Investor(sentiment=0.9, holdings=100.0)
        seller = Investor(sentiment=-0.5, holdings=200.0)
        volume = seller.trade(buyer)
        assert volume == pytest.approx(140.0)
        assert buyer.holdings == pytest.approx(240.0)
        assert seller.holdings == pytest.approx(60.0)

    def test_any_buyer_and_seller_trade(self):
        # A lukewarm buyer and a convinced seller still transact.
        buyer = Investor(sentiment=0.4, holdings=100.0)
        seller = Investor(sentiment=-0.9, holdings=200.0)
        assert buyer.trade(seller) == pytest.approx(130.0)

    def test_trade_requires_buyer_and_seller(self):
        one = Investor(sentiment=0.9, holdings=100.0)
        other = Investor(sentiment=0.8, holdings=200.0)
        assert one.trade(other) == 0.0
        assert one.holdings == 100.0 and other.holdings == 200.0

    def test_self_trade_is_noop(self):
        investor = Investor(sentiment=0.9, holdings=100.0)
        assert investor.trade(investor) == 0.0
        assert investor.holdings == 100.0

    def test_trade_never_overdraws_seller(self):
        buyer = Investor(sentiment=1.0, holdings=0.0)
        seller = Investor(sentiment=-1.0, holdings=50.0)
        assert buyer.trade(seller) == pytest.approx(50.0)
        assert seller.holdings == pytest.approx(0.0)

    def test_fomo_scales_percent_move_into_sentiment_units(self):
        investor = Investor(sentiment=0.0, holdings=1.0)
        investor.update_status(0.0, 2.0, 0.0, ALWAYS, ScriptedRandom([NO_NOISE, FIRE, SKIP, SKIP]))
        assert investor.sentiment == pytest.approx(2.0 * FOMO_SENSITIVITY)
        assert abs(investor.sentiment) < 1.0  # a 2% move must not saturate sentiment

    def test_fomo_skipped_leaves_sentiment_alone(self):
        investor = Investor(sentiment=0.25, holdings=1.0)
        investor.update_status(0.0, 50.0, 0.0, ALWAYS, ScriptedRandom([NO_NOISE, SKIP, SKIP, SKIP]))
        assert investor.sentiment == pytest.approx(0.25)

    def test_panic_is_a_damped_contrarian_flip(self):
        investor = Investor(sentiment=0.9, holdings=1.0)
        investor.update_status(0.0, 0.0, 0.0, ALWAYS, ScriptedRandom([NO_NOISE, SKIP, FIRE]))
        assert investor.sentiment == pytest.approx(0.9 * PANIC_REVERSAL)
        assert investor.state == SELL

    def test_narrative_pull_uses_its_own_draw_and_the_live_narrative(self):
        investor = Investor(sentiment=0.0, holdings=1.0)
        narrative = -0.5
        rng = ScriptedRandom([NO_NOISE, SKIP, SKIP, FIRE])
        investor.update_status(narrative, 0.0, 0.0, ALWAYS, rng)
        assert investor.sentiment == pytest.approx(FOLLOW_NARRATIVE_WEIGHT * narrative)

    def test_panic_takes_precedence_over_narrative(self):
        investor = Investor(sentiment=0.9, holdings=1.0)
        rng = ScriptedRandom([NO_NOISE, SKIP, FIRE, FIRE])
        investor.update_status(-1.0, 0.0, 0.0, ALWAYS, rng)
        assert investor.sentiment == pytest.approx(0.9 * PANIC_REVERSAL)
        assert rng._values == [FIRE]  # the narrative draw was never consumed

    def test_noise_is_symmetric_and_bounded(self):
        low = Investor(sentiment=0.0, holdings=1.0)
        low.update_status(0.0, 0.0, 0.0, ALWAYS, ScriptedRandom([0.0, SKIP, SKIP, SKIP]))
        high = Investor(sentiment=0.0, holdings=1.0)
        high.update_status(0.0, 0.0, 0.0, ALWAYS, ScriptedRandom([1.0, SKIP, SKIP, SKIP]))
        assert low.sentiment == pytest.approx(-SENTIMENT_NOISE)
        assert high.sentiment == pytest.approx(SENTIMENT_NOISE)

    def test_value_pull_is_off_by_default(self):
        investor = Investor(sentiment=0.0, holdings=1.0)
        rng = ScriptedRandom([NO_NOISE, SKIP, SKIP, SKIP])
        investor.update_status(0.0, 0.0, math.log(4.0), ALWAYS, rng)
        assert investor.sentiment == 0.0

    def test_value_pull_leans_against_mispricing(self):
        config = SimulationConfig(fomo_threshold=1.0, panic_prob=1.0, value_pull=0.5)
        investor = Investor(sentiment=0.0, holdings=1.0)
        rng = ScriptedRandom([NO_NOISE, SKIP, SKIP, SKIP])
        investor.update_status(0.0, 0.0, math.log(2.0), config, rng)
        assert investor.sentiment == pytest.approx(-0.5 * math.log(2.0))

    def test_sentiment_is_clamped(self):
        investor = Investor(sentiment=0.95, holdings=1.0)
        investor.update_status(0.0, 0.0, 0.0, ALWAYS, ScriptedRandom([1.0, SKIP, SKIP, SKIP]))
        assert investor.sentiment == 1.0
        investor = Investor(sentiment=-0.95, holdings=1.0)
        investor.update_status(0.0, 0.0, 0.0, ALWAYS, ScriptedRandom([0.0, SKIP, SKIP, SKIP]))
        assert investor.sentiment == -1.0

    def test_update_status_always_stays_in_bounds(self):
        rng = random.Random(7)
        config = SimulationConfig()
        for _ in range(500):
            investor = Investor(sentiment=rng.uniform(-1, 1), holdings=10.0)
            investor.update_status(
                rng.uniform(-1, 1), rng.uniform(-50, 50), rng.uniform(-3, 3), config, rng
            )
            assert -1.0 <= investor.sentiment <= 1.0
            assert investor.state in (BUY, HOLD, SELL)


class TestCountStates:
    def test_counts(self):
        investors = [
            Investor(sentiment=0.9, holdings=1.0),
            Investor(sentiment=0.5, holdings=1.0),
            Investor(sentiment=0.0, holdings=1.0),
            Investor(sentiment=-0.9, holdings=1.0),
        ]
        assert count_states(investors) == (2, 1, 1)


class TestInitialState:
    def test_allocation_uses_all_shares_in_whole_units(self):
        for seed in range(5):
            simulation = Simulation(rng=random.Random(seed))
            holdings = [investor.holdings for investor in simulation.investors]
            assert sum(holdings) == simulation.config.max_shares
            assert all(h >= 0 and h == int(h) for h in holdings)
            assert len(holdings) == simulation.config.num_investors

    def test_allocation_does_not_depend_on_creation_order(self):
        count = SimulationConfig().num_investors
        by_index = [0.0] * count
        seeds = 200
        for seed in range(seeds):
            simulation = Simulation(rng=random.Random(seed))
            for index, investor in enumerate(simulation.investors):
                by_index[index] += investor.holdings / seeds
        expected = SimulationConfig().max_shares / count
        assert 0.9 * expected < min(by_index) and max(by_index) < 1.1 * expected

    def test_allocation_is_bounded_in_inequality(self):
        simulation = Simulation(rng=random.Random(42))
        holdings = sorted(investor.holdings for investor in simulation.investors)
        # Weights are uniform in [1, 10], so the top investor holds at most ~10x the bottom.
        assert holdings[-1] <= 10.5 * max(holdings[0], 1.0)

    def test_initial_sentiments_in_bounds_and_skewed_by_narrative(self):
        bull = Simulation(SimulationConfig(sentiment_skew=1.0), rng=random.Random(3))
        bear = Simulation(SimulationConfig(sentiment_skew=-1.0), rng=random.Random(3))
        for simulation in (bull, bear):
            assert all(-1.0 <= inv.sentiment <= 1.0 for inv in simulation.investors)
        assert bull.average_sentiment() > bear.average_sentiment()

    def test_series_start_aligned(self):
        simulation = Simulation(rng=random.Random(1))
        assert len(simulation.price_history) == 1
        assert len(simulation.volume_history) == 1
        assert len(simulation.buyers) == len(simulation.holders) == len(simulation.sellers) == 1
        assert simulation.steps_completed == 0


class TestSimulation:
    def test_shares_conserved_and_never_negative_over_run(self):
        simulation = Simulation(rng=random.Random(42))
        simulation.run(200)
        assert simulation.total_holdings() == pytest.approx(simulation.config.max_shares, rel=1e-9)
        assert min(inv.holdings for inv in simulation.investors) >= 0

    def test_series_stay_aligned_and_counts_sum_to_population(self):
        simulation = Simulation(rng=random.Random(11))
        simulation.run(50)
        length = len(simulation.price_history)
        assert length == 51
        assert len(simulation.volume_history) == length
        for series in (simulation.buyers, simulation.holders, simulation.sellers):
            assert len(series) == length
        total = simulation.config.num_investors
        for b, h, s in zip(simulation.buyers, simulation.holders, simulation.sellers, strict=True):
            assert b + h + s == total

    def test_same_seed_same_series(self):
        first = Simulation(rng=random.Random(123))
        second = Simulation(rng=random.Random(123))
        assert first.run(300) == second.run(300)
        assert first.volume_history == second.volume_history
        assert first.buyers == second.buyers

    def test_different_seeds_diverge(self):
        first = Simulation(rng=random.Random(1))
        second = Simulation(rng=random.Random(2))
        assert first.run(100) != second.run(100)

    def test_percent_trajectory_independent_of_start_price(self):
        reference = Simulation(SimulationConfig(start_price=100.0), rng=random.Random(5)).run(200)
        for start in (0.5, 10.0, 1000.0):
            history = Simulation(SimulationConfig(start_price=start), rng=random.Random(5)).run(200)
            assert [p / start for p in history] == pytest.approx([p / 100.0 for p in reference])

    def test_zero_trades_means_no_shares_move(self):
        simulation = Simulation(SimulationConfig(num_trades=0), rng=random.Random(9))
        before = [inv.holdings for inv in simulation.investors]
        simulation.run(50)
        assert simulation.volume_history == [0.0] * 51
        assert [inv.holdings for inv in simulation.investors] == before

    def test_one_trade_means_one_real_counterparty(self):
        simulation = Simulation(SimulationConfig(num_trades=1), rng=random.Random(9))
        simulation.run(20)
        assert max(simulation.volume_history) > 0.0

    def test_shocks_are_independent_of_panic(self, monkeypatch):
        # Silence the imbalance-driven impact and return noise so only shocks can move the price.
        monkeypatch.setattr(sim_module, "PRICE_IMPACT", 0.0)
        monkeypatch.setattr(sim_module, "RETURN_NOISE", 0.0)
        quiet = Simulation(SimulationConfig(panic_prob=0.0, shock_prob=0.0), rng=random.Random(2))
        assert quiet.run(50) == [100.0] * 51
        shocked = Simulation(SimulationConfig(panic_prob=0.0, shock_prob=1.0), rng=random.Random(2))
        history = shocked.run(50)
        ratios = [history[i] / history[i - 1] for i in range(1, len(history))]
        assert all(0.8 <= r <= 1.25 and r != 1.0 for r in ratios)

    def test_narrative_fades_with_half_life(self):
        simulation = Simulation(
            SimulationConfig(sentiment_skew=-1.0, narrative_half_life=10.0), rng=random.Random(0)
        )
        assert simulation.narrative_sentiment() == pytest.approx(-1.0)
        simulation.run(10)
        assert simulation.narrative_sentiment() == pytest.approx(-0.5)
        simulation.run(10)
        assert simulation.narrative_sentiment() == pytest.approx(-0.25)

    def test_narrative_never_fades_with_infinite_half_life(self):
        simulation = Simulation(
            SimulationConfig(sentiment_skew=0.5, narrative_half_life=math.inf), rng=random.Random(0)
        )
        simulation.run(100)
        assert simulation.narrative_sentiment() == 0.5

    def test_step_is_safe_after_price_reaches_zero(self):
        simulation = Simulation(rng=random.Random(4))
        simulation.run(5)
        simulation.current_price = 0.0
        simulation.price_history.append(0.0)
        simulation.step()  # must not raise on the zero previous price
        assert simulation.current_price == 0.0
        length = len(simulation.price_history)
        simulation.run(10)
        assert len(simulation.price_history) == length + 1  # stopped early

    def test_prices_stay_positive_under_defaults(self):
        for seed in range(10):
            history = Simulation(rng=random.Random(seed)).run(500)
            assert min(history) > 0.0
            assert len(history) == 501

    def test_narrative_moves_prices_directionally(self):
        def final_price(skew: float, seed: int) -> float:
            config = SimulationConfig(sentiment_skew=skew)
            return Simulation(config, rng=random.Random(seed)).run(150)[-1]

        seeds = range(8)
        bull = [final_price(SENTIMENT_SKEWS["VERY_BULL"], seed) for seed in seeds]
        bear = [final_price(SENTIMENT_SKEWS["VERY_BEAR"], seed) for seed in seeds]
        start = SimulationConfig().start_price
        assert all(b > r for b, r in zip(bull, bear, strict=True))
        assert mean(bear) < start < mean(bull)

    def test_event_produces_a_large_move(self):
        start = SimulationConfig().start_price
        peaks, troughs = [], []
        for seed in range(8):
            bull = SimulationConfig(sentiment_skew=SENTIMENT_SKEWS["VERY_BULL"])
            bear = SimulationConfig(sentiment_skew=SENTIMENT_SKEWS["VERY_BEAR"])
            peaks.append(max(Simulation(bull, rng=random.Random(seed)).run(300)))
            troughs.append(min(Simulation(bear, rng=random.Random(seed)).run(300)))
        assert mean(peaks) > 1.5 * start
        assert mean(troughs) < 0.5 * start

    def test_value_pull_brings_prices_back_toward_start(self):
        start = SimulationConfig().start_price

        def distance(value_pull: float) -> float:
            finals = []
            for seed in range(8):
                config = SimulationConfig(
                    sentiment_skew=SENTIMENT_SKEWS["VERY_BULL"], value_pull=value_pull
                )
                finals.append(Simulation(config, rng=random.Random(seed)).run(500)[-1])
            return mean(abs(final - start) for final in finals)

        assert distance(0.2) < distance(0.0)

    def test_outcomes_vary_across_seeds(self):
        finals = {round(Simulation(rng=random.Random(seed)).run(200)[-1], 2) for seed in range(8)}
        assert len(finals) == 8

    def test_golden_trajectory_for_seed_42(self):
        # Pins the exact default-model semantics: any change to a rule, a constant,
        # or the order of random draws moves these numbers. Regenerate deliberately.
        simulation = Simulation(rng=random.Random(42))
        history = simulation.run(500)
        assert (simulation.buyers[0], simulation.holders[0], simulation.sellers[0]) == (22, 25, 53)
        assert history[1] == pytest.approx(99.81708326677688, rel=1e-9)
        assert history[2] == pytest.approx(100.46455237302924, rel=1e-9)
        assert simulation.volume_history[1] == pytest.approx(44110.42256687234, rel=1e-9)
        assert history[50] == pytest.approx(40.71983802929725, rel=1e-9)
        assert history[500] == pytest.approx(21.139655257184522, rel=1e-9)
        end = (simulation.buyers[500], simulation.holders[500], simulation.sellers[500])
        assert end == (6, 87, 7)


class TestEventStep:
    def test_default_event_is_applied_before_the_first_recorded_state(self):
        bull = Simulation(SimulationConfig(sentiment_skew=1.0), rng=random.Random(2))
        neutral = Simulation(SimulationConfig(sentiment_skew=0.0), rng=random.Random(2))
        assert bull.average_sentiment() > neutral.average_sentiment() + 0.2
        assert bull.buyers[0] > neutral.buyers[0]

    def test_market_is_neutral_until_the_event(self):
        bull = SimulationConfig(sentiment_skew=SENTIMENT_SKEWS["VERY_BULL"], event_step=150)
        neutral = SimulationConfig(sentiment_skew=SENTIMENT_SKEWS["NEUTRAL"])
        with_event = Simulation(bull, rng=random.Random(3))
        without = Simulation(neutral, rng=random.Random(3))
        with_event.run(300)
        without.run(300)
        # Nothing about the event leaks backward: the two runs share every random
        # draw, so they are identical up to and including the price at the event step.
        assert with_event.price_history[:151] == without.price_history[:151]
        assert with_event.volume_history[:151] == without.volume_history[:151]
        assert with_event.buyers[:151] == without.buyers[:151]
        assert with_event.price_history[151:] != without.price_history[151:]

    def test_narrative_is_silent_before_the_event_then_fades(self):
        config = SimulationConfig(sentiment_skew=-1.0, event_step=20, narrative_half_life=10.0)
        simulation = Simulation(config, rng=random.Random(0))
        assert simulation.narrative_sentiment() == 0.0
        simulation.run(19)
        assert simulation.narrative_sentiment() == 0.0
        simulation.run(1)
        assert simulation.narrative_sentiment() == pytest.approx(-1.0)
        simulation.run(10)
        assert simulation.narrative_sentiment() == pytest.approx(-0.5)

    def test_event_jolts_sentiment_when_it_hits(self):
        bull = SimulationConfig(sentiment_skew=SENTIMENT_SKEWS["VERY_BULL"], event_step=5)
        neutral = SimulationConfig(sentiment_skew=SENTIMENT_SKEWS["NEUTRAL"])
        with_event = Simulation(bull, rng=random.Random(8))
        twin = Simulation(neutral, rng=random.Random(8))
        with_event.run(5)
        twin.run(5)
        assert with_event.average_sentiment() == pytest.approx(twin.average_sentiment())
        with_event.run(1)
        twin.run(1)
        assert with_event.average_sentiment() > twin.average_sentiment() + 0.1
        assert with_event.buyers[-1] > twin.buyers[-1]

    def test_event_after_the_run_ends_is_a_neutral_run(self):
        bear = SimulationConfig(sentiment_skew=SENTIMENT_SKEWS["VERY_BEAR"], event_step=1_000)
        late = Simulation(bear, rng=random.Random(1)).run(100)
        neutral = Simulation(SimulationConfig(sentiment_skew=0.0), rng=random.Random(1)).run(100)
        assert late == neutral
