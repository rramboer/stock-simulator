"""Unit tests for the simulation model."""

import random
from statistics import mean

import pytest

from stock_simulator.simulation import (
    BUY,
    HOLD,
    SELL,
    SENTIMENT_SKEWS,
    Investor,
    Simulation,
    SimulationConfig,
    count_states,
    percent_change,
    state_for_sentiment,
)


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


class TestInvestor:
    def test_initial_state_derived_from_sentiment(self):
        assert Investor(sentiment=0.9, holdings=10.0).state == BUY
        assert Investor(sentiment=-0.9, holdings=10.0).state == SELL
        assert Investor(sentiment=0.1, holdings=10.0).state == HOLD

    def test_trade_conserves_shares(self):
        buyer = Investor(sentiment=0.9, holdings=100.0)
        seller = Investor(sentiment=-0.5, holdings=200.0)
        volume = buyer.trade(seller)
        assert volume == pytest.approx(40.0)  # mean sentiment 0.2 * 200 held
        assert buyer.holdings + seller.holdings == pytest.approx(300.0)
        assert seller.holdings >= 0

    def test_trade_is_symmetric_in_direction(self):
        buyer = Investor(sentiment=0.9, holdings=100.0)
        seller = Investor(sentiment=-0.5, holdings=200.0)
        volume = seller.trade(buyer)  # seller initiating moves its own shares
        assert volume == pytest.approx(40.0)
        assert buyer.holdings == pytest.approx(140.0)
        assert seller.holdings == pytest.approx(160.0)

    def test_trade_negative_mean_sentiment_moves_nothing(self):
        buyer = Investor(sentiment=0.4, holdings=100.0)
        seller = Investor(sentiment=-0.9, holdings=200.0)
        assert buyer.trade(seller) == 0.0
        assert buyer.holdings == 100.0
        assert seller.holdings == 200.0

    def test_trade_requires_buyer_and_seller(self):
        one = Investor(sentiment=0.9, holdings=100.0)
        other = Investor(sentiment=0.8, holdings=200.0)
        assert one.trade(other) == 0.0

    def test_self_trade_is_noop(self):
        investor = Investor(sentiment=0.9, holdings=100.0)
        assert investor.trade(investor) == 0.0
        assert investor.holdings == 100.0

    def test_update_status_clamps_sentiment_high(self):
        config = SimulationConfig(fomo_threshold=1.0, panic_prob=0.0)
        investor = Investor(sentiment=0.9, holdings=10.0)
        # Seed chosen so the shared draw skips the follow-the-market branch.
        investor.update_status(0.0, 500.0, config, random.Random(0))
        assert investor.sentiment == 1.0
        assert investor.state == BUY

    def test_update_status_clamps_sentiment_low(self):
        config = SimulationConfig(fomo_threshold=1.0, panic_prob=0.0)
        investor = Investor(sentiment=-0.9, holdings=10.0)
        investor.update_status(0.0, -500.0, config, random.Random(0))
        assert investor.sentiment == -1.0
        assert investor.state == SELL

    def test_update_status_always_stays_in_bounds(self):
        rng = random.Random(7)
        config = SimulationConfig()
        for _ in range(500):
            investor = Investor(sentiment=rng.uniform(-1, 1), holdings=10.0)
            investor.update_status(rng.uniform(-100, 100), rng.uniform(-50, 50), config, rng)
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


class TestSimulation:
    def test_initial_allocation_uses_all_shares(self):
        for seed in range(5):
            simulation = Simulation(rng=random.Random(seed))
            assert simulation.total_holdings() == simulation.config.max_shares
            assert len(simulation.investors) == simulation.config.num_investors

    def test_initial_sentiments_in_bounds(self):
        simulation = Simulation(rng=random.Random(3))
        assert all(-1.0 <= inv.sentiment <= 1.0 for inv in simulation.investors)

    def test_shares_conserved_over_run(self):
        simulation = Simulation(rng=random.Random(42))
        simulation.run(200)
        assert simulation.total_holdings() == pytest.approx(simulation.config.max_shares, rel=1e-9)
        assert min(inv.holdings for inv in simulation.investors) >= 0

    def test_same_seed_same_price_series(self):
        first = Simulation(rng=random.Random(123))
        second = Simulation(rng=random.Random(123))
        assert first.run(300) == second.run(300)
        assert first.buyers == second.buyers
        assert first.holders == second.holders
        assert first.sellers == second.sellers

    def test_different_seeds_diverge(self):
        first = Simulation(rng=random.Random(1))
        second = Simulation(rng=random.Random(2))
        assert first.run(100) != second.run(100)

    def test_price_floor_and_early_stop(self):
        # VERY_BEAR runs usually crash; whenever the price reaches zero the
        # run must stop, so zero can only ever be the final recorded price.
        for seed in range(10):
            simulation = Simulation(rng=random.Random(seed))
            history = simulation.run(500)
            assert min(history) >= 0.0
            assert 0.0 not in history[:-1]

    def test_step_tracks_state_counts(self):
        simulation = Simulation(rng=random.Random(11))
        simulation.run(50)
        steps = len(simulation.price_history) - 1
        assert len(simulation.buyers) == steps
        assert len(simulation.holders) == steps
        assert len(simulation.sellers) == steps
        total = simulation.config.num_investors
        for step in range(steps):
            assert (
                simulation.buyers[step] + simulation.holders[step] + simulation.sellers[step]
                == total
            )

    def test_sentiment_skew_moves_prices_directionally(self):
        # Statistical check: very bullish markets should end meaningfully
        # higher than very bearish ones on average across seeds.
        def final_price(skew: float, seed: int) -> float:
            config = SimulationConfig(sentiment_skew=skew)
            simulation = Simulation(config, rng=random.Random(seed))
            return simulation.run(150)[-1]

        seeds = range(15)
        bull_mean = mean(final_price(SENTIMENT_SKEWS["VERY_BULL"], seed) for seed in seeds)
        bear_mean = mean(final_price(SENTIMENT_SKEWS["VERY_BEAR"], seed) for seed in seeds)
        start = SimulationConfig().start_price
        assert bull_mean > bear_mean
        assert bear_mean < start  # bear markets lose value on average
