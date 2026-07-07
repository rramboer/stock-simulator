"""Agent-based Monte Carlo model of a stock rally or crash.

A population of investors, each holding shares and carrying a sentiment in
``[-1, 1]``, trade with one another every time step. Sentiment maps to a
BUY/HOLD/SELL state, and the stock price moves with the traded volume and the
aggregate market sentiment. Fear-of-missing-out (FOMO) couples each investor's
sentiment to the latest price move, panic randomly reverses sentiment, and
rare exogenous market shocks jolt the price directly.

The update rules and constants below are preserved exactly from the original
model so that its published dynamics are unchanged.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

# Investor states.
BUY = 1
SELL = -1
HOLD = 0

# Sentiment presets: the skew mixed into every investor's initial sentiment
# (and targeted by the follow-the-market behavior).
SENTIMENT_SKEWS: dict[str, float] = {
    "VERY_BEAR": -1.0,
    "BEARISH": -0.5,
    "NEUTRAL": 0.0,
    "BULLISH": 0.5,
    "VERY_BULL": 1.0,
}

# Defaults matching the original hardcoded configuration.
DEFAULT_SENTIMENT = "VERY_BEAR"
DEFAULT_STEPS = 500

# --- Model constants (values preserved from the original model) -------------

# Sentiment above +STATE_THRESHOLD means BUY, below -STATE_THRESHOLD means SELL.
STATE_THRESHOLD = 0.333

# Multiplier applied to sentiment when an investor panics: the sign flips and
# the magnitude is damped.
PANIC_REVERSAL = -0.66

# Probability that an investor "follows the market" toward the sentiment skew.
# NOTE: this branch shares a single random draw with the FOMO check and is
# chained behind the panic check, so with the default FOMO threshold of 0.5 it
# can only fire on steps where the FOMO adjustment also fired. Preserved
# as-is because changing it would alter the simulation dynamics.
FOLLOW_MARKET_PROB = 0.2

# Weight of the sentiment skew when an investor follows the market.
FOLLOW_MARKET_WEIGHT = 0.8

# Weight of the sentiment skew mixed into each investor's initial sentiment.
INITIAL_SKEW_WEIGHT = 0.3

# Fraction of the remaining share pool that bounds each investor's initial
# holdings (drawn between 1% and 10% of what is left).
HOLDINGS_NORM_FRACTION = 0.1

# The per-step price impact is amplified by a multiplier drawn uniformly from
# [PRICE_NOISE_MIN, PRICE_NOISE_MAX). The draw's mean of +2 amplifies the
# impact in the direction of aggregate sentiment (whether bullish or bearish);
# negative draws occasionally flip it.
PRICE_NOISE_MIN = -3.0
PRICE_NOISE_MAX = 7.0

# An unforeseen market event rescales the price by a factor drawn uniformly
# from [MARKET_SHOCK_MIN, MARKET_SHOCK_MAX); it occurs each step with
# probability panic_prob / MARKET_EVENT_DIVISOR.
MARKET_SHOCK_MIN = 0.75
MARKET_SHOCK_MAX = 1.25
MARKET_EVENT_DIVISOR = 8


def percent_change(previous: float, current: float) -> float:
    """Return the percent change from ``previous`` to ``current`` (5.0 means +5%).

    ``previous`` must be nonzero; the simulation guarantees this because a run
    stops as soon as the price reaches zero.
    """
    if previous == 0:
        raise ValueError("previous price must be nonzero")
    return ((current / previous) * 100.0) - 100.0


def state_for_sentiment(sentiment: float) -> int:
    """Map a sentiment value to the BUY, SELL, or HOLD state."""
    if sentiment < -STATE_THRESHOLD:
        return SELL
    if sentiment > STATE_THRESHOLD:
        return BUY
    return HOLD


def count_states(investors: Sequence[Investor]) -> tuple[int, int, int]:
    """Return the number of investors currently buying, holding, and selling."""
    buyers = sum(1 for investor in investors if investor.state == BUY)
    sellers = sum(1 for investor in investors if investor.state == SELL)
    holders = len(investors) - buyers - sellers
    return buyers, holders, sellers


@dataclass(frozen=True)
class SimulationConfig:
    """Tunable parameters of the simulation (defaults match the original model)."""

    num_investors: int = 100
    """Number of investors in the market."""

    max_shares: int = 100_000
    """Total number of stock shares outstanding."""

    start_price: float = 100.0
    """Starting stock price."""

    num_trades: int = 5
    """Trades per investor per time step (the first pairing is a self-trade no-op)."""

    fomo_threshold: float = 0.5
    """Susceptibility to FOMO in (0, 1]: 0 means none, 1 means maximum."""

    panic_prob: float = 0.2
    """Probability per step that an investor panic-trades (sentiment reverses)."""

    sentiment_skew: float = SENTIMENT_SKEWS[DEFAULT_SENTIMENT]
    """Market-wide sentiment bias in [-1, 1]; see SENTIMENT_SKEWS for presets."""


@dataclass
class Investor:
    """A market participant with a sentiment in ``[-1, 1]`` and share holdings.

    Holdings start as whole shares but become fractional through trading; this
    is intentional, and the total across all investors is conserved.
    """

    sentiment: float
    holdings: float
    state: int = field(init=False)

    def __post_init__(self) -> None:
        self.state = state_for_sentiment(self.sentiment)

    def update_status(
        self,
        mean_sentiment: float,
        last_price_change: float,
        config: SimulationConfig,
        rng: random.Random,
    ) -> None:
        """Apply the FOMO, panic, and follow-the-market rules, then refresh the state.

        ``mean_sentiment`` is the market's average sentiment for this step and
        ``last_price_change`` is the most recent percent price move (0.0 on the
        first step).
        """
        rand_val = rng.random()
        if rand_val < config.fomo_threshold:
            # FOMO: chase the latest price move. The percent change is in
            # percent units (a 5% move adds 5.0) while sentiment lives in
            # [-1, 1], so any move larger than ~1% saturates at the clamp
            # below -- FOMO acts as a near-binary flip. Preserved as-is.
            self.sentiment += last_price_change
        if rng.random() < config.panic_prob:
            # Panic: reverse and damp sentiment, nudged by the market mean
            # scaled down again by the population size (as in the original).
            self.sentiment = self.sentiment * PANIC_REVERSAL + (
                mean_sentiment / config.num_investors
            )
        elif rand_val < FOLLOW_MARKET_PROB:
            # Follow the market: snap toward the configured sentiment skew.
            self.sentiment = config.sentiment_skew * FOLLOW_MARKET_WEIGHT
        self.state = state_for_sentiment(self.sentiment)
        # Sentiment boundary checking.
        self.sentiment = max(-1.0, min(1.0, self.sentiment))

    def trade(self, other: Investor) -> float:
        """Exchange shares with ``other`` and return the number of shares moved.

        Shares move only between a buyer and a seller, in proportion to the
        average of the two sentiments times the seller's holdings, so holdings
        never go negative and the market total is conserved.
        """
        exchange_amount = 0.0
        if self.state == BUY and other.state == SELL:
            exchange_amount = ((self.sentiment + other.sentiment) / 2) * other.holdings
            exchange_amount = max(0.0, exchange_amount)
            self.holdings += exchange_amount
            other.holdings -= exchange_amount
        elif self.state == SELL and other.state == BUY:
            exchange_amount = ((self.sentiment + other.sentiment) / 2) * self.holdings
            exchange_amount = max(0.0, exchange_amount)
            self.holdings -= exchange_amount
            other.holdings += exchange_amount
        return exchange_amount


class Simulation:
    """A single simulated market: a population of investors and a price series."""

    def __init__(
        self,
        config: SimulationConfig | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config if config is not None else SimulationConfig()
        self.rng = rng if rng is not None else random.Random()
        self.current_price: float = self.config.start_price
        self.price_history: list[float] = [self.config.start_price]
        self.buyers: list[int] = []
        self.holders: list[int] = []
        self.sellers: list[int] = []
        self.investors: list[Investor] = self._create_investors()

    def _create_investors(self) -> list[Investor]:
        """Populate the market, allocating exactly ``max_shares`` across investors."""
        config, rng = self.config, self.rng
        investors: list[Investor] = []
        outstanding_shares = config.max_shares
        for _ in range(config.num_investors):
            # Uniform sentiment in [-1, 1), shifted by the skew and clamped.
            sentiment = (rng.random() * 2.0 - 1.0) + (config.sentiment_skew * INITIAL_SKEW_WEIGHT)
            sentiment = max(-1.0, min(1.0, sentiment))

            # Holdings drawn from 1%..10% of the remaining share pool.
            norm = int(outstanding_shares * HOLDINGS_NORM_FRACTION)
            holdings = rng.randint(int(norm * HOLDINGS_NORM_FRACTION), norm)
            if outstanding_shares - holdings < 0:  # holdings boundary checking
                holdings = 0
            investors.append(Investor(sentiment=sentiment, holdings=float(holdings)))
            outstanding_shares -= holdings

        # Hand out any stray shares round-robin starting at investor 0 (bulk
        # equivalent of the original one-share-at-a-time loop).
        base_extra, remainder = divmod(outstanding_shares, config.num_investors)
        for index, investor in enumerate(investors):
            investor.holdings += base_extra + (1 if index < remainder else 0)
        return investors

    def market_sentiment(self) -> float:
        """Return the sum of all investor sentiments."""
        return sum(investor.sentiment for investor in self.investors)

    def average_sentiment(self) -> float:
        """Return the mean investor sentiment."""
        return self.market_sentiment() / self.config.num_investors

    def total_holdings(self) -> float:
        """Return the total shares held across all investors."""
        return sum(investor.holdings for investor in self.investors)

    def step(self) -> None:
        """Advance one time step: trade, update sentiments, then move the price."""
        config, rng = self.config, self.rng

        # A randomly ordered copy of the investors; freezing the aggregate
        # sentiment first means all investors update pseudo-parallelly.
        shuffled = rng.sample(self.investors, config.num_investors)
        total_sentiment = self.market_sentiment()

        # Each investor trades with the next num_trades investors in the
        # shuffled order. The j == 0 pairing is the investor itself, which is
        # always a no-op (nobody is both buyer and seller), leaving
        # num_trades - 1 real counterparties. Preserved as-is.
        volume = 0.0
        for i in range(config.num_investors):
            for j in range(config.num_trades):
                volume += shuffled[i].trade(shuffled[(j + i) % config.num_investors])

        last_price_change = (
            percent_change(self.price_history[-2], self.price_history[-1])
            if len(self.price_history) > 1
            else 0.0
        )
        mean_sentiment = total_sentiment / config.num_investors
        for investor in shuffled:
            investor.update_status(mean_sentiment, last_price_change, config, rng)

        # Price impact: traded volume times aggregate sentiment, scaled by the
        # total float and amplified by the asymmetric noise multiplier.
        price_change = volume * total_sentiment / config.max_shares
        price_change *= rng.random() * (PRICE_NOISE_MAX - PRICE_NOISE_MIN) + PRICE_NOISE_MIN
        self.current_price += price_change

        if rng.random() < config.panic_prob / MARKET_EVENT_DIVISOR:
            # Unforeseen market event: random shock in [0.75, 1.25).
            self.current_price *= (
                rng.random() * (MARKET_SHOCK_MAX - MARKET_SHOCK_MIN) + MARKET_SHOCK_MIN
            )

        # Price floor at zero; a run ends once the price reaches it.
        self.current_price = max(self.current_price, 0.0)
        self.price_history.append(self.current_price)

        num_buyers, num_holders, num_sellers = count_states(self.investors)
        self.buyers.append(num_buyers)
        self.holders.append(num_holders)
        self.sellers.append(num_sellers)

    def run(self, max_steps: int = DEFAULT_STEPS) -> list[float]:
        """Run up to ``max_steps`` steps, stopping early if the price hits zero.

        Returns the price history (one entry per step plus the starting price).
        """
        for _ in range(max_steps):
            self.step()
            if self.current_price <= 0:
                break
        return self.price_history
