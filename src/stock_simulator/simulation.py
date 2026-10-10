"""Agent-based Monte Carlo model of a stock rally or crash driven by a social event.

A population of investors, each holding shares and carrying a sentiment in
``[-1, 1]``, trade with one another every time step. Sentiment maps to a
BUY/HOLD/SELL state, and the price return each step is proportional to the
buy/sell imbalance (buyers minus sellers, as a fraction of the population)
plus a little zero-mean noise.

The run starts just after a social event -- a short squeeze, a doom
narrative -- has pushed the crowd's sentiment one way. That *narrative*
sentiment keeps pulling investors toward it, but fades with a configurable
half-life. Fear-of-missing-out (FOMO) pulls each investor's sentiment toward
the latest price move, so a move recruits more of the crowd and feeds on
itself; panic flips an investor to the opposite camp; idiosyncratic noise
keeps the crowd heterogeneous; and rare exogenous shocks jolt the price
directly. An optional value pull (off by default, because a repriced stock
need not return to where it started) leans everyone against the price's
departure from its start, which makes bubbles pop and crashes get bought. Trading
moves shares from sellers to buyers; the traded volume is recorded but does
not itself move the price.

All price dynamics are multiplicative (percent-based), so a run's percent
trajectory does not depend on the starting price.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

# Investor states.
BUY = 1
SELL = -1
HOLD = 0

# Sentiment presets: the narrative sentiment of the social event.
SENTIMENT_SKEWS: dict[str, float] = {
    "VERY_BEAR": -1.0,
    "BEARISH": -0.5,
    "NEUTRAL": 0.0,
    "BULLISH": 0.5,
    "VERY_BULL": 1.0,
}

DEFAULT_SENTIMENT = "VERY_BEAR"
DEFAULT_STEPS = 500

# --- Model constants ---------------------------------------------------------

# Sentiment above +STATE_THRESHOLD means BUY, below -STATE_THRESHOLD means SELL.
STATE_THRESHOLD = 0.333

# Weight of the narrative sentiment mixed into each investor's initial sentiment.
INITIAL_SKEW_WEIGHT = 0.3

# Each investor's sentiment drifts by an idiosyncratic amount drawn uniformly
# from [-SENTIMENT_NOISE, SENTIMENT_NOISE) every step (private news). This keeps
# the population heterogeneous; without it the herding and panic rules pull
# everyone to HOLD and the market freezes.
SENTIMENT_NOISE = 0.3

# Sentiment change per percentage point of the last price move when an
# investor chases it (FOMO).
FOMO_SENSITIVITY = 0.1

# Multiplier applied to sentiment when an investor panics: a contrarian flip
# to the opposite camp with the magnitude damped.
PANIC_REVERSAL = -0.66

# Probability that a non-panicking investor follows the narrative, and how far
# toward the current narrative sentiment they move (1.0 would snap onto it).
FOLLOW_NARRATIVE_PROB = 0.2
FOLLOW_NARRATIVE_WEIGHT = 0.8

# Steps for the narrative's pull to halve. ``math.inf`` keeps it constant.
DEFAULT_NARRATIVE_HALF_LIFE = 50.0

# Fractional price change per unit of excess demand, where excess demand is
# (buyers - sellers) / investors in [-1, 1]. At 0.04 a market that is entirely
# buyers rises 4% per step before noise.
PRICE_IMPACT = 0.04

# Standard deviation of the zero-mean Gaussian return noise added each step.
RETURN_NOISE = 0.01

# An exogenous market shock rescales the price by a factor between
# 1 / MARKET_SHOCK_MAX_FACTOR and MARKET_SHOCK_MAX_FACTOR, uniform in log space
# so shocks carry no drift; it occurs each step with probability ``shock_prob``.
MARKET_SHOCK_MAX_FACTOR = 1.25
DEFAULT_SHOCK_PROB = 0.025

# Initial holdings are proportional to a per-investor weight drawn uniformly
# from this range, so the richest investor holds at most ten times the poorest
# and the allocation does not depend on creation order.
HOLDINGS_WEIGHT_MIN = 1.0
HOLDINGS_WEIGHT_MAX = 10.0


def percent_change(previous: float, current: float) -> float:
    """Return the percent change from ``previous`` to ``current`` (5.0 means +5%).

    ``previous`` must be nonzero.
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


def _clamp_sentiment(sentiment: float) -> float:
    return max(-1.0, min(1.0, sentiment))


@dataclass(frozen=True)
class SimulationConfig:
    """Tunable parameters of the simulation. Invalid values raise ``ValueError``."""

    num_investors: int = 100
    """Number of investors in the market (at least 1)."""

    max_shares: int = 100_000
    """Total number of stock shares outstanding (at least 1)."""

    start_price: float = 100.0
    """Starting price (positive); also the reference for the optional value pull."""

    num_trades: int = 5
    """Distinct counterparties each investor trades with per step, 0..num_investors-1."""

    fomo_threshold: float = 0.5
    """Probability per step that an investor chases the last price move, in [0, 1]."""

    panic_prob: float = 0.2
    """Probability per step that an investor panics (sentiment flips), in [0, 1]."""

    shock_prob: float = DEFAULT_SHOCK_PROB
    """Probability per step of an exogenous market shock, in [0, 1]."""

    sentiment_skew: float = SENTIMENT_SKEWS[DEFAULT_SENTIMENT]
    """Narrative sentiment of the social event in [-1, 1]; see SENTIMENT_SKEWS."""

    narrative_half_life: float = DEFAULT_NARRATIVE_HALF_LIFE
    """Steps for the narrative's pull to halve (positive; ``math.inf`` never fades)."""

    value_pull: float = 0.0
    """Sentiment lost per step per unit of ln(price / start_price), non-negative.

    Off by default so a repriced stock can stay repriced. Around 0.1 a stock at
    twice its start loses ~0.07 sentiment per step, enough to end a squeeze once
    the narrative fades and to get a crash bought back.
    """

    def __post_init__(self) -> None:
        if self.num_investors < 1:
            raise ValueError("num_investors must be at least 1")
        if self.max_shares < 1:
            raise ValueError("max_shares must be at least 1")
        if not self.start_price > 0:
            raise ValueError("start_price must be positive")
        if not 0 <= self.num_trades <= self.num_investors - 1:
            raise ValueError("num_trades must be between 0 and num_investors - 1")
        for name in ("fomo_threshold", "panic_prob", "shock_prob"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if not -1.0 <= self.sentiment_skew <= 1.0:
            raise ValueError("sentiment_skew must be between -1 and 1")
        if not self.narrative_half_life > 0:
            raise ValueError("narrative_half_life must be positive")
        if self.value_pull < 0:
            raise ValueError("value_pull must be non-negative")


@dataclass
class Investor:
    """A market participant with a sentiment in ``[-1, 1]`` and share holdings.

    Holdings start as whole shares but become fractional through trading; the
    total across all investors is conserved.
    """

    sentiment: float
    holdings: float

    @property
    def state(self) -> int:
        """BUY, HOLD, or SELL, always derived from the current sentiment."""
        return state_for_sentiment(self.sentiment)

    def update_status(
        self,
        narrative: float,
        last_price_change: float,
        mispricing: float,
        config: SimulationConfig,
        rng: random.Random,
    ) -> None:
        """Apply noise and any value pull, then the FOMO, panic, and narrative rules.

        Each probabilistic rule uses its own random draw. ``narrative`` is the
        social event's current (faded) sentiment, ``last_price_change`` is the
        most recent percent price move (0.0 on the first step), and
        ``mispricing`` is ``ln(price / start_price)``.
        """
        # Private news, plus the optional lean against the price's departure from its start.
        self.sentiment += rng.uniform(-SENTIMENT_NOISE, SENTIMENT_NOISE)
        self.sentiment -= config.value_pull * mispricing
        if rng.random() < config.fomo_threshold:
            # FOMO: chase the latest price move, scaled into sentiment units.
            self.sentiment += last_price_change * FOMO_SENSITIVITY
        if rng.random() < config.panic_prob:
            # Panic: contrarian reversal, damped.
            self.sentiment *= PANIC_REVERSAL
        elif rng.random() < FOLLOW_NARRATIVE_PROB:
            # Follow the narrative: move most of the way toward the story.
            self.sentiment += FOLLOW_NARRATIVE_WEIGHT * (narrative - self.sentiment)
        self.sentiment = _clamp_sentiment(self.sentiment)

    def trade(self, other: Investor) -> float:
        """Exchange shares with ``other`` and return the number of shares moved.

        Shares move only between a buyer and a seller. The seller parts with a
        fraction of their holdings equal to the mean of the buyer's bullishness
        and the seller's bearishness, so any buyer/seller pair trades and the
        most convinced pairs trade the most. Holdings never go negative and the
        market total is conserved.
        """
        if self.state == BUY and other.state == SELL:
            buyer, seller = self, other
        elif self.state == SELL and other.state == BUY:
            buyer, seller = other, self
        else:
            return 0.0
        eagerness = (buyer.sentiment - seller.sentiment) / 2.0  # in (STATE_THRESHOLD, 1]
        amount = eagerness * seller.holdings
        seller.holdings -= amount
        buyer.holdings += amount
        return amount


class Simulation:
    """A single simulated market: a population of investors and a price series.

    ``price_history``, ``volume_history``, ``buyers``, ``holders``, and
    ``sellers`` all have one entry for the initial state plus one per step, so
    they line up index-for-index.
    """

    def __init__(
        self,
        config: SimulationConfig | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config if config is not None else SimulationConfig()
        self.rng = rng if rng is not None else random.Random()
        self.current_price: float = self.config.start_price
        self.price_history: list[float] = [self.config.start_price]
        self.volume_history: list[float] = [0.0]
        self.buyers: list[int] = []
        self.holders: list[int] = []
        self.sellers: list[int] = []
        self.investors: list[Investor] = self._create_investors()
        self._record_states()

    def _create_investors(self) -> list[Investor]:
        """Populate the market, allocating exactly ``max_shares`` whole shares."""
        config, rng = self.config, self.rng
        count = config.num_investors
        sentiments: list[float] = []
        weights: list[float] = []
        for _ in range(count):
            # Uniform sentiment in [-1, 1), shifted by the narrative and clamped.
            raw = (rng.random() * 2.0 - 1.0) + (config.sentiment_skew * INITIAL_SKEW_WEIGHT)
            sentiments.append(_clamp_sentiment(raw))
            weights.append(rng.uniform(HOLDINGS_WEIGHT_MIN, HOLDINGS_WEIGHT_MAX))

        # Largest-remainder rounding: floor each proportional share, then hand
        # the leftover whole shares to the largest fractional parts.
        total_weight = sum(weights)
        exact = [config.max_shares * weight / total_weight for weight in weights]
        whole = [int(value) for value in exact]
        leftover = config.max_shares - sum(whole)
        by_fraction = sorted(range(count), key=lambda k: exact[k] - whole[k], reverse=True)
        for index in by_fraction[:leftover]:
            whole[index] += 1

        return [
            Investor(sentiment=sentiment, holdings=float(holdings))
            for sentiment, holdings in zip(sentiments, whole, strict=True)
        ]

    def _record_states(self) -> None:
        num_buyers, num_holders, num_sellers = count_states(self.investors)
        self.buyers.append(num_buyers)
        self.holders.append(num_holders)
        self.sellers.append(num_sellers)

    @property
    def steps_completed(self) -> int:
        """Number of steps run so far."""
        return len(self.price_history) - 1

    def narrative_sentiment(self) -> float:
        """Return the social event's sentiment pull as of now, faded by its half-life."""
        config = self.config
        if math.isinf(config.narrative_half_life):
            return config.sentiment_skew
        return config.sentiment_skew * 0.5 ** (self.steps_completed / config.narrative_half_life)

    def mispricing(self) -> float:
        """Return ``ln(price / start_price)``, or 0.0 if the price has reached zero."""
        if self.current_price <= 0:
            return 0.0
        return math.log(self.current_price / self.config.start_price)

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
        """Advance one time step: trade, update sentiments, then move the price.

        Safe to call after the price has reached zero; the price then stays there.
        """
        config, rng = self.config, self.rng
        count = config.num_investors

        # Freeze the aggregate picture entering this step so that every
        # investor trades and updates against the same market state.
        num_buyers, _, num_sellers = count_states(self.investors)
        excess_demand = (num_buyers - num_sellers) / count
        narrative = self.narrative_sentiment()
        mispricing = self.mispricing()

        # Each investor trades with the next num_trades *other* investors in a
        # randomly shuffled order.
        shuffled = rng.sample(self.investors, count)
        volume = 0.0
        for i in range(count):
            for offset in range(1, config.num_trades + 1):
                volume += shuffled[i].trade(shuffled[(i + offset) % count])

        history = self.price_history
        last_price_change = 0.0
        if len(history) > 1 and history[-2] > 0:
            last_price_change = percent_change(history[-2], history[-1])
        for investor in shuffled:
            investor.update_status(narrative, last_price_change, mispricing, config, rng)

        # Multiplicative price impact: proportional to the imbalance, plus noise.
        fractional_change = PRICE_IMPACT * excess_demand + rng.gauss(0.0, RETURN_NOISE)
        self.current_price *= max(0.0, 1.0 + fractional_change)

        if rng.random() < config.shock_prob:
            # Exogenous market shock: log-symmetric random rescale.
            log_range = math.log(MARKET_SHOCK_MAX_FACTOR)
            self.current_price *= math.exp(rng.uniform(-log_range, log_range))

        self.price_history.append(self.current_price)
        self.volume_history.append(volume)
        self._record_states()

    def run(self, max_steps: int = DEFAULT_STEPS) -> list[float]:
        """Run up to ``max_steps`` steps, stopping early if the price reaches zero.

        Returns the price history (the starting price plus one entry per step).
        """
        for _ in range(max_steps):
            self.step()
            if self.current_price <= 0:
                break
        return self.price_history
