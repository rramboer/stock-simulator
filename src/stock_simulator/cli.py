"""Command-line interface for the stock simulator."""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence

from stock_simulator.plotting import save_investor_chart, save_price_chart
from stock_simulator.simulation import (
    DEFAULT_SENTIMENT,
    DEFAULT_STEPS,
    SENTIMENT_SKEWS,
    Simulation,
    SimulationConfig,
    count_states,
    percent_change,
)

_DEFAULTS = SimulationConfig()


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="stock-simulator",
        description=(
            "Agent-based Monte Carlo simulation of a stock rally or crash driven by "
            "investor sentiment, FOMO, and panic. Writes the price and investor "
            "charts as PNG files (headless; no display needed)."
        ),
    )
    parser.add_argument(
        "--sentiment",
        choices=sorted(SENTIMENT_SKEWS),
        default=DEFAULT_SENTIMENT,
        help=(
            "narrative sentiment of the social event the run starts just after "
            f"(default: {DEFAULT_SENTIMENT})"
        ),
    )
    parser.add_argument(
        "--investors",
        type=int,
        default=_DEFAULTS.num_investors,
        help=f"number of investors in the market (default: {_DEFAULTS.num_investors})",
    )
    parser.add_argument(
        "--trades",
        type=int,
        default=_DEFAULTS.num_trades,
        help=(
            "distinct counterparties each investor trades with per time step; "
            f"0 disables trading (default: {_DEFAULTS.num_trades})"
        ),
    )
    parser.add_argument(
        "--fomo-threshold",
        type=float,
        default=_DEFAULTS.fomo_threshold,
        help=(
            "per-step probability that an investor chases the last price move, "
            f"in [0, 1] (default: {_DEFAULTS.fomo_threshold})"
        ),
    )
    parser.add_argument(
        "--panic-prob",
        type=float,
        default=_DEFAULTS.panic_prob,
        help=(
            "per-step probability that an investor panics and reverses sentiment, "
            f"in [0, 1] (default: {_DEFAULTS.panic_prob})"
        ),
    )
    parser.add_argument(
        "--shock-prob",
        type=float,
        default=_DEFAULTS.shock_prob,
        help=(
            "per-step probability of an exogenous market shock that rescales the "
            f"price by a factor in [0.75, 1.25), in [0, 1] (default: {_DEFAULTS.shock_prob})"
        ),
    )
    parser.add_argument(
        "--narrative-half-life",
        type=float,
        default=_DEFAULTS.narrative_half_life,
        metavar="STEPS",
        help=(
            "steps for the social event's pull on sentiment to halve; use inf for a "
            f"story that never fades (default: {_DEFAULTS.narrative_half_life:g})"
        ),
    )
    parser.add_argument(
        "--value-pull",
        type=float,
        default=_DEFAULTS.value_pull,
        metavar="RATE",
        help=(
            "sentiment lost per step per unit of ln(price / start price), leaning the "
            "crowd against how far the price has strayed; 0 lets a repriced stock stay "
            f"repriced, around 0.1 makes squeezes collapse (default: {_DEFAULTS.value_pull:g})"
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=(
            "maximum number of time steps; the run stops early if the price "
            f"reaches zero (default: {DEFAULT_STEPS})"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for a reproducible run (default: unseeded, randomized)",
    )
    parser.add_argument(
        "--price-chart",
        default="price_chart.png",
        metavar="PATH",
        help="output path for the price chart PNG (default: price_chart.png)",
    )
    parser.add_argument(
        "--investor-chart",
        default="investor_chart.png",
        metavar="PATH",
        help="output path for the investor chart PNG (default: investor_chart.png)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="also print end-of-run market statistics",
    )
    return parser


def _print_market_stats(simulation: Simulation, sentiment_label: str) -> None:
    """Print end-of-run market statistics (holdings are fractional by design)."""
    num_buyers, num_holders, num_sellers = count_states(simulation.investors)
    print(f"Total Holdings = {simulation.total_holdings():,.0f}")
    print(f"Average Sentiment = {simulation.average_sentiment():.4f}")
    print(f"Total Buyers = {num_buyers}")
    print(f"Total Holders = {num_holders}")
    print(f"Total Sellers = {num_sellers}")
    print(f"Sentiment Skew = {sentiment_label}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run a simulation, save both charts, and print the price summary."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be at least 1")

    try:
        config = SimulationConfig(
            num_investors=args.investors,
            num_trades=args.trades,
            fomo_threshold=args.fomo_threshold,
            panic_prob=args.panic_prob,
            shock_prob=args.shock_prob,
            narrative_half_life=args.narrative_half_life,
            value_pull=args.value_pull,
            sentiment_skew=SENTIMENT_SKEWS[args.sentiment],
        )
    except ValueError as error:
        parser.error(str(error))

    simulation = Simulation(config, rng=random.Random(args.seed))
    simulation.run(args.steps)

    price_path = save_price_chart(
        simulation.price_history,
        args.price_chart,
        title=f"Stock Price (sentiment: {args.sentiment})",
    )
    investor_path = save_investor_chart(
        simulation.buyers,
        simulation.holders,
        simulation.sellers,
        args.investor_chart,
        title=f"Investor States (sentiment: {args.sentiment})",
    )

    history = simulation.price_history
    current = history[-1]
    previous = history[-2] if len(history) > 1 else current
    daily = percent_change(previous, current) if previous > 0 else 0.0
    print(f"Current Price: {current:.2f}")
    print(f"Percent Change (DAILY): {daily:.2f}%")
    print(f"Percent Change (ALL TIME): {percent_change(config.start_price, current):.2f}%")
    if args.verbose:
        _print_market_stats(simulation, args.sentiment)
    print(f"Saved charts to {price_path} and {investor_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
