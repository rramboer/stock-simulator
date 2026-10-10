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
    """Build the argument parser (defaults match the original hardcoded model)."""
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
        help=f"initial market sentiment skew (default: {DEFAULT_SENTIMENT})",
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
        help=f"trades per investor per time step (default: {_DEFAULTS.num_trades})",
    )
    parser.add_argument(
        "--fomo-threshold",
        type=float,
        default=_DEFAULTS.fomo_threshold,
        help=(
            "susceptibility to FOMO in (0, 1]; 0 is none, 1 is maximum "
            f"(default: {_DEFAULTS.fomo_threshold})"
        ),
    )
    parser.add_argument(
        "--panic-prob",
        type=float,
        default=_DEFAULTS.panic_prob,
        help=(
            f"per-step probability that an investor panic trades (default: {_DEFAULTS.panic_prob})"
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=(
            "maximum number of time steps; the run stops early if the price "
            f"hits zero (default: {DEFAULT_STEPS})"
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
    if args.investors < 1:
        parser.error("--investors must be at least 1")
    if args.trades < 0:
        parser.error("--trades must not be negative")
    if args.steps < 1:
        parser.error("--steps must be at least 1")

    config = SimulationConfig(
        num_investors=args.investors,
        num_trades=args.trades,
        fomo_threshold=args.fomo_threshold,
        panic_prob=args.panic_prob,
        sentiment_skew=SENTIMENT_SKEWS[args.sentiment],
    )
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
    print(f"Current Price: {current:.2f}")
    print(f"Percent Change (DAILY): {percent_change(previous, current):.2f}%")
    print(f"Percent Change (ALL TIME): {percent_change(config.start_price, current):.2f}%")
    if args.verbose:
        _print_market_stats(simulation, args.sentiment)
    print(f"Saved charts to {price_path} and {investor_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
