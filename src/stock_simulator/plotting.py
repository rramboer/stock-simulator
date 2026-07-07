"""Headless chart rendering for simulation results.

Charts are written straight to PNG files with ``savefig``; nothing is ever
shown interactively, so the simulator runs fine on machines with no display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Force the non-interactive Agg backend before any rendering machinery loads,
# so chart generation works on headless machines.
matplotlib.use("Agg")

from matplotlib.figure import Figure


def save_price_chart(
    price_history: list[float],
    path: str | Path,
    title: str = "Stock Price",
) -> Path:
    """Plot the price series and write it to ``path`` as a PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(8.0, 4.5))
    axes = figure.subplots()
    axes.plot(price_history, label="Stock Price")
    axes.set_title(title)
    axes.set_xlabel("Time step")
    axes.set_ylabel("Price")
    axes.legend()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    return path


def save_investor_chart(
    buyers: list[int],
    holders: list[int],
    sellers: list[int],
    path: str | Path,
    title: str = "Investor States",
) -> Path:
    """Plot the buyer/holder/seller counts and write them to ``path`` as a PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(8.0, 4.5))
    axes = figure.subplots()
    axes.plot(buyers, label="Buyers")
    axes.plot(holders, label="Holders")
    axes.plot(sellers, label="Sellers")
    axes.set_title(title)
    axes.set_xlabel("Time step")
    axes.set_ylabel("Number of investors")
    axes.legend()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    return path
