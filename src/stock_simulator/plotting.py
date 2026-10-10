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

from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _mark_event(axes: Axes, event_step: int | None) -> None:
    """Draw a dashed line where the social event hits, unless it is at the start."""
    if event_step:
        axes.axvline(event_step, color="gray", linestyle="--", linewidth=1, label="Social event")


def save_price_chart(
    price_history: list[float],
    path: str | Path,
    title: str = "Stock Price",
    event_step: int | None = None,
) -> Path:
    """Plot the price series and write it to ``path`` as a PNG.

    A dashed vertical line marks ``event_step`` when the social event hits after
    the start.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(8.0, 4.5))
    axes = figure.subplots()
    axes.plot(price_history, label="Stock Price")
    _mark_event(axes, event_step)
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
    event_step: int | None = None,
) -> Path:
    """Plot the buyer/holder/seller counts and write them to ``path`` as a PNG.

    A dashed vertical line marks ``event_step`` when the social event hits after
    the start.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(8.0, 4.5))
    axes = figure.subplots()
    axes.plot(buyers, label="Buyers")
    axes.plot(holders, label="Holders")
    axes.plot(sellers, label="Sellers")
    _mark_event(axes, event_step)
    axes.set_title(title)
    axes.set_xlabel("Time step")
    axes.set_ylabel("Number of investors")
    axes.legend()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    return path
