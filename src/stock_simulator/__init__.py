"""Agent-based Monte Carlo simulator of stock rallies and crashes."""

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

__version__ = "1.0.0"

__all__ = [
    "BUY",
    "HOLD",
    "SELL",
    "SENTIMENT_SKEWS",
    "Investor",
    "Simulation",
    "SimulationConfig",
    "__version__",
    "count_states",
    "percent_change",
    "state_for_sentiment",
]
