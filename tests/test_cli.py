"""Smoke and behavior tests for the command-line interface."""

import pytest

from stock_simulator.cli import build_parser, main


def test_smoke_writes_both_charts(tmp_path, capsys):
    price_path = tmp_path / "price.png"
    investor_path = tmp_path / "investors.png"
    exit_code = main(
        [
            "--seed",
            "42",
            "--steps",
            "50",
            "--price-chart",
            str(price_path),
            "--investor-chart",
            str(investor_path),
        ]
    )
    assert exit_code == 0
    assert price_path.stat().st_size > 0
    assert investor_path.stat().st_size > 0
    out = capsys.readouterr().out
    assert "Current Price:" in out
    assert "Percent Change (DAILY):" in out
    assert "Percent Change (ALL TIME):" in out


def test_same_seed_gives_same_summary(tmp_path, capsys):
    def run(name: str) -> str:
        main(
            [
                "--seed",
                "7",
                "--steps",
                "100",
                "--sentiment",
                "BULLISH",
                "--price-chart",
                str(tmp_path / f"{name}-price.png"),
                "--investor-chart",
                str(tmp_path / f"{name}-investors.png"),
            ]
        )
        out = capsys.readouterr().out
        return out[: out.index("Saved charts")]  # drop the path-dependent line

    assert run("first") == run("second")


def test_verbose_prints_market_stats(tmp_path, capsys):
    main(
        [
            "--seed",
            "1",
            "--steps",
            "20",
            "--verbose",
            "--price-chart",
            str(tmp_path / "price.png"),
            "--investor-chart",
            str(tmp_path / "investors.png"),
        ]
    )
    out = capsys.readouterr().out
    assert "Total Holdings = 100,000" in out
    assert "Sentiment Skew = VERY_BEAR" in out


def test_output_directories_are_created(tmp_path):
    nested = tmp_path / "a" / "b"
    exit_code = main(
        [
            "--seed",
            "3",
            "--steps",
            "10",
            "--price-chart",
            str(nested / "price.png"),
            "--investor-chart",
            str(nested / "investors.png"),
        ]
    )
    assert exit_code == 0
    assert (nested / "price.png").exists()


def test_invalid_sentiment_rejected():
    with pytest.raises(SystemExit):
        main(["--sentiment", "SIDEWAYS"])


@pytest.mark.parametrize(
    "argv",
    [
        ["--investors", "0"],
        ["--trades", "-1"],
        ["--trades", "100"],  # 100 investors have only 99 counterparties each
        ["--steps", "0"],
        ["--fomo-threshold", "1.5"],
        ["--fomo-threshold", "-0.1"],
        ["--panic-prob", "2"],
        ["--shock-prob", "-1"],
        ["--shock-prob", "1.01"],
        ["--narrative-half-life", "0"],
        ["--value-pull", "-1"],
        ["--event-step", "-1"],
    ],
)
def test_invalid_numeric_arguments_rejected(argv, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_zero_trades_is_allowed(tmp_path):
    exit_code = main(
        [
            "--trades",
            "0",
            "--seed",
            "5",
            "--steps",
            "10",
            "--price-chart",
            str(tmp_path / "price.png"),
            "--investor-chart",
            str(tmp_path / "investors.png"),
        ]
    )
    assert exit_code == 0


def test_late_event_run_writes_charts(tmp_path):
    exit_code = main(
        [
            "--event-step",
            "10",
            "--steps",
            "30",
            "--seed",
            "2",
            "--price-chart",
            str(tmp_path / "price.png"),
            "--investor-chart",
            str(tmp_path / "investors.png"),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "price.png").stat().st_size > 0
    assert (tmp_path / "investors.png").stat().st_size > 0


def test_parser_defaults():
    args = build_parser().parse_args([])
    assert args.sentiment == "VERY_BEAR"
    assert args.investors == 100
    assert args.trades == 5
    assert args.fomo_threshold == 0.5
    assert args.panic_prob == 0.2
    assert args.shock_prob == 0.025
    assert args.narrative_half_life == 50.0
    assert args.value_pull == 0.0
    assert args.event_step == 0
    assert args.steps == 500
    assert args.seed is None
    assert args.price_chart == "price_chart.png"
    assert args.investor_chart == "investor_chart.png"
