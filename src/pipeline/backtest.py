import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import Config
from src.data.fetcher import fetch_historical
from src.features.indicators import add_all_indicators
from src.features.signals import generate_rule_based_signals
from src.analysis.trading_strategy import backtest_strategy
from src.analysis.risk import full_risk_report
from src.utils.metrics import evaluate, direction_accuracy


def backtest_pipeline(
    cfg: Config,
    ticker: str,
    plot: bool = True,
) -> dict:
    df = fetch_historical(ticker, cfg.data.start_date, cfg.data.end_date)
    df = add_all_indicators(df, cfg.indicators)
    df = generate_rule_based_signals(df, cfg.indicators)

    strategy_result = backtest_strategy(
        df,
        initial_capital=cfg.strategy.initial_capital,
        position_size=cfg.strategy.position_size,
        stop_loss_pct=cfg.strategy.stop_loss_pct,
        take_profit_pct=cfg.strategy.take_profit_pct,
        trailing_stop_pct=cfg.strategy.trailing_stop_pct or 0,
        commission_pct=cfg.strategy.commission_pct,
        signal_threshold_buy=cfg.strategy.signal_threshold_buy,
        signal_threshold_sell=cfg.strategy.signal_threshold_sell,
    )

    daily_returns = df["Price_Change"].dropna().values if "Price_Change" in df else np.zeros(10)
    risk = full_risk_report(
        daily_returns,
        max_dd_pct=strategy_result.get("max_drawdown_pct", 0),
        risk_free=cfg.analysis.risk_free_rate,
    )

    result = {**strategy_result, "risk": risk}

    if plot:
        result_dir = Path(cfg.report_dir) / "figures"
        result_dir.mkdir(parents=True, exist_ok=True)
        plot_results(df, strategy_result, ticker, result_dir)

    return result


def plot_results(df: pd.DataFrame, strategy_result: dict, ticker: str, save_dir: Path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor="#111111")

    for ax in axes:
        ax.set_facecolor("#111111")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#333333")
        ax.title.set_color("white")

    ax1 = axes[0]
    ax1.plot(df.index[-len(strategy_result["equity_curve"]):], strategy_result["equity_curve"],
             color="#00ff88", linewidth=1.2, label="Equity Curve")
    ax1.axhline(y=strategy_result["initial_capital"], color="#555555", linestyle="--", linewidth=0.8)
    ax1.set_title(f"{ticker} - Equity Curve", color="white")
    ax1.legend(loc="upper left", labelcolor="white")

    ax2 = axes[1]
    equity_series = pd.Series(strategy_result["equity_curve"])
    drawdown = (equity_series.cummax() - equity_series) / equity_series.cummax() * 100
    ax2.fill_between(range(len(drawdown)), drawdown, 0, color="#ff4444", alpha=0.4)
    ax2.set_title(f"{ticker} - Drawdown", color="white")

    ax3 = axes[2]
    ax3.plot(df.index, df["Close"], color="#8888ff", linewidth=1, alpha=0.5, label="Close")
    trades = strategy_result.get("trades", [])
    buy_dates = [t["date"] for t in trades if t["type"] == "BUY"]
    sell_dates = [t["date"] for t in trades if t["type"].startswith("SELL")]
    buy_prices = [t["price"] for t in trades if t["type"] == "BUY"]
    sell_prices = [t["price"] for t in trades if t["type"].startswith("SELL")]
    ax3.scatter(buy_dates, buy_prices, color="#00ff88", marker="^", s=80, label="Buy")
    ax3.scatter(sell_dates, sell_prices, color="#ff4444", marker="v", s=80, label="Sell")
    ax3.set_title(f"{ticker} - Trades", color="white")
    ax3.legend(loc="upper left", labelcolor="white")

    plt.tight_layout()
    plt.savefig(str(save_dir / f"{ticker}_backtest.png"), dpi=150)
    plt.close()
