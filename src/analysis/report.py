import numpy as np
import pandas as pd
from src.data.fetcher import lookup_ticker_info, fetch_historical
from src.features.indicators import add_all_indicators
from src.features.signals import generate_rule_based_signals, create_target_labels
from src.analysis.trading_strategy import backtest_strategy, monte_carlo_projection
from src.analysis.risk import full_risk_report


def deep_analysis(ticker: str, start: str, end: str, cfg) -> dict:
    info = lookup_ticker_info(ticker)
    df = fetch_historical(ticker, start, end)
    df = add_all_indicators(df, cfg.indicators)
    df = generate_rule_based_signals(df, cfg.indicators)
    df = create_target_labels(df, horizon=cfg.data.prediction_horizon)

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

    daily_returns = df["Price_Change"].dropna().values
    risk = full_risk_report(
        daily_returns,
        max_dd_pct=strategy_result.get("max_drawdown_pct", 0),
        risk_free=cfg.analysis.risk_free_rate,
    )

    mcs = monte_carlo_projection(
        daily_returns,
        initial_capital=cfg.strategy.initial_capital,
        num_simulations=cfg.analysis.monte_carlo_sims,
        forecast_days=cfg.analysis.monte_carlo_days,
        confidence_level=cfg.analysis.confidence_level,
    )

    current_signal = df["Signal_Strength"].iloc[-1] if "Signal_Strength" in df else 0
    last_rsi = df.get(f"RSI_{cfg.indicators.rsi_period}", pd.Series([50])).iloc[-1]
    last_price = df["Close"].iloc[-1]

    recommendation = "HOLD"
    if current_signal > cfg.strategy.signal_threshold_buy:
        recommendation = "BUY"
    elif current_signal < cfg.strategy.signal_threshold_sell:
        recommendation = "SELL"

    return {
        "ticker": ticker,
        "company": info,
        "current_price": round(last_price, 2),
        "current_rsi": round(last_rsi, 1),
        "signal_strength": round(current_signal, 3),
        "recommendation": recommendation,
        "strategy": strategy_result,
        "risk": risk,
        "projection": mcs,
        "date_range": f"{start} to {end}",
        "data_points": len(df),
    }
