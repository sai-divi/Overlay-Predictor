import pandas as pd
import numpy as np


def backtest_strategy(
    df: pd.DataFrame,
    signal_col: str = "Signal_Strength",
    initial_capital: float = 10000.0,
    position_size: float = 0.95,
    stop_loss_pct: float = 0.05,
    take_profit_pct: float = 0.15,
    trailing_stop_pct: float = 0.03,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.001,
    signal_threshold_buy: float = 0.3,
    signal_threshold_sell: float = -0.3,
) -> dict:
    df = df.copy().dropna()
    n = len(df)
    if n < 2:
        return {"error": "Not enough data"}

    cash = initial_capital
    shares = 0.0
    entry_price = 0.0
    trailing_stop = 0.0
    in_position = False

    trades = []
    equity_curve = []
    peak = initial_capital

    for i in range(1, n):
        price = df["Close"].iloc[i]
        signal = df[signal_col].iloc[i]
        date = df.index[i]

        if not in_position:
            if signal >= signal_threshold_buy:
                buy_price = price * (1 + slippage_pct)
                commission = buy_price * commission_pct
                invest = cash * position_size
                shares = (invest - commission) / buy_price
                cash -= invest
                entry_price = buy_price
                trailing_stop = buy_price * (1 - trailing_stop_pct)
                in_position = True
                trades.append({"date": date, "type": "BUY", "price": buy_price, "shares": shares})
        else:
            stop_price = max(trailing_stop, entry_price * (1 - stop_loss_pct))
            trailing_stop = max(trailing_stop, price * (1 - trailing_stop_pct))

            sell_signal = signal <= signal_threshold_sell
            stop_hit = price <= stop_price
            take_profit = price >= entry_price * (1 + take_profit_pct)

            if sell_signal or stop_hit or take_profit:
                sell_price = price * (1 - slippage_pct)
                commission = sell_price * commission_pct
                proceeds = shares * sell_price - commission
                cash += proceeds
                pnl = proceeds - (shares * entry_price)
                trades.append({"date": date, "type": "SELL", "price": sell_price, "shares": shares, "pnl": pnl})
                shares = 0.0
                in_position = False

        net_worth = cash + shares * price
        equity_curve.append(net_worth)
        peak = max(peak, net_worth)

    if in_position:
        final_price = df["Close"].iloc[-1] * (1 - slippage_pct)
        commission = final_price * commission_pct
        proceeds = shares * final_price - commission
        cash += proceeds
        trades.append({"date": df.index[-1], "type": "SELL(CLOSE)", "price": final_price, "shares": shares, "pnl": proceeds - shares * entry_price})
        shares = 0.0

    final_value = cash
    returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
    total_return = (final_value - initial_capital) / initial_capital * 100

    equity_series = pd.Series(equity_curve, index=df.index[-len(equity_curve):])
    drawdown = (equity_series.cummax() - equity_series) / equity_series.cummax() * 100
    max_dd = drawdown.max()

    winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
    losing_trades = [t for t in trades if t.get("pnl", 0) < 0]

    return {
        "initial_capital": initial_capital,
        "final_value": round(final_value, 2),
        "total_return_pct": round(total_return, 2),
        "total_profit": round(final_value - initial_capital, 2),
        "num_trades": len([t for t in trades if t["type"].startswith("SELL")]),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": round(len(winning_trades) / max(len([t for t in trades if t["type"].startswith("SELL")]), 1) * 100, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "avg_profit_per_trade": round(np.mean([t.get("pnl", 0) for t in trades if t["type"].startswith("SELL")]), 2) if any(t["type"].startswith("SELL") for t in trades) else 0,
        "avg_return_per_trade_pct": round(np.mean(returns) * 100, 4) if len(returns) > 0 else 0,
        "equity_curve": equity_curve,
        "trades": trades,
        "sharpe_ratio": round(np.mean(returns) / np.std(returns) * np.sqrt(252), 2) if len(returns) > 1 and np.std(returns) > 0 else 0,
    }


def monte_carlo_projection(
    daily_returns: np.ndarray,
    initial_capital: float = 10000.0,
    num_simulations: int = 1000,
    forecast_days: int = 252,
    confidence_level: float = 0.95,
) -> dict:
    if len(daily_returns) < 10:
        return {"error": "Not enough return data"}

    mu = np.mean(daily_returns)
    sigma = np.std(daily_returns)

    simulations = np.zeros((forecast_days, num_simulations))
    for i in range(num_simulations):
        rand_rets = np.random.normal(mu, sigma, forecast_days)
        price_path = initial_capital * np.cumprod(1 + rand_rets)
        simulations[:, i] = price_path

    final_values = simulations[-1, :]
    pct = (1 - confidence_level) / 2
    lower = np.percentile(final_values, pct * 100)
    upper = np.percentile(final_values, (1 - pct) * 100)
    median = np.median(final_values)

    target_profit = initial_capital * 0.1
    days_to_target = []
    for i in range(num_simulations):
        hits = np.where(simulations[:, i] >= initial_capital + target_profit)[0]
        if len(hits) > 0:
            days_to_target.append(hits[0])

    return {
        "initial_capital": initial_capital,
        "median_final_value": round(median, 2),
        "lower_bound": round(lower, 2),
        "upper_bound": round(upper, 2),
        "expected_return_pct": round((median - initial_capital) / initial_capital * 100, 2),
        "prob_profit_pct": round(np.mean(final_values > initial_capital) * 100, 2),
        "median_days_to_10pct_profit": round(np.median(days_to_target)) if days_to_target else None,
        "prob_10pct_in_1yr": round(len(days_to_target) / num_simulations * 100, 2),
        "simulations": simulations[:, :100].tolist(),
    }
