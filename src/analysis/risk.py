import numpy as np
import pandas as pd


def compute_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    return np.percentile(returns, (1 - confidence) * 100)


def compute_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    var = compute_var(returns, confidence)
    return returns[returns <= var].mean()


def compute_beta(returns: np.ndarray, market_returns: np.ndarray) -> float:
    if len(returns) != len(market_returns) or len(returns) < 2:
        return 0.0
    cov = np.cov(returns, market_returns)[0, 1]
    market_var = np.var(market_returns)
    return cov / market_var if market_var != 0 else 0.0


def compute_sortino(returns: np.ndarray, risk_free: float = 0.05) -> float:
    excess = returns - risk_free / 252
    downside = returns[returns < 0]
    if len(downside) == 0 or np.std(downside) == 0:
        return 0.0
    return np.mean(excess) / np.std(downside) * np.sqrt(252)


def compute_calmar(returns: np.ndarray, max_drawdown_pct: float) -> float:
    if max_drawdown_pct == 0:
        return 0.0
    annual_return = np.mean(returns) * 252
    return annual_return / (max_drawdown_pct / 100)


def full_risk_report(returns: np.ndarray, max_dd_pct: float = 0, market_returns: np.ndarray = None, risk_free: float = 0.05) -> dict:
    return {
        "volatility_pct": round(np.std(returns) * np.sqrt(252) * 100, 2),
        "sharpe_ratio": round(np.mean(returns - risk_free / 252) / np.std(returns) * np.sqrt(252), 2) if np.std(returns) > 0 else 0,
        "sortino_ratio": round(compute_sortino(returns, risk_free), 2),
        "calmar_ratio": round(compute_calmar(returns, max_dd_pct), 2),
        "var_95": round(compute_var(returns) * 100, 2),
        "cvar_95": round(compute_cvar(returns) * 100, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "beta": round(compute_beta(returns, market_returns), 2) if market_returns is not None else None,
    }
