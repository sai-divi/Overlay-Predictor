import pandas as pd
import numpy as np


def generate_rule_based_signals(df: pd.DataFrame, config=None) -> pd.DataFrame:
    df = df.copy()
    rsi_period = config.rsi_period if config else 14
    rsi_ob = config.rsi_overbought if config else 70
    rsi_os = config.rsi_oversold if config else 30

    rsi_col = f"RSI_{rsi_period}"

    conditions_buy = []
    conditions_sell = []

    if rsi_col in df:
        conditions_buy.append(df[rsi_col] < rsi_os)
        conditions_sell.append(df[rsi_col] > rsi_ob)

    if "MACD_hist" in df:
        conditions_buy.append((df["MACD_hist"] > 0) & (df["MACD_hist"].shift(1) <= 0))
        conditions_sell.append((df["MACD_hist"] < 0) & (df["MACD_hist"].shift(1) >= 0))

    if "BB_position" in df:
        conditions_buy.append(df["BB_position"] < 0.05)
        conditions_sell.append(df["BB_position"] > 0.95)

    if "Stoch_%K" in df:
        conditions_buy.append((df["Stoch_%K"] < 20) & (df["Stoch_%K"] > df["Stoch_%K"].shift(1)))
        conditions_sell.append((df["Stoch_%K"] > 80) & (df["Stoch_%K"] < df["Stoch_%K"].shift(1)))

    if "Volume_Change" in df:
        conditions_buy.append((df["Close"] > df["Close"].shift(1)) & (df["Volume_Change"] > 0.5))
        conditions_sell.append((df["Close"] < df["Close"].shift(1)) & (df["Volume_Change"] > 0.5))

    if "SMA_50" in df and "SMA_200" in df:
        conditions_buy.append((df["SMA_50"] > df["SMA_200"]) & (df["SMA_50"].shift(1) <= df["SMA_200"].shift(1)))
        conditions_sell.append((df["SMA_50"] < df["SMA_200"]) & (df["SMA_50"].shift(1) >= df["SMA_200"].shift(1)))

    df["Signal_Buy_Rule"] = (pd.concat(conditions_buy, axis=1).sum(axis=1) if conditions_buy else 0) / len(conditions_buy) if conditions_buy else 0
    df["Signal_Sell_Rule"] = (pd.concat(conditions_sell, axis=1).sum(axis=1) if conditions_sell else 0) / len(conditions_sell) if conditions_sell else 0

    df["Signal_Strength"] = df["Signal_Buy_Rule"] - df["Signal_Sell_Rule"]
    return df


def create_target_labels(df: pd.DataFrame, horizon: int = 1, threshold_pct: float = 0.0) -> pd.DataFrame:
    df = df.copy()
    future_price = df["Close"].shift(-horizon)
    future_return = (future_price - df["Close"]) / df["Close"]

    df["target_Return"] = future_return
    df["target_Class"] = 0
    df.loc[future_return > threshold_pct, "target_Class"] = 1
    df.loc[future_return < -threshold_pct, "target_Class"] = -1
    return df
