import pandas as pd
import numpy as np
from typing import List, Optional

from src.features.indicators import add_all_indicators


def engineer_features(
    df: pd.DataFrame,
    additional_features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    df = add_all_indicators(df)

    df["Price_Change"] = df["Close"].pct_change()
    df["High_Low_Ratio"] = df["High"] / df["Low"]
    df["Close_Open_Ratio"] = df["Close"] / df["Open"]
    df["Volume_Change"] = df["Volume"].pct_change()

    if additional_features is not None:
        for col in additional_features.columns:
            df[col] = additional_features[col]

    return df.dropna()
