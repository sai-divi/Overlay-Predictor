import pandas as pd
import numpy as np
from typing import List, Optional

from src.features.indicators import add_all_indicators


def engineer_features(
    df: pd.DataFrame,
    additional_features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    df = add_all_indicators(df)

    if additional_features is not None:
        for col in additional_features.columns:
            df[col] = additional_features[col]

    return df.dropna()
