import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional
import joblib


class Preprocessor:
    def __init__(self, scaler: Optional[StandardScaler] = None):
        self.scaler = scaler
        self.feature_cols: List[str] = []

    def fit_transform(
        self, df: pd.DataFrame, feature_cols: List[str], target_col: str = "Close"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        self.feature_cols = feature_cols
        df = df.dropna().copy()
        X = df[feature_cols].values
        y = df[target_col].values

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        X_df = pd.DataFrame(X_scaled, index=df.index, columns=feature_cols)
        y_series = pd.Series(y, index=df.index, name=target_col)
        return X_df, y_series

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, index=df.index, columns=self.feature_cols)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(X)

    def save(self, path: str):
        joblib.dump({"scaler": self.scaler, "feature_cols": self.feature_cols}, path)

    @classmethod
    def load(cls, path: str) -> "Preprocessor":
        data = joblib.load(path)
        p = Preprocessor(scaler=data["scaler"])
        p.feature_cols = data["feature_cols"]
        return p


def create_sequences(
    X: np.ndarray, y: np.ndarray, seq_length: int = 60
) -> Tuple[np.ndarray, np.ndarray]:
    X_seq, y_seq = [], []
    for i in range(seq_length, len(X)):
        X_seq.append(X[i - seq_length : i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_split: float = 0.8,
    val_split: float = 0.1,
) -> Tuple:
    n = len(X)
    train_end = int(n * train_split)
    val_end = train_end + int(n * val_split)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    return X_train, X_val, X_test, y_train, y_val, y_test
