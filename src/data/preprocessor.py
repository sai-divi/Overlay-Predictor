import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional
import joblib


class Preprocessor:
    def __init__(self, scaler: Optional[StandardScaler] = None):
        self.scaler = scaler
        self.feature_cols: List[str] = []

    def _feature_frame(self, df: pd.DataFrame, allow_missing: bool = False) -> pd.DataFrame:
        missing = [col for col in self.feature_cols if col not in df.columns]
        if missing and not allow_missing:
            raise KeyError(f"Missing feature columns: {', '.join(missing)}")

        out = df.copy()
        for col in missing:
            out[col] = 0.0
        out = out[self.feature_cols].replace([np.inf, -np.inf], np.nan)
        return out

    def fit(self, df: pd.DataFrame, feature_cols: List[str]) -> "Preprocessor":
        self.feature_cols = list(feature_cols)
        X = self._feature_frame(df).dropna()
        if X.empty:
            raise ValueError("No complete rows available to fit preprocessor")
        self.scaler = StandardScaler()
        self.scaler.fit(X.values)
        return self

    def fit_transform(
        self, df: pd.DataFrame, feature_cols: List[str], target_col: str = "Close"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        self.feature_cols = list(feature_cols)
        if target_col not in df.columns:
            raise KeyError(f"Missing target column: {target_col}")
        target_name = "__target__"
        clean = pd.concat([self._feature_frame(df), df[target_col].rename(target_name)], axis=1)
        clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            raise ValueError("No complete rows available to fit preprocessor")

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(clean[self.feature_cols].values)

        X_df = pd.DataFrame(X_scaled, index=clean.index, columns=self.feature_cols)
        y_series = pd.Series(clean[target_name].values, index=clean.index, name=target_col)
        return X_df, y_series

    def transform(self, df: pd.DataFrame, allow_missing: bool = True, dropna: bool = True) -> pd.DataFrame:
        if self.scaler is None:
            raise ValueError("Preprocessor has not been fitted")
        X = self._feature_frame(df, allow_missing=allow_missing)
        if dropna:
            X = X.dropna()
        else:
            X = X.ffill().bfill().fillna(0.0)
        if X.empty:
            return pd.DataFrame(columns=self.feature_cols, index=X.index)
        X_scaled = self.scaler.transform(X.values)
        return pd.DataFrame(X_scaled, index=X.index, columns=self.feature_cols)

    def transform_with_target(
        self,
        df: pd.DataFrame,
        target_col: str = "Close",
        allow_missing: bool = False,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if target_col not in df.columns:
            raise KeyError(f"Missing target column: {target_col}")
        X_raw = self._feature_frame(df, allow_missing=allow_missing)
        target_name = "__target__"
        clean = pd.concat([X_raw, df[target_col].rename(target_name)], axis=1)
        clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
        X_scaled = self.scaler.transform(clean[self.feature_cols].values)
        X_df = pd.DataFrame(X_scaled, index=clean.index, columns=self.feature_cols)
        y_series = pd.Series(clean[target_name].values, index=clean.index, name=target_col)
        return X_df, y_series

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
    if len(X) <= seq_length:
        feature_shape = X.shape[1:] if X.ndim > 1 else (1,)
        return np.empty((0, seq_length, *feature_shape)), np.empty((0,))
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
