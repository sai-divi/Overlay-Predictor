import numpy as np
from typing import Any, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from src.models.base import BaseModel


class XGBoostModel(BaseModel):
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.7,
        min_child_weight: int = 3,
        gamma: float = 0.1,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        objective: str = "reg:squarederror",
        num_class: Optional[int] = None,
        early_stopping_rounds: Optional[int] = 50,
    ):
        self.early_stopping_rounds = early_stopping_rounds
        self.num_class = num_class
        self.model = None
        self.feature_importances_ = None
        self.train_score = None
        self.val_score = None

    def train(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray = None, y_val: np.ndarray = None,
    ) -> Any:
        if X_train.ndim == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
            if X_val is not None:
                X_val = X_val.reshape(X_val.shape[0], -1)

        if self.num_class:
            self.model = RandomForestClassifier(
                n_estimators=300, max_depth=self.num_class * 2,
                random_state=42, n_jobs=-1
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=300, max_depth=10,
                random_state=42, n_jobs=-1
            )

        self.model.fit(X_train, y_train)

        if hasattr(self.model, "feature_importances_"):
            self.feature_importances_ = self.model.feature_importances_

        if X_val is not None and y_val is not None:
            self.val_score = self.model.score(X_val, y_val)

        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def get_top_features(self, feature_names: list, n: int = 10) -> list:
        if self.feature_importances_ is None:
            return []
        idx = np.argsort(self.feature_importances_)[::-1][:n]
        return [(feature_names[i], self.feature_importances_[i]) for i in idx]

    def save(self, path: str):
        import joblib
        joblib.dump(self.model, path)

    def load(self, path: str) -> "XGBoostModel":
        import joblib
        self.model = joblib.load(path)
        return self

    @property
    def name(self) -> str:
        return "random_forest"
