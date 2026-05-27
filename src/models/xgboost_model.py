import numpy as np
import xgboost as xgb
from typing import Any, Optional

from src.models.base import BaseModel


class XGBoostModel(BaseModel):
    def __init__(
        self,
        n_estimators: int = 1000,
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
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "objective": "multi:softprob" if num_class else objective,
            "num_class": num_class,
            "random_state": 42,
        }
        self.num_class = num_class
        self.model: xgb.XGBModel = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.train_score: Optional[float] = None
        self.val_score: Optional[float] = None

    def train(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray = None, y_val: np.ndarray = None,
    ) -> Any:
        if X_train.ndim == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
            if X_val is not None:
                X_val = X_val.reshape(X_val.shape[0], -1)

        params = dict(self.params)
        if self.early_stopping_rounds:
            params["early_stopping_rounds"] = self.early_stopping_rounds

        if self.num_class:
            self.model = xgb.XGBClassifier(**params)
        else:
            self.model = xgb.XGBRegressor(**params)

        fit_kwargs = {"verbose": False}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["eval_metric"] = "mlogloss" if self.num_class else "rmse"

        self.model.fit(X_train, y_train, **fit_kwargs)

        # Track feature importance
        if hasattr(self.model, "feature_importances_"):
            self.feature_importances_ = self.model.feature_importances_

        # Track best score
        if hasattr(self.model, "best_score"):
            self.val_score = self.model.best_score
        if hasattr(self.model, "best_iteration"):
            self.best_iteration = self.model.best_iteration

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
        self.model.save_model(path)

    def load(self, path: str) -> "XGBoostModel":
        cls = xgb.XGBClassifier if self.num_class else xgb.XGBRegressor
        self.model = cls()
        self.model.load_model(path)
        return self

    @property
    def name(self) -> str:
        return "xgboost"
