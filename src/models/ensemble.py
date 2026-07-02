import numpy as np
import joblib
from typing import List, Tuple, Any, Optional
from scipy.stats import mode

from src.models.base import BaseModel


class EnsembleModel(BaseModel):
    def __init__(self, models: List[BaseModel], weights: Optional[List[float]] = None):
        if not models:
            raise ValueError("EnsembleModel requires at least one model")
        self.models = models
        if weights and len(weights) == len(models):
            weights_arr = np.array(weights, dtype=float)
        else:
            weights_arr = np.ones(len(models), dtype=float)
        total = weights_arr.sum()
        self.weights = weights_arr / total if total else np.ones(len(models)) / len(models)

    def train(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> List[Any]:
        histories = []
        for model in self.models:
            h = model.train(X_train, y_train, X_val, y_val)
            histories.append(h)
        return histories

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.array([m.predict(X) for m in self.models])
        return np.average(preds, axis=0, weights=self.weights)

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        votes = np.array([m.predict(X) for m in self.models])
        weighted_votes = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int), weights=self.weights).argmax()
            if len(np.unique(x)) > 1 else x[0],
            axis=0, arr=votes,
        )
        return weighted_votes

    def predict_with_confidence(self, X: np.ndarray) -> tuple:
        preds = np.array([m.predict(X) for m in self.models])
        mean_pred = np.average(preds, axis=0, weights=self.weights)
        std_pred = np.sqrt(np.average((preds - mean_pred) ** 2, axis=0, weights=self.weights))
        return mean_pred, std_pred

    def save(self, path: str):
        for i, model in enumerate(self.models):
            model.save(f"{path}_{model.name}")
        joblib.dump({"weights": self.weights, "model_names": [m.name for m in self.models]}, f"{path}_meta")

    def load(self, path: str) -> "EnsembleModel":
        raise NotImplementedError("Load each model individually")

    @property
    def name(self) -> str:
        return "ensemble"
