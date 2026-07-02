import warnings
import numpy as np
from typing import Any, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from src.models.base import BaseModel

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None


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
        random_state: int = 42,
        use_grid_search: bool = False,
        grid_search_cv: int = 5,
        class_weight_balanced: bool = True,
    ):
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
            "objective": objective,
            "random_state": random_state,
        }
        self.early_stopping_rounds = early_stopping_rounds
        self.num_class = num_class
        self.class_labels_ = None
        self.backend = "xgboost" if XGBRegressor is not None else "random_forest"
        self.model = None
        self.feature_importances_ = None
        self.train_score = None
        self.val_score = None
        self.use_grid_search = use_grid_search
        self.grid_search_cv = grid_search_cv
        self.class_weight_balanced = class_weight_balanced
        self.best_params_ = None

    def train(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray = None, y_val: np.ndarray = None,
    ) -> Any:
        if X_train.ndim == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
            if X_val is not None:
                X_val = X_val.reshape(X_val.shape[0], -1)

        is_classification = self.num_class is not None
        y_fit = y_train
        y_val_fit = y_val

        if is_classification:
            labels = np.array(sorted(np.unique(y_train)))
            self.class_labels_ = labels
            y_fit = np.searchsorted(labels, y_train)
            if y_val is not None:
                y_val_fit = np.searchsorted(labels, y_val)

        if XGBRegressor is not None:
            base_params = {
                "n_estimators": self.params["n_estimators"],
                "max_depth": self.params["max_depth"],
                "learning_rate": self.params["learning_rate"],
                "subsample": self.params["subsample"],
                "colsample_bytree": self.params["colsample_bytree"],
                "min_child_weight": self.params["min_child_weight"],
                "gamma": self.params["gamma"],
                "reg_alpha": self.params["reg_alpha"],
                "reg_lambda": self.params["reg_lambda"],
                "random_state": self.params["random_state"],
                "n_jobs": -1,
                "tree_method": "hist",
            }
            if is_classification:
                clf_params = {
                    **base_params,
                    "objective": "multi:softprob" if len(self.class_labels_) > 2 else "binary:logistic",
                    "eval_metric": "mlogloss" if len(self.class_labels_) > 2 else "logloss",
                }
                if len(self.class_labels_) > 2:
                    clf_params["num_class"] = len(self.class_labels_)
                self.model = XGBClassifier(**clf_params)
            else:
                self.model = XGBRegressor(
                    **base_params,
                    objective=self.params["objective"],
                    eval_metric="rmse",
                )
            fit_kwargs = {}
            if X_val is not None and y_val_fit is not None and len(X_val) > 0:
                fit_kwargs["eval_set"] = [(X_val, y_val_fit)]
                fit_kwargs["verbose"] = False
            try:
                self.model.fit(X_train, y_fit, **fit_kwargs)
            except TypeError:
                fit_kwargs.pop("verbose", None)
                self.model.fit(X_train, y_fit, **fit_kwargs)
        elif is_classification:
            warnings.warn("xgboost is unavailable; falling back to RandomForestClassifier")
            self.backend = "random_forest"

            cw = "balanced" if self.class_weight_balanced else None

            # stock-prophet style: GridSearchCV for optimal params
            if self.use_grid_search and len(X_train) >= 100:
                self._grid_search_rf(X_train, y_train, cw)
            else:
                self.model = RandomForestClassifier(
                    n_estimators=min(self.params["n_estimators"], 400),
                    max_depth=max(3, self.params["max_depth"]),
                    random_state=self.params["random_state"],
                    class_weight=cw,
                    n_jobs=-1,
                )
                self.model.fit(X_train, y_train)
        else:
            warnings.warn("xgboost is unavailable; falling back to RandomForestRegressor")
            self.backend = "random_forest"
            self.model = RandomForestRegressor(
                n_estimators=min(self.params["n_estimators"], 400),
                max_depth=max(3, self.params["max_depth"]),
                random_state=self.params["random_state"],
                n_jobs=-1,
            )
            self.model.fit(X_train, y_train)

        if hasattr(self.model, "feature_importances_"):
            self.feature_importances_ = self.model.feature_importances_

        if X_val is not None and y_val is not None:
            score_y = y_val_fit if is_classification and self.backend == "xgboost" else y_val
            self.val_score = self.model.score(X_val, score_y)

        return self.model

    def _grid_search_rf(self, X_train, y_train, class_weight):
        """GridSearchCV on RandomForest (stock-prophet style)."""
        param_grid = {
            "n_estimators": [50, 100, 300],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
        base_rf = RandomForestClassifier(
            random_state=self.params["random_state"],
            class_weight=class_weight,
            n_jobs=1,
        )
        gs = GridSearchCV(
            estimator=base_rf,
            param_grid=param_grid,
            cv=min(self.grid_search_cv, 5),
            scoring="accuracy",
            n_jobs=-1,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        self.model = gs.best_estimator_
        self.best_params_ = gs.best_params_

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        preds = self.model.predict(X)
        if self.class_labels_ is not None and self.backend == "xgboost":
            idx = np.asarray(preds, dtype=int)
            idx = np.clip(idx, 0, len(self.class_labels_) - 1)
            return self.class_labels_[idx]
        return preds

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
        joblib.dump(
            {
                "model": self.model,
                "class_labels": self.class_labels_,
                "params": self.params,
                "backend": self.backend,
            },
            path,
        )

    def load(self, path: str) -> "XGBoostModel":
        import joblib
        data = joblib.load(path)
        if isinstance(data, dict) and "model" in data:
            self.model = data["model"]
            self.class_labels_ = data.get("class_labels")
            self.params.update(data.get("params", {}))
            self.backend = data.get("backend", self.backend)
        else:
            self.model = data
            self.backend = "random_forest"
        return self

    @property
    def name(self) -> str:
        return "xgboost"
