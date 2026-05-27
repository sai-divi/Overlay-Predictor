import numpy as np
from pathlib import Path
from typing import Optional

from src.config import Config
from src.data.fetcher import fetch_historical
from src.data.preprocessor import Preprocessor, create_sequences, train_val_test_split
from src.features.indicators import add_all_indicators
from src.features.signals import create_target_labels
from src.models.lstm_model import LSTMModel, lstm_for_classification
from src.models.xgboost_model import XGBoostModel
from src.models.ensemble import EnsembleModel
from src.utils.metrics import evaluate
from src.utils.helpers import set_seed


def train_pipeline(
    cfg: Config,
    ticker: str,
    additional_features: Optional[dict] = None,
    mode: str = "regression",
):
    set_seed(cfg.seed)

    raw_path = f"{cfg.data_dir}/raw/{ticker}.csv"
    df = fetch_historical(ticker, cfg.data.start_date, cfg.data.end_date, save_path=raw_path)
    df = add_all_indicators(df, cfg.indicators)

    if mode == "classification":
        df = create_target_labels(df, horizon=cfg.data.prediction_horizon, threshold_pct=0.005)
        target_col = "Target_Class"
    else:
        target_col = cfg.data.target_column

    preprocessor = Preprocessor()
    X, y = preprocessor.fit_transform(df, cfg.data.features, target_col)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X.values, y.values, cfg.data.train_split, cfg.data.val_split
    )

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, cfg.model.sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, cfg.model.sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, cfg.model.sequence_length)

    if mode == "classification":
        lstm = lstm_for_classification(
            units=cfg.model.lstm_units,
            dropout=cfg.model.lstm_dropout,
            learning_rate=cfg.model.lstm_learning_rate,
            epochs=cfg.model.lstm_epochs,
            batch_size=cfg.model.lstm_batch_size,
            sequence_length=cfg.model.sequence_length,
        )
        xgb = XGBoostModel(
            n_estimators=cfg.model.xgb_n_estimators,
            max_depth=cfg.model.xgb_max_depth,
            learning_rate=cfg.model.xgb_learning_rate,
            subsample=cfg.model.xgb_subsample,
            colsample_bytree=cfg.model.xgb_colsample_bytree,
            objective="multi:softprob",
            num_class=3,
        )
    else:
        lstm = LSTMModel(
            units=cfg.model.lstm_units,
            dropout=cfg.model.lstm_dropout,
            learning_rate=cfg.model.lstm_learning_rate,
            epochs=cfg.model.lstm_epochs,
            batch_size=cfg.model.lstm_batch_size,
            sequence_length=cfg.model.sequence_length,
        )
        xgb = XGBoostModel(
            n_estimators=cfg.model.xgb_n_estimators,
            max_depth=cfg.model.xgb_max_depth,
            learning_rate=cfg.model.xgb_learning_rate,
            subsample=cfg.model.xgb_subsample,
            colsample_bytree=cfg.model.xgb_colsample_bytree,
        )

    ensemble = EnsembleModel(models=[lstm, xgb], weights=cfg.model.ensemble_weights)
    print(f"Training {ensemble.name} on {ticker} (mode={mode})...")
    ensemble.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq)

    preds = ensemble.predict(X_test_seq)
    if mode == "classification":
        preds = np.round(preds).clip(-1, 1)

    metrics = evaluate(y_test_seq, preds)
    print(f"Test metrics for {ticker}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    ensemble.save(str(model_dir / ticker))
    preprocessor.save(str(model_dir / f"{ticker}_preprocessor.pkl"))

    print(f"Model saved to {model_dir / ticker}")
    return ensemble, preprocessor, metrics
