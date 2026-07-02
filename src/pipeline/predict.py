import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from src.config import Config
from src.data.fetcher import fetch_historical, fetch_realtime, fetch_latest_price
from src.data.preprocessor import Preprocessor, create_sequences
from src.features.indicators import add_all_indicators
from src.features.signals import generate_rule_based_signals
from src.models.ensemble import EnsembleModel
from src.models.xgboost_model import XGBoostModel


def _load_prediction_models(model_base: Path) -> list:
    models = []
    lstm_path = Path(str(model_base) + "_lstm")
    if lstm_path.exists():
        try:
            from src.models.lstm_model import LSTMModel
            models.append(LSTMModel().load(str(lstm_path)))
        except Exception as exc:
            print(f"Skipping LSTM model: {exc}")

    xgb_path = Path(str(model_base) + "_xgboost")
    old_rf_path = Path(str(model_base) + "_random_forest")
    if xgb_path.exists():
        models.append(XGBoostModel().load(str(xgb_path)))
    elif old_rf_path.exists():
        models.append(XGBoostModel().load(str(old_rf_path)))

    if not models:
        raise FileNotFoundError(f"No saved models found for {model_base.name}")
    return models


def predict_pipeline(
    cfg: Config,
    ticker: str,
    realtime: bool = False,
) -> dict:
    if realtime:
        df = fetch_realtime(ticker, cfg.data.realtime_interval, cfg.data.realtime_period)
    else:
        df = fetch_historical(ticker, cfg.data.start_date, cfg.data.end_date)

    df = add_all_indicators(df, cfg.indicators)
    df = generate_rule_based_signals(df, cfg.indicators)

    preprocessor = Preprocessor.load(str(Path(cfg.model_dir) / f"{ticker}_preprocessor.pkl"))
    X = preprocessor.transform(df)
    if X.empty:
        raise ValueError("No complete feature rows available for prediction")

    X_seq, _ = create_sequences(X.values, np.zeros(len(X)), cfg.model.sequence_length)
    if len(X_seq) == 0:
        raise ValueError(
            f"Need more than {cfg.model.sequence_length} complete rows for prediction; got {len(X)}"
        )

    model_dir = Path(cfg.model_dir) / ticker
    models = _load_prediction_models(model_dir)
    weights = cfg.model.ensemble_weights if len(models) == len(cfg.model.ensemble_weights) else None
    ensemble = EnsembleModel(models=models, weights=weights)

    mean_pred, std_pred = ensemble.predict_with_confidence(X_seq)
    latest_price = fetch_latest_price(ticker) or df["Close"].iloc[-1]
    signal_strength = df["Signal_Strength"].iloc[-1] if "Signal_Strength" in df else 0
    predicted = float(mean_pred[-1])
    current = float(latest_price)
    pred_std = float(std_pred[-1]) if len(std_pred) > 0 else 0.0
    if len(models) == 1 or pred_std == 0:
        confidence = min(0.95, max(0.05, 0.55 + abs(float(signal_strength)) * 0.4))
    else:
        confidence = 1 - pred_std / max(abs(predicted), 0.01)
        confidence = min(0.99, max(0.0, confidence))

    return {
        "ticker": ticker,
        "current_price": round(current, 2),
        "predicted_price": round(predicted, 2),
        "confidence": round(confidence, 3),
        "signal_strength": round(signal_strength, 3),
        "direction": "UP" if predicted > df["Close"].iloc[-1] else "DOWN" if predicted < df["Close"].iloc[-1] else "FLAT",
        "is_realtime": realtime,
    }
