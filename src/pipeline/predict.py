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
from src.models.lstm_model import LSTMModel
from src.models.xgboost_model import XGBoostModel


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
    X, _ = preprocessor.fit_transform(df, cfg.data.features, cfg.data.target_column)

    X_seq, _ = create_sequences(X.values, np.zeros(len(X)), cfg.model.sequence_length)

    model_dir = Path(cfg.model_dir) / ticker
    lstm = LSTMModel().load(str(model_dir) + "_lstm")
    xgb = XGBoostModel().load(str(model_dir) + "_xgboost")
    ensemble = EnsembleModel(models=[lstm, xgb], weights=cfg.model.ensemble_weights)

    mean_pred, std_pred = ensemble.predict_with_confidence(X_seq)
    latest_price = fetch_latest_price(ticker) or df["Close"].iloc[-1]
    signal_strength = df["Signal_Strength"].iloc[-1] if "Signal_Strength" in df else 0

    return {
        "ticker": ticker,
        "current_price": round(latest_price, 2),
        "predicted_price": round(mean_pred[-1], 2) if len(mean_pred) > 0 else 0,
        "confidence": round(1 - std_pred[-1] / max(abs(mean_pred[-1]), 0.01), 3) if len(std_pred) > 0 else 0,
        "signal_strength": round(signal_strength, 3),
        "direction": "UP" if mean_pred[-1] > df["Close"].iloc[-1] else "DOWN" if mean_pred[-1] < df["Close"].iloc[-1] else "FLAT",
        "is_realtime": realtime,
    }
