import numpy as np
from pathlib import Path
from typing import Optional

from src.config import Config
from src.data.fetcher import fetch_historical
from src.data.preprocessor import Preprocessor, create_sequences
from src.features.indicators import add_all_indicators
from src.features.signals import create_target_labels
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
    lstm_error = None
    LSTMModel = None
    lstm_for_classification = None
    try:
        from src.models.lstm_model import LSTMModel, lstm_for_classification
    except Exception as exc:
        lstm_error = exc

    raw_path = f"{cfg.data_dir}/raw/{ticker}.csv"
    df = fetch_historical(ticker, cfg.data.start_date, cfg.data.end_date, save_path=raw_path)
    df = add_all_indicators(df, cfg.indicators)

    if mode == "binary_5d":
        from src.features.signals import create_target_labels_5d
        df = create_target_labels_5d(df, future_days=cfg.model.future_target_days)
        target_col = "Target_Binary"
    elif mode == "classification":
        df = create_target_labels(df, horizon=cfg.data.prediction_horizon, threshold_pct=0.005)
        target_col = "Target_Class"
    else:
        target_col = cfg.data.target_column

    features = [col for col in cfg.data.features if col in df.columns]
    missing_features = [col for col in cfg.data.features if col not in df.columns]
    if missing_features:
        print(f"Skipping unavailable features: {', '.join(missing_features)}")
    if not features:
        raise ValueError("No configured feature columns are available after indicator generation")

    clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features + [target_col])
    if len(clean) <= cfg.model.sequence_length + 5:
        raise ValueError(
            f"Not enough clean rows ({len(clean)}) for sequence length {cfg.model.sequence_length}"
        )

    n = len(clean)
    train_end = int(n * cfg.data.train_split)
    val_end = train_end + int(n * cfg.data.val_split)
    train_df = clean.iloc[:train_end]
    val_df = clean.iloc[train_end:val_end]
    test_df = clean.iloc[val_end:]
    if len(train_df) <= cfg.model.sequence_length:
        raise ValueError("Training split is too small for the configured sequence length")

    preprocessor = Preprocessor()
    preprocessor.fit(train_df, features)
    X_train_df, y_train_s = preprocessor.transform_with_target(train_df, target_col)
    X_val_df, y_val_s = preprocessor.transform_with_target(val_df, target_col)
    X_test_df, y_test_s = preprocessor.transform_with_target(test_df, target_col)

    X_train_seq, y_train_seq = create_sequences(X_train_df.values, y_train_s.values, cfg.model.sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val_df.values, y_val_s.values, cfg.model.sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_df.values, y_test_s.values, cfg.model.sequence_length)
    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        raise ValueError("Not enough rows to create train/test sequences")

    models = []
    weights = []
    if mode == "classification":
        xgb = XGBoostModel(
            n_estimators=cfg.model.xgb_n_estimators,
            max_depth=cfg.model.xgb_max_depth,
            learning_rate=cfg.model.xgb_learning_rate,
            subsample=cfg.model.xgb_subsample,
            colsample_bytree=cfg.model.xgb_colsample_bytree,
            objective="multi:softprob",
            num_class=3,
            use_grid_search=cfg.model.use_grid_search,
            grid_search_cv=cfg.model.grid_search_cv,
            class_weight_balanced=cfg.model.class_weight_balanced,
        )
        models.append(xgb)
        weights.append(1.0)
    else:
        if LSTMModel is not None and len(X_val_seq) > 0:
            lstm = LSTMModel(
                units=cfg.model.lstm_units,
                dropout=cfg.model.lstm_dropout,
                learning_rate=cfg.model.lstm_learning_rate,
                epochs=cfg.model.lstm_epochs,
                batch_size=cfg.model.lstm_batch_size,
                sequence_length=cfg.model.sequence_length,
            )
            models.append(lstm)
            weights.append(cfg.model.ensemble_weights[0])
        elif lstm_error is not None:
            print(f"TensorFlow/LSTM unavailable; training XGBoost only ({lstm_error})")

        xgb = XGBoostModel(
            n_estimators=cfg.model.xgb_n_estimators,
            max_depth=cfg.model.xgb_max_depth,
            learning_rate=cfg.model.xgb_learning_rate,
            subsample=cfg.model.xgb_subsample,
            colsample_bytree=cfg.model.xgb_colsample_bytree,
        )
        models.append(xgb)
        weights.append(cfg.model.ensemble_weights[-1] if len(cfg.model.ensemble_weights) > 1 else 1.0)

    ensemble = EnsembleModel(models=models, weights=weights)
    print(f"Training {ensemble.name} on {ticker} (mode={mode})...")
    ensemble.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq)

    preds = ensemble.predict(X_test_seq)
    if mode == "classification":
        preds = np.asarray(preds).astype(int)
        metrics = {"Accuracy": float(np.mean(preds == y_test_seq))}
    else:
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
