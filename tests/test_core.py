import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.analysis.trading_strategy import backtest_strategy
from src.config import Config
from src.data.preprocessor import Preprocessor, create_sequences
from src.features.indicators import add_all_indicators
from src.features.signals import create_target_labels, generate_rule_based_signals
from src.models.xgboost_model import XGBoostModel
from src.pipeline.predict import predict_pipeline
from src.pipeline.train import train_pipeline


def synthetic_ohlcv(rows=320):
    idx = pd.date_range("2024-01-01", periods=rows, freq="D")
    x = np.arange(rows, dtype=float)
    close = 100 + x * 0.08 + np.sin(x / 7) * 2
    open_ = close + np.sin(x / 5) * 0.3
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    volume = 1_000_000 + (np.cos(x / 11) * 50_000).astype(int)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=idx,
    )


class CoreBehaviorTests(unittest.TestCase):
    def test_indicators_produce_configured_features(self):
        cfg = Config()
        df = add_all_indicators(synthetic_ohlcv(), cfg.indicators)
        missing = [col for col in cfg.data.features if col not in df.columns]
        self.assertEqual(missing, [])
        last = df[cfg.data.features].tail(1).replace([np.inf, -np.inf], np.nan)
        self.assertFalse(last.isna().any(axis=None))

    def test_preprocessor_fits_train_only_and_aligns_missing_realtime_columns(self):
        cfg = Config()
        df = add_all_indicators(synthetic_ohlcv(), cfg.indicators).dropna()
        train = df.iloc[:80]
        future = df.iloc[80:100].drop(columns=["SMA_200"])

        prep = Preprocessor().fit(train, cfg.data.features)
        X_train = prep.transform(train)
        X_future = prep.transform(future, allow_missing=True, dropna=False)

        self.assertLess(abs(float(X_train.mean().mean())), 1e-9)
        self.assertEqual(list(X_future.columns), cfg.data.features)
        self.assertEqual(len(X_future), len(future))

    def test_target_labels_and_sequences(self):
        df = create_target_labels(synthetic_ohlcv(40), horizon=3, threshold_pct=0.001)
        self.assertIn("Target_Return", df.columns)
        self.assertIn("Target_Class", df.columns)
        self.assertTrue(set(df["Target_Class"].dropna().unique()).issubset({-1, 0, 1}))

        X_seq, y_seq = create_sequences(np.ones((10, 3)), np.arange(10), seq_length=4)
        self.assertEqual(X_seq.shape, (6, 4, 3))
        self.assertEqual(y_seq.tolist(), [4, 5, 6, 7, 8, 9])

    def test_xgboost_model_regression_and_classification_round_trip(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(80, 4))
        y_reg = X[:, 0] * 0.5 - X[:, 1] * 0.25
        y_cls = np.where(y_reg > 0.2, 1, np.where(y_reg < -0.2, -1, 0))

        with tempfile.TemporaryDirectory() as tmp:
            reg_path = Path(tmp) / "reg.joblib"
            reg = XGBoostModel(n_estimators=5, max_depth=2).train(X[:60], y_reg[:60], X[60:], y_reg[60:])
            self.assertIsNotNone(reg)
            reg_model = XGBoostModel(n_estimators=5, max_depth=2)
            reg_model.train(X[:60], y_reg[:60], X[60:], y_reg[60:])
            reg_model.save(str(reg_path))
            loaded = XGBoostModel().load(str(reg_path))
            self.assertEqual(loaded.predict(X[:3]).shape, (3,))

            cls_path = Path(tmp) / "cls.joblib"
            cls_model = XGBoostModel(n_estimators=5, max_depth=2, num_class=3)
            cls_model.train(X[:60], y_cls[:60], X[60:], y_cls[60:])
            cls_model.save(str(cls_path))
            cls_loaded = XGBoostModel().load(str(cls_path))
            preds = cls_loaded.predict(X[:10])
            self.assertTrue(set(preds.tolist()).issubset({-1, 0, 1}))
            self.assertEqual(cls_loaded.predict_proba(X[:2]).shape[0], 2)

    def test_backtest_uses_negative_sell_threshold(self):
        df = add_all_indicators(synthetic_ohlcv(260), Config().indicators)
        df = generate_rule_based_signals(df, Config().indicators).dropna()
        result = backtest_strategy(df, signal_threshold_buy=0.2, signal_threshold_sell=-0.2)
        self.assertNotIn("error", result)
        self.assertGreater(result["final_value"], 0)
        self.assertIsInstance(result["trades"], list)

    def test_train_and_predict_pipeline_offline(self):
        cfg = Config()
        cfg.model.sequence_length = 5
        cfg.model.xgb_n_estimators = 5
        cfg.model.xgb_max_depth = 2
        cfg.model.lstm_epochs = 1
        df = synthetic_ohlcv(340)

        with tempfile.TemporaryDirectory() as tmp:
            cfg.data_dir = str(Path(tmp) / "data")
            cfg.model_dir = str(Path(tmp) / "models")
            with patch("src.pipeline.train.fetch_historical", return_value=df):
                ensemble, prep, metrics = train_pipeline(cfg, "TEST")

            self.assertTrue((Path(cfg.model_dir) / "TEST_xgboost").exists())
            self.assertTrue((Path(cfg.model_dir) / "TEST_preprocessor.pkl").exists())
            self.assertIn("RMSE", metrics)

            with patch("src.pipeline.predict.fetch_historical", return_value=df), patch(
                "src.pipeline.predict.fetch_latest_price", return_value=float(df["Close"].iloc[-1])
            ):
                pred = predict_pipeline(cfg, "TEST", realtime=False)

            self.assertEqual(pred["ticker"], "TEST")
            self.assertIn(pred["direction"], {"UP", "DOWN", "FLAT"})
            self.assertGreater(pred["current_price"], 0)


if __name__ == "__main__":
    unittest.main()
