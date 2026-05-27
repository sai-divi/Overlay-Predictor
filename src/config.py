from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class IndicatorConfig:
    sma_windows: List[int] = field(default_factory=lambda: [10, 20, 50, 200])
    ema_windows: List[int] = field(default_factory=lambda: [10, 20, 50])
    rsi_period: int = 14
    rsi_overbought: float = 70
    rsi_oversold: float = 30
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    roc_period: int = 10
    williams_period: int = 14


@dataclass
class StrategyConfig:
    initial_capital: float = 10000.0
    position_size: float = 0.95
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.15
    trailing_stop_pct: Optional[float] = 0.03
    max_positions: int = 1
    commission_pct: float = 0.001
    slippage_pct: float = 0.001
    signal_threshold_buy: float = 0.6
    signal_threshold_sell: float = 0.6


@dataclass
class ModelConfig:
    name: str = "ensemble"
    lstm_units: List[int] = field(default_factory=lambda: [128, 64, 32])
    lstm_dropout: float = 0.3
    lstm_learning_rate: float = 0.001
    lstm_epochs: int = 100
    lstm_batch_size: int = 64
    xgb_n_estimators: int = 500
    xgb_max_depth: int = 7
    xgb_learning_rate: float = 0.01
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    sequence_length: int = 60
    ensemble_weights: List[float] = field(default_factory=lambda: [0.6, 0.4])


@dataclass
class DataConfig:
    tickers: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL"])
    start_date: str = "2015-01-01"
    end_date: str = "2025-01-01"
    target_column: str = "Close"
    features: List[str] = field(default_factory=lambda: [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_10", "SMA_20", "SMA_50", "SMA_200",
        "EMA_10", "EMA_20", "EMA_50",
        "RSI_14", "RSI_14_ma", "MACD", "MACD_signal", "MACD_hist",
        "BB_upper", "BB_lower", "BB_middle", "BB_width", "BB_position",
        "ATR_14", "ATR_pct", "OBV", "OBV_SMA", "OBV_ratio",
        "ROC_10", "ROC_10_ma",
        "Williams_%R", "Stoch_%K", "Stoch_%D",
        "Price_Change", "Price_Change_5", "Price_Change_21",
        "High_Low_Ratio", "Close_Open_Ratio",
        "Volume_Change", "Volume_Change_5",
        "Regime_SMA", "Regime_Volatility",
    ])
    train_split: float = 0.7
    val_split: float = 0.15
    normalize: bool = True
    prediction_horizon: int = 1
    realtime_interval: str = "1m"
    realtime_period: str = "5d"


@dataclass
class AnalysisConfig:
    enable_fundamentals: bool = False
    enable_sentiment: bool = False
    monte_carlo_sims: int = 1000
    monte_carlo_days: int = 252
    confidence_level: float = 0.95
    risk_free_rate: float = 0.05


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    data_dir: str = "data"
    model_dir: str = "models/saved"
    report_dir: str = "reports"
    seed: int = 42
