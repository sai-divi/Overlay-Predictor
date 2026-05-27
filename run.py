import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.pipeline.train import train_pipeline
from src.pipeline.backtest import backtest


def main():
    parser = argparse.ArgumentParser(description="Stock ML Prediction Pipeline")
    parser.add_argument("mode", choices=["train", "backtest"], help="Pipeline mode")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2025-01-01", help="End date")
    parser.add_argument("--sequence-length", type=int, default=60, help="LSTM sequence length")
    parser.add_argument("--epochs", type=int, default=100, help="LSTM training epochs")
    args = parser.parse_args()

    cfg = Config()
    cfg.data.start_date = args.start
    cfg.data.end_date = args.end
    cfg.data.tickers = [args.ticker]
    cfg.model.sequence_length = args.sequence_length
    cfg.model.lstm_epochs = args.epochs

    if args.mode == "train":
        train_pipeline(cfg, args.ticker)
    elif args.mode == "backtest":
        backtest(cfg, args.ticker, plot=True)


if __name__ == "__main__":
    main()
