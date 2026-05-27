import yfinance as yf
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime


def fetch_historical(
    ticker: str,
    start: str,
    end: str,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker} from {start} to {end}")
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.index = pd.to_datetime(df.index)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path)
    return df


def fetch_realtime(ticker: str, interval: str = "1m", period: str = "5d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No real-time data for {ticker}")
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.index = pd.to_datetime(df.index)
    return df


def fetch_latest_price(ticker: str) -> float:
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1d", interval="1m")
    if hist.empty:
        return tk.fast_info.get("lastPrice", None)
    return hist["Close"].iloc[-1]


def fetch_recent(ticker: str, interval: str = "1m") -> pd.DataFrame:
    df = yf.download(ticker, period="1d", interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        return df
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.index = pd.to_datetime(df.index)
    return df


def _fmt_earnings(val):
    if val is None:
        return "N/A"
    if isinstance(val, list):
        if not val:
            return "N/A"
        val = val[0]
    try:
        return datetime.utcfromtimestamp(val).strftime("%b %d, %Y")
    except Exception:
        return str(val) if val else "N/A"


def lookup_ticker_info(ticker: str) -> dict:
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        if not info or "quoteType" not in info:
            return {"name": ticker}
        return {
            "name": info.get("longName") or info.get("shortName") or ticker,
            "type": info.get("quoteType", ""),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap"),
            "prev_close": info.get("regularMarketPreviousClose") or info.get("previousClose"),
            "open": info.get("regularMarketOpen") or info.get("open"),
            "day_low": info.get("regularMarketDayLow") or info.get("dayLow"),
            "day_high": info.get("regularMarketDayHigh") or info.get("dayHigh"),
            "volume": info.get("regularMarketVolume") or info.get("volume"),
            "avg_volume": info.get("averageVolume"),
            "pe_ratio": info.get("trailingPE"),
            "fwd_pe": info.get("forwardPE"),
            "eps": info.get("trailingEps"),
            "dividend_yield": info.get("dividendYield") or info.get("yield"),
            "beta": info.get("beta"),
            "peg": info.get("pegRatio"),
            "price_book": info.get("priceToBook"),
            "short_ratio": info.get("shortRatio"),
            "earnings_date": _fmt_earnings(info.get("earningsDate")),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "expense_ratio": info.get("annualReportExpenseRatio"),
            "net_assets": info.get("totalAssets"),
            "ytd_return": info.get("ytdReturn"),
            "etf_category": info.get("category"),
            "holdings": info.get("heldPercentInstitutions"),
            "inception": info.get("fundInceptionDate"),
            "open_interest": info.get("openInterest"),
            "contract_currency": info.get("currency"),
            "exchange": info.get("exchange"),
        }
    except Exception:
        return {"name": ticker}


def load_local(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)
