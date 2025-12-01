# metals_utils.py

import yfinance as yf
import pandas as pd
from typing import Dict, Any, Optional


class MetalsUtils:
    """
    Helper utilities for fetching metal / commodity data using Yahoo Finance
    through yfinance. No extra API key needed.
    """

    # Map friendly names to Yahoo Finance symbols
    METAL_SYMBOLS: Dict[str, str] = {
        "gold": "GC=F",         # Gold futures
        "silver": "SI=F",       # Silver futures
        "platinum": "PL=F",
        "palladium": "PA=F",
        "copper": "HG=F",
        "rare earth": "REMX",   # Rare earth ETF
        "rare earths": "REMX",
        "lithium": "LIT",       # Lithium ETF
    }

    @staticmethod
    def resolve_symbol(metal: str) -> str:
        """
        Convert a user input like 'gold' or 'silver' into a Yahoo Finance symbol.
        If we don't know it, assume the user entered a valid symbol already.
        """
        if not metal:
            return ""
        key = metal.strip().lower()
        if not key:
            return ""
        return MetalsUtils.METAL_SYMBOLS.get(key, metal.upper())

    @staticmethod
    def get_metal_info(metal: str) -> Dict[str, Any]:
        """
        Fetch basic information about a metal / commodity or related ETF.
        `metal` can be 'gold', 'silver', etc., or a direct symbol like 'GC=F'.
        """
        symbol = MetalsUtils.resolve_symbol(metal)
        if not symbol:
            raise ValueError("Empty metal or symbol input.")

        ticker = yf.Ticker(symbol)
        try:
            info: Dict[str, Any] = ticker.info or {}
        except Exception as e:
            print(f"[MetalsUtils.get_metal_info] Error fetching info for {symbol}: {e}")
            info = {}

        name = (
            info.get("longName")
            or info.get("shortName")
            or metal.title()
        )
        current_price = info.get("regularMarketPrice")
        prev_close = info.get("regularMarketPreviousClose")
        day_high = info.get("regularMarketDayHigh")
        day_low = info.get("regularMarketDayLow")

        change_pct: Optional[float] = None
        if current_price is not None and prev_close not in (None, 0):
            try:
                change_pct = (current_price - prev_close) / prev_close * 100.0
            except Exception:
                change_pct = None

        return {
            "input": metal,
            "name": name,
            "symbol": symbol,
            "currency": info.get("currency", "USD"),
            "current_price": current_price,
            "previous_close": prev_close,
            "day_high": day_high,
            "day_low": day_low,
            "change_pct": change_pct,
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "volume": info.get("regularMarketVolume"),
        }

    @staticmethod
    def get_metal_history(metal: str, period: str = "1y") -> pd.DataFrame:
        """
        Returns a DataFrame of daily closing prices for the last `period`
        (e.g. '1y', '6mo', etc.) for the given metal/symbol.
        """
        symbol = MetalsUtils.resolve_symbol(metal)
        if not symbol:
            return pd.DataFrame(columns=["price"])

        data = yf.download(symbol, period=period, interval="1d", progress=False)
        if data is None or data.empty or "Close" not in data.columns:
            return pd.DataFrame(columns=["price"])

        df = data[["Close"]].rename(columns={"Close": "price"}).copy()
        df.index.name = "date"
        return df
