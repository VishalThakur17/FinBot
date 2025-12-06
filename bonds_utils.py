# bonds_utils.py

import yfinance as yf
import pandas as pd
from typing import Dict, Any, Optional


class BondsUtils:
    """
    Helper utilities for fetching US bond yields and bond ETFs using Yahoo Finance.

    - Supports Treasury yields via ^TNX, ^TYX, ^FVX, etc.
    - Supports popular Treasury bond ETFs like TLT, SHY, IEF, etc.
    - No API key needed (yfinance wraps Yahoo Finance).
    """

    # Map friendly user input -> Yahoo symbols
    BOND_SYMBOLS: Dict[str, str] = {
        # Yields (Treasury rates)
        "10-year": "^TNX",
        "10 year": "^TNX",
        "10y": "^TNX",
        "10yr": "^TNX",

        "30-year": "^TYX",
        "30 year": "^TYX",
        "30y": "^TYX",
        "30yr": "^TYX",

        "5-year": "^FVX",
        "5 year": "^FVX",
        "5y": "^FVX",
        "5yr": "^FVX",

        # Some ETF shortcuts
        "short-term": "SHY",
        "short term": "SHY",
        "short": "SHY",
        "1-3 year": "SHY",
        "1–3 year": "SHY",

        "intermediate": "IEF",
        "7-10 year": "IEF",
        "7–10 year": "IEF",

        "long-term": "TLT",
        "long term": "TLT",
        "long": "TLT",
        "20+ year": "TLT",
        "20-plus year": "TLT",
    }

    @staticmethod
    def resolve_symbol(bond: str) -> str:
        """
        Convert user input like '10-year', 'long-term', or 'tlt' into a Yahoo Finance symbol.

        If we don't know it, assume the user already entered a valid symbol and just normalize it.
        """
        if not bond:
            return ""
        key = bond.strip().lower()
        if not key:
            return ""
        return BondsUtils.BOND_SYMBOLS.get(key, bond.upper())

    @staticmethod
    def _is_yield_symbol(symbol: str) -> bool:
        """
        Simple check: Yahoo uses ^TNX, ^TYX, ^FVX, etc. for yields.
        """
        return symbol.startswith("^")

    @staticmethod
    def get_bond_info(bond: str) -> Dict[str, Any]:
        """
        Fetch basic information about a US bond yield or bond ETF.

        `bond` can be:
          - Friendly name: '10-year', 'long-term', 'short-term'
          - Direct symbol: '^TNX', 'TLT', 'SHY', etc.
        """
        symbol = BondsUtils.resolve_symbol(bond)
        if not symbol:
            raise ValueError("Empty bond input.")

        ticker = yf.Ticker(symbol)
        try:
            info: Dict[str, Any] = ticker.info or {}
        except Exception as e:
            print(f"[BondsUtils.get_bond_info] Error fetching info for {symbol}: {e}")
            info = {}

        is_yield = BondsUtils._is_yield_symbol(symbol)

        name = (
            info.get("longName")
            or info.get("shortName")
            or bond.title()
        )

        raw_current = info.get("regularMarketPrice")
        raw_prev = info.get("regularMarketPreviousClose")
        raw_high = info.get("regularMarketDayHigh")
        raw_low = info.get("regularMarketDayLow")

        # For yield symbols, Yahoo reports ~10x the yield, e.g. 42.35 ~ 4.235%
        def to_yield(x: Optional[float]) -> Optional[float]:
            if x is None:
                return None
            try:
                return float(x) / 10.0
            except Exception:
                return None

        if is_yield:
            current_value = to_yield(raw_current)
            previous_value = to_yield(raw_prev)
            day_high = to_yield(raw_high)
            day_low = to_yield(raw_low)
            unit = "%"   # yields are in percent
        else:
            current_value = raw_current
            previous_value = raw_prev
            day_high = raw_high
            day_low = raw_low
            unit = info.get("currency", "USD") or "USD"

        change_pct: Optional[float] = None
        if current_value is not None and previous_value not in (None, 0):
            try:
                change_pct = (current_value - previous_value) / previous_value * 100.0
            except Exception:
                change_pct = None

        if is_yield:
            wk_low = to_yield(info.get("fiftyTwoWeekLow"))
            wk_high = to_yield(info.get("fiftyTwoWeekHigh"))
        else:
            wk_low = info.get("fiftyTwoWeekLow")
            wk_high = info.get("fiftyTwoWeekHigh")

        volume = info.get("regularMarketVolume")

        return {
            "input": bond,
            "name": name,
            "symbol": symbol,
            "display_type": "yield" if is_yield else "etf",
            "unit": unit,  # "%" for yields, currency for ETFs
            "current_value": current_value,
            "previous_value": previous_value,
            "day_high": day_high,
            "day_low": day_low,
            "change_pct": change_pct,
            "fifty_two_week_low": wk_low,
            "fifty_two_week_high": wk_high,
            "volume": volume,
        }

    @staticmethod
    def get_bond_history(bond: str, period: str = "1y") -> pd.DataFrame:
        """
        Returns a DataFrame of daily values for the last `period` (default '1y').

        - For yield symbols (^TNX, ^TYX, ^FVX), this is yield (%) over time.
        - For ETFs (TLT, SHY, IEF, etc.), this is closing price.
        """
        symbol = BondsUtils.resolve_symbol(bond)
        if not symbol:
            return pd.DataFrame(columns=["value"])

        is_yield = BondsUtils._is_yield_symbol(symbol)

        data = yf.download(symbol, period=period, interval="1d", progress=False)
        if data is None or data.empty or "Close" not in data.columns:
            return pd.DataFrame(columns=["value"])

        df = data[["Close"]].copy()
        if is_yield:
            df["value"] = df["Close"] / 10.0  # convert raw quote to 10-based yield
        else:
            df["value"] = df["Close"]

        df = df[["value"]]
        df.index.name = "date"
        return df

