# finrobot/data_source/crypto_utils.py

import requests
import pandas as pd
from datetime import datetime
import re


class CryptoUtils:
    """
    Helper utilities for fetching crypto data from the free CoinGecko API.
    No API key required.
    """

    BASE_URL = "https://api.coingecko.com/api/v3"

    @staticmethod
    def _clean_description(raw_desc: str, max_sentences: int = 3) -> str:
        """
        Remove basic HTML tags from the CoinGecko description and trim it
        to a shorter number of sentences so it's suitable for UI display.
        """
        if not raw_desc:
            return ""

        # Strip HTML tags
        text = re.sub(r"<.*?>", "", raw_desc)
        text = text.replace("\r", " ").replace("\n", " ").strip()

        if not text:
            return ""

        # Split into sentences in a simple way
        # (CoinGecko descriptions are usually well-formed.)
        parts = re.split(r"(?<=[.!?])\s+", text)
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            return ""

        trimmed = " ".join(parts[:max_sentences])
        # Ensure it ends with a period for neatness
        if trimmed and trimmed[-1] not in ".!?":
            trimmed += "."
        return trimmed

    @staticmethod
    def get_crypto_info(symbol: str) -> dict:
        """
        Fetch basic information for a given cryptocurrency.

        `symbol` here is the CoinGecko coin ID, e.g. 'bitcoin', 'ethereum', 'solana'.
        """
        url = f"{CryptoUtils.BASE_URL}/coins/{symbol.lower()}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        m = data["market_data"]

        raw_desc = data.get("description", {}).get("en", "") or ""
        short_desc = CryptoUtils._clean_description(raw_desc, max_sentences=3)

        return {
            "name": data["name"],
            "symbol": data["symbol"],
            "market_cap": m["market_cap"]["usd"],
            "current_price": m["current_price"]["usd"],
            "high_24h": m["high_24h"]["usd"],
            "low_24h": m["low_24h"]["usd"],
            "price_change_24h": m["price_change_percentage_24h"],
            "description": short_desc,  # short, cleaned description
        }

    @staticmethod
    def get_crypto_history(symbol: str, days: int = 365) -> pd.DataFrame:
        """
        Returns a DataFrame of daily prices for the last `days` days (max 365)
        for the given cryptocurrency.

        The DataFrame is indexed by datetime with a single 'price' column.
        """
        url = f"{CryptoUtils.BASE_URL}/coins/{symbol.lower()}/market_chart"
        params = {"vs_currency": "usd", "days": days}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        prices = data.get("prices", [])
        if not prices:
            return pd.DataFrame(columns=["price"])

        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["date"] = df["timestamp"].apply(lambda t: datetime.fromtimestamp(t / 1000))
        df.set_index("date", inplace=True)
        df.drop(columns=["timestamp"], inplace=True)

        return df


