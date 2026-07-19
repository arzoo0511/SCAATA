"""7-feature technical pipeline, ported unchanged from the v1 notebook."""
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds returns, ma_10, ma_50, volatility, momentum, rsi, volume_ma_30 per ticker."""
    data = df.copy()

    data["returns"] = data.groupby("Ticker")["Close"].pct_change()
    data["ma_10"] = data.groupby("Ticker")["Close"].transform(lambda x: x.rolling(10).mean())
    data["ma_50"] = data.groupby("Ticker")["Close"].transform(lambda x: x.rolling(50).mean())
    data["volatility"] = data.groupby("Ticker")["returns"].transform(lambda x: x.rolling(10).std())
    data["momentum"] = data.groupby("Ticker")["Close"].transform(lambda x: x / x.shift(10) - 1)
    data["rsi"] = data.groupby("Ticker")["Close"].transform(_rsi)
    data["volume_ma_30"] = data.groupby("Ticker")["Volume"].transform(lambda x: x.rolling(30).mean())

    return data.dropna()
