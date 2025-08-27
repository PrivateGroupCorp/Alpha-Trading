import pandas as pd


def _tf_to_rule(tf: str) -> str:
    unit = tf[0]
    num = int(tf[1:])
    mapping = {"S": "S", "M": "T", "H": "H", "D": "D"}
    return f"{num}{mapping[unit]}"


def resample_ohlc(df: pd.DataFrame, tf_to: str) -> pd.DataFrame:
    rule = _tf_to_rule(tf_to)
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in df.columns:
        agg["volume"] = "sum"
    res = df.resample(rule, label="left", closed="left", origin="start_day").agg(agg)
    res.index = res.index + pd.Timedelta(rule)
    res = res.dropna(subset=[c for c in ["open", "high", "low", "close"] if c in res.columns])
    return res
