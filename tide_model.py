import json
import os
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def load_config():
    with open("config.json", "r") as f:
        return json.load(f)


def get_data(symbol):
    data_15m = yf.download(symbol, period="7d", interval="15m", progress=False, auto_adjust=True)
    data_1h = yf.download(symbol, period="60d", interval="1h", progress=False, auto_adjust=True)

    data_15m = flatten_columns(data_15m).dropna()
    data_1h = flatten_columns(data_1h).dropna()

    return data_15m, data_1h


def calc_regime(data_1h, fast=50, slow=200):
    data_1h["MA50"] = data_1h["Close"].rolling(fast).mean()
    data_1h["MA200"] = data_1h["Close"].rolling(slow).mean()

    latest = data_1h.iloc[-1]

    if latest["MA50"] < latest["MA200"]:
        return "downtrend"
    return "not_downtrend"


def calc_sweeps(data_15m, cfg):
    df = data_15m.copy()

    lookback = int(cfg["rolling_low_hours"] * 4)
    cluster_window = timedelta(hours=cfg["cluster_hours"])

    df["prior_rolling_low"] = df["Low"].rolling(lookback).min().shift(1)
    df["prior_avg_volume"] = df["Volume"].rolling(lookback).mean().shift(1)

    candle_range = df["High"] - df["Low"]
    lower_wick = np.minimum(df["Open"], df["Close"]) - df["Low"]
    df["lower_wick_ratio"] = lower_wick / candle_range.replace(0, np.nan)

    df["bullish_sweep"] = (
        (df["Low"] < df["prior_rolling_low"]) &
        (df["Close"] > df["prior_rolling_low"]) &
        (df["lower_wick_ratio"] > cfg["wick_ratio"]) &
        (df["Volume"] > cfg["volume_multiplier"] * df["prior_avg_volume"])
    )

    now = df.index[-1]
    recent_start = now - cluster_window

    recent_sweeps = df[(df.index >= recent_start) & (df["bullish_sweep"])]

    return df, recent_sweeps


def classify_status(regime, recent_sweeps, latest_time, cfg):
    if regime != "downtrend":
        return "NO SIGNAL"

    if len(recent_sweeps) == 0:
        return "NO SIGNAL"

    signal_time = recent_sweeps.index[-1]
    age_hours = (latest_time - signal_time).total_seconds() / 3600

    if age_hours <= cfg["fresh_hours"]:
        return "FRESH SIGNAL"

    if age_hours <= cfg["cluster_hours"]:
        return "ACTIVE BUT LATE"

    return "EXPIRED SIGNAL"


def main():
    cfg = load_config()
    symbol = cfg["symbol"]

    data_15m, data_1h = get_data(symbol)

    regime = calc_regime(
        data_1h,
        fast=cfg["trend_ma_fast"],
        slow=cfg["trend_ma_slow"]
    )

    sweep_df, recent_sweeps = calc_sweeps(data_15m, cfg)

    latest_time = data_15m.index[-1]
    latest_price = float(data_15m["Close"].iloc[-1])
    status = classify_status(regime, recent_sweeps, latest_time, cfg)

    result = {
        "latest_check_time_utc": datetime.now(timezone.utc).isoformat(),
        "latest_candle_time_utc": str(latest_time),
        "symbol": symbol,
        "latest_price": round(latest_price, 2),
        "regime_1h": regime,
        "recent_signal_count_18h": int(len(recent_sweeps)),
        "status": status,
        "note": ""
    }

    if len(recent_sweeps) > 0:
        signal = recent_sweeps.iloc[-1]
        signal_time = recent_sweeps.index[-1]
        planned_exit = signal_time + timedelta(hours=cfg["cluster_hours"])

        result.update({
            "signal_time_utc": str(signal_time),
            "entry_price": round(float(signal["Close"]), 2),
            "planned_exit_time_utc": str(planned_exit),
            "latest_price": round(latest_price, 2)
        })

    if status in ["FRESH SIGNAL", "ACTIVE BUT LATE"]:
        result["note"] = "Forward testing only."
    elif status == "NO SIGNAL":
        result["note"] = "No valid model signal is present."
    else:
        result["note"] = "Signal is only for review and should not be chased."

    with open(cfg["signal_file"], "w") as f:
        json.dump(result, f, indent=2)

    history_row = pd.DataFrame([result])
    history_file = cfg["history_file"]

    if os.path.exists(history_file):
        old = pd.read_csv(history_file)
        new = pd.concat([old, history_row], ignore_index=True)
    else:
        new = history_row

    new.to_csv(history_file, index=False)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
