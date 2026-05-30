import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import joblib
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange

# ---------------------------------------------------------------------------
# Fetch training data — use SPY for broad market representation
# More tickers = better generalization
# ---------------------------------------------------------------------------
TRAINING_TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA", "GC=F", "BTC-USD"]

def build_features(ticker: str) -> pd.DataFrame:
    print(f"Fetching {ticker}...")
    df = yf.Ticker(ticker).history(period="5y")
    
    if df.empty or len(df) < 100:
        return pd.DataFrame()

    df.columns = [c.lower() for c in df.columns]

    # --- Compute all indicators ---
    df["rsi"] = RSIIndicator(close=df["close"], window=14).rsi()

    macd = MACD(close=df["close"], window_fast=12, window_slow=26, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = df["macd"] - df["macd_signal"]  # raw crossover distance

    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_width"] + 1e-9)  # 0=bottom, 1=top

    df["atr"] = AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range()
    df["atr_pct"] = df["atr"] / df["close"]  # ATR as % of price

    df["returns"] = np.log(df["close"] / df["close"].shift(1))
    df["vol_10d"] = df["returns"].rolling(10).std()

    df["volume_ma20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_ma20"] + 1e-9)

    # --- Target: did price go up 3 days from now? ---
    # This is what LightGBM learns to predict
    # Feature importance = which indicators most predicted this outcome
    df["future_return"] = df["close"].shift(-3) / df["close"] - 1
    df["target"] = (df["future_return"] > 0).astype(int)  # 1=up, 0=down

    feature_cols = [
        "rsi", "macd_diff", "bb_position", "bb_width",
        "atr_pct", "vol_10d", "volume_ratio"
    ]

    df = df[feature_cols + ["target"]].dropna()
    return df


# ---------------------------------------------------------------------------
# Build combined dataset from all tickers
# ---------------------------------------------------------------------------
all_frames = []
for ticker in TRAINING_TICKERS:
    try:
        frame = build_features(ticker)
        if not frame.empty:
            all_frames.append(frame)
    except Exception as e:
        print(f"Skipped {ticker}: {e}")

combined = pd.concat(all_frames, ignore_index=True)
print(f"Total training rows: {len(combined)}")

FEATURE_COLS = [
    "rsi", "macd_diff", "bb_position", "bb_width",
    "atr_pct", "vol_10d", "volume_ratio"
]

X = combined[FEATURE_COLS].values
y = combined["target"].values

# ---------------------------------------------------------------------------
# Train LightGBM
# ---------------------------------------------------------------------------
model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    random_state=42,
    verbose=-1
)

print("Training LightGBM...")
model.fit(X, y)

# ---------------------------------------------------------------------------
# Save model + feature column order
# Column order MUST match exactly when running inference in main.py
# ---------------------------------------------------------------------------
joblib.dump({"model": model, "features": FEATURE_COLS}, "lightgbm_importance.pkl")
print("Saved lightgbm_importance.pkl")

# Print what it learned
importances = model.feature_importances_
for col, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True):
    print(f"  {col}: {imp}")