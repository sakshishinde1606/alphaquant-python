import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import IsolationForest
import joblib

print("Fetching SPY training data (2020–2026)...")
df = yf.download("SPY", start="2020-01-01", end="2026-01-01")

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df["returns"] = np.log(df["Close"] / df["Close"].shift(1))
df["price_vol_10d"] = df["returns"].rolling(window=10).std()
df["volume_ma20"] = df["Volume"].rolling(window=20).mean()
df["volume_ratio"] = df["Volume"] / df["volume_ma20"]

features = df[["price_vol_10d", "volume_ratio"]].dropna()

print(f"Training on {len(features)} trading days...")
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(features.values)

joblib.dump(model, "isolation_forest_model.pkl")
print("Done — isolation_forest_model.pkl saved.")