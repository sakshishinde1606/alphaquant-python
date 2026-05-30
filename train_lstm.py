import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import joblib
import json

# Try torch first, fall back to guidance
try:
    import torch
    import torch.nn as nn
    BACKEND = "torch"
except ImportError:
    BACKEND = None
    print("Install PyTorch: pip install torch")
    exit()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 20      # 20 trading days lookback window
PREDICTION_HORIZON = 3    # predict 3 days ahead
TRAINING_TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA", "GC=F", "BTC-USD"]

PATTERN_LABELS = {
    0: "BEARISH_BREAKDOWN",
    1: "BEARISH_CONSOLIDATION", 
    2: "NEUTRAL_CONTINUATION",
    3: "BULLISH_CONSOLIDATION",
    4: "BULLISH_BREAKOUT"
}

# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------
def build_ohlcv_features(ticker: str) -> np.ndarray:
    df = yf.Ticker(ticker).history(period="5y")
    if df.empty or len(df) < 100:
        return np.array([])

    df.columns = [c.lower() for c in df.columns]

    # Normalize OHLCV into return-based features
    df["ret"] = df["close"].pct_change()
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["co_range"] = (df["close"] - df["open"]) / df["open"]
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    feature_cols = ["ret", "hl_range", "co_range", "vol_ratio"]
    df = df[feature_cols].dropna()

    return df.values


# ---------------------------------------------------------------------------
# Sequence builder — returns X (sequences) and y (5-class pattern labels)
# ---------------------------------------------------------------------------
def build_sequences(data: np.ndarray):
    X, y = [], []

    for i in range(len(data) - SEQUENCE_LENGTH - PREDICTION_HORIZON):
        seq = data[i:i + SEQUENCE_LENGTH]
        
        # Future return over prediction horizon
        future_ret = data[i + SEQUENCE_LENGTH:i + SEQUENCE_LENGTH + PREDICTION_HORIZON, 0].sum()
        
        # Classify into 5 patterns based on return magnitude and direction
        if future_ret < -0.03:
            label = 0  # BEARISH_BREAKDOWN
        elif future_ret < -0.01:
            label = 1  # BEARISH_CONSOLIDATION
        elif future_ret < 0.01:
            label = 2  # NEUTRAL_CONTINUATION
        elif future_ret < 0.03:
            label = 3  # BULLISH_CONSOLIDATION
        else:
            label = 4  # BULLISH_BREAKOUT

        X.append(seq)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ---------------------------------------------------------------------------
# LSTM model definition
# ---------------------------------------------------------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # last timestep only


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------
all_X, all_y = [], []
for ticker in TRAINING_TICKERS:
    try:
        data = build_ohlcv_features(ticker)
        if len(data) < SEQUENCE_LENGTH + PREDICTION_HORIZON + 10:
            continue
        X, y = build_sequences(data)
        all_X.append(X)
        all_y.append(y)
        print(f"{ticker}: {len(X)} sequences")
    except Exception as e:
        print(f"Skipped {ticker}: {e}")

X_all = np.concatenate(all_X)
y_all = np.concatenate(all_y)
print(f"Total sequences: {len(X_all)}")

# Normalize features
scaler = MinMaxScaler()
X_flat = X_all.reshape(-1, X_all.shape[-1])
X_scaled = scaler.fit_transform(X_flat).reshape(X_all.shape)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.LongTensor(y_all)

dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

model = LSTMClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("Training LSTM...")
model.train()
for epoch in range(15):
    total_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/15 — Loss: {total_loss/len(loader):.4f}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
torch.save(model.state_dict(), "lstm_model.pth")
joblib.dump(scaler, "lstm_scaler.pkl")
joblib.dump(PATTERN_LABELS, "lstm_labels.pkl")
print("Saved lstm_model.pth, lstm_scaler.pkl, lstm_labels.pkl")