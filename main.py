import os
os.environ["HF_HOME"] = "D:/huggingface_cache"
import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import time
import threading
from datetime import datetime, timezone

from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange

from contextlib import asynccontextmanager
import asyncio

# Replace the lifespan function with this simple version
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Engine ready.")
    yield
    # Runs on shutdown — nothing to clean up

app = FastAPI(
    title="AlphaQuant Advanced Macro Engine",
    version="3.0.0",
    lifespan=lifespan
)
# --- CORS ---
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://yourfrontend.vercel.app"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Constants ---
YIELD_THRESHOLD = 4.20
VIX_THRESHOLD = 20.0
CACHE_DURATION_SECONDS = 900

RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL_WINDOW = 9
BB_WINDOW = 20
BB_STD = 2
ATR_WINDOW = 14
MIN_HISTORY_BARS = 60
VOLUME_SPIKE_MULTIPLIER = 1.5
ATR_VOLATILITY_THRESHOLD = 0.03

# --- Caches ---
MACRO_CACHE: dict = {"data": None, "expiry": 0}
OPTIONS_CACHE: dict = {"data": None, "expiry": 0}

def get_options_put_call_ratio() -> dict:
    current_time = time.time()
    if OPTIONS_CACHE["data"] and current_time < OPTIONS_CACHE["expiry"]:
        return OPTIONS_CACHE["data"]

    fallback = {"put_call_ratio": 1.0, "options_regime": "NEUTRAL"}

    try:
        import requests
        csv_url = "https://www.cboe.com/publish/scheduledtask/mktdata/datahouse/totalpc.csv"
        res = requests.get(csv_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)

        if res.status_code == 200:
            lines = [l for l in res.text.strip().split("\n") if l.strip()]
            # Skip header lines — find first line that starts with a date
            data_lines = [l for l in lines if l[0].isdigit()]
            if data_lines:
                last = data_lines[-1].split(",")
                ratio = float(last[1])
                regime = (
                    "BEARISH_HEDGING" if ratio > 1.0
                    else "BULLISH_COMPLACENCY" if ratio < 0.7
                    else "NEUTRAL"
                )
                result = {"put_call_ratio": round(ratio, 3), "options_regime": regime}
                OPTIONS_CACHE["data"] = result
                OPTIONS_CACHE["expiry"] = current_time + 1800
                return result

    except Exception as e:
        print(f"[Options] Put/call fetch failed: {e}")

    return fallback
# --- FinBERT singleton ---
_finbert = None
_finbert_lock = threading.Lock()



lstm_model = None
print("[LSTM] Disabled on free tier.")
# =============================================================================
# REQUEST SCHEMA
# =============================================================================

# Load LightGBM at startup
try:
    lgbm_payload = joblib.load("lightgbm_importance.pkl")
    lgbm_model = lgbm_payload["model"]
    lgbm_features = lgbm_payload["features"]
    print("[LightGBM] Model loaded successfully.")
except Exception as e:
    print(f"[LightGBM] Model not found — run train_lightgbm.py first. {e}")
    lgbm_model = None
    lgbm_features = []
# Load at startup — if missing, anomaly detection degrades gracefully
try:
    anomaly_detector = joblib.load("isolation_forest_model.pkl")
    print("[IsolationForest] Model loaded successfully.")
except Exception as e:
    print(f"[IsolationForest] Model not found — run train_anomaly.py first. {e}")
    anomaly_detector = None
class AnalyzeRequest(BaseModel):
    ticker: str
    news: str = ""


# =============================================================================
# FINBERT ENGINE
# =============================================================================

def get_finbert():
    """
    Loads ProsusAI/finbert once at first call and reuses the instance.
    Thread-safe via double-checked locking.
    ~400MB download on first run, cached by HuggingFace locally after that.
    """
    global _finbert
    if _finbert is None:
        with _finbert_lock:
            if _finbert is None:
                print("[FinBERT] Loading model — this takes ~10s on first run...")
                _finbert = pipeline(
                    "text-classification",
                    model="ProsusAI/finbert",
                    top_k=None
                )
                print("[FinBERT] Model ready.")
    return _finbert


def compute_finbert_score(news_text: str) -> float:
    """
    Lightweight fallback — keyword-based sentiment when FinBERT unavailable.
    Returns -1.0 to +1.0
    """
    if not news_text or not news_text.strip():
        return 0.0

    text = news_text.lower()

    positive_words = [
        "surge", "rally", "gain", "growth", "beat", "strong", "bullish",
        "upgrade", "record", "profit", "revenue", "outperform", "buy",
        "positive", "rise", "boost", "high", "up", "soar", "jump"
    ]
    negative_words = [
        "fall", "drop", "decline", "loss", "miss", "weak", "bearish",
        "downgrade", "sell", "negative", "crash", "low", "down", "plunge",
        "cut", "risk", "concern", "fear", "recession", "layoff"
    ]

    pos = sum(1 for w in positive_words if w in text)
    neg = sum(1 for w in negative_words if w in text)
    total = pos + neg

    if total == 0:
        return 0.0

    return round((pos - neg) / total, 4)
# =============================================================================
# MACRO ENGINE
# =============================================================================

def get_macro_regime_telemetry() -> dict:
    """
    Fetches US 10Y Treasury Yield, VIX, and DXY (Dollar Index).
    Cached for 15 minutes to reduce external API pressure.
    """
    current_time = time.time()
    if MACRO_CACHE["data"] and current_time < MACRO_CACHE["expiry"]:
        return MACRO_CACHE["data"]

    fallback = {
        "us_10y_yield": 4.25,
        "vix_close": 18.5,
        "dxy": 104.0,
        "liquidity_regime": "RESTRICTIVE_LIQUIDITY",
        "volatility_regime": "RISK_ON",
        "dxy_regime": "STRONG_DOLLAR"
    }

    try:
        tnx_data = yf.Ticker("^TNX").history(period="5d")
        vix_data = yf.Ticker("^VIX").history(period="5d")
        dxy_data = yf.Ticker("DX-Y.NYB").history(period="5d")

        if (
            tnx_data.empty or vix_data.empty
            or "Close" not in tnx_data.columns
            or "Close" not in vix_data.columns
        ):
            return fallback

        current_yield = float(tnx_data["Close"].iloc[-1])
        current_vix = float(vix_data["Close"].iloc[-1])

        # DXY is optional — degrade gracefully if unavailable
        current_dxy = 104.0
        if not dxy_data.empty and "Close" in dxy_data.columns:
            dxy_val = dxy_data["Close"].iloc[-1]
            if not pd.isna(dxy_val):
                current_dxy = float(dxy_val)

        if pd.isna(current_yield) or pd.isna(current_vix):
            return fallback

        result = {
            "us_10y_yield": round(current_yield, 2),
            "vix_close": round(current_vix, 2),
            "dxy": round(current_dxy, 2),
            "liquidity_regime": (
                "RESTRICTIVE_LIQUIDITY" if current_yield > YIELD_THRESHOLD
                else "EXPANSIVE_ACCOMMODATION"
            ),
            "volatility_regime": "RISK_OFF" if current_vix > VIX_THRESHOLD else "RISK_ON",
            "dxy_regime": "STRONG_DOLLAR" if current_dxy > 104 else "WEAK_DOLLAR"
        }

        MACRO_CACHE["data"] = result
        MACRO_CACHE["expiry"] = current_time + CACHE_DURATION_SECONDS
        return result

    except Exception:
        return fallback

# =============================================================================
# SCORING ENGINE
# =============================================================================

def calculate_composite_score(
    rsi: float,
    macd_signal: str,
    bb_position: str,
    macro: dict,
    finbert_score: float,
    options_data: dict
) -> int:
    weights = {"rsi": 0.20, "macd": 0.20, "bb": 0.15, "finbert": 0.20, "macro": 0.10, "options": 0.15}

    if rsi <= 30:
        rsi_score = 1.0
    elif rsi >= 70:
        rsi_score = -1.0
    else:
        rsi_score = 1.0 - 2.0 * ((rsi - 30) / 40)

    macd_score = 1.0 if macd_signal == "BULLISH" else (-1.0 if macd_signal == "BEARISH" else 0.0)
    bb_score = 1.0 if bb_position == "OVERSOLD" else (-1.0 if bb_position == "OVERBOUGHT" else 0.0)
    finbert_clamped = max(min(float(finbert_score), 1.0), -1.0)

    # DXY: strong dollar is bearish for commodities and emerging markets
    dxy_regime = macro.get("dxy_regime", "NEUTRAL")
    macro_score = -0.5 if dxy_regime == "STRONG_DOLLAR" else 0.5

    # Options flow: high put/call = bearish hedging = contrarian buy signal
    options_regime = options_data.get("options_regime", "NEUTRAL")
    options_score = (
        1.0 if options_regime == "BEARISH_HEDGING"
        else -0.5 if options_regime == "BULLISH_COMPLACENCY"
        else 0.0
    )

    base = (
        rsi_score * weights["rsi"]
        + macd_score * weights["macd"]
        + bb_score * weights["bb"]
        + finbert_clamped * weights["finbert"]
        + macro_score * weights["macro"]
        + options_score * weights["options"]
    )

    score = ((base + 1) / 2) * 100

    if macro["liquidity_regime"] == "RESTRICTIVE_LIQUIDITY":
        score *= 0.93
    elif macro["liquidity_regime"] == "EXPANSIVE_ACCOMMODATION":
        score *= 1.07

    if macro["volatility_regime"] == "RISK_OFF":
        score *= 0.90
    elif macro["volatility_regime"] == "RISK_ON":
        score *= 1.03

    return max(min(int(score), 95), 1)
# =============================================================================
# ANOMALY DETECTION
# =============================================================================

def compute_anomaly_flag(df: pd.DataFrame) -> bool:
    """
    Real Isolation Forest anomaly detection.
    Features match train_anomaly.py exactly — log return vol + volume ratio.
    Falls back to False if model not loaded or data insufficient.
    """
    if anomaly_detector is None:
        return False

    try:
        df = df.copy()
        df["returns"] = np.log(df["close"] / df["close"].shift(1))
        df["price_vol_10d"] = df["returns"].rolling(window=10).std()
        df["volume_ma20"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma20"]

        latest = df.iloc[-1]
        price_vol = float(latest["price_vol_10d"])
        volume_ratio = float(latest["volume_ratio"])

        if np.isnan(price_vol) or np.isnan(volume_ratio):
            return False

        prediction = anomaly_detector.predict([[price_vol, volume_ratio]])[0]
        return bool(prediction == -1)

    except Exception as e:
        print(f"[IsolationForest] Inference failed: {e}")
        return False


def compute_lstm_pattern(df: pd.DataFrame) -> str:
    return "NEUTRAL_CONTINUATION"

def compute_feature_importance(
    rsi: float,
    macd_diff: float,
    bb_position: float,
    bb_width: float,
    atr_pct: float,
    vol_10d: float,
    volume_ratio: float,
) -> list[str]:
    """
    LightGBM-based feature importance.
    Returns features ranked by their trained importance weights
    multiplied by how active each signal currently is.
    Falls back to rule-based ranking if model not loaded.
    """
    FEATURE_LABELS = {
        "rsi": "RSI",
        "macd_diff": "MACD",
        "bb_position": "BOLLINGER",
        "bb_width": "BB_WIDTH",
        "atr_pct": "ATR",
        "vol_10d": "VOLATILITY",
        "volume_ratio": "VOLUME"
    }

    if lgbm_model is None:
        # Rule-based fallback
        scores = {
            "RSI": abs(rsi - 50) / 20.0,
            "MACD": min(abs(macd_diff), 1.0),
            "BOLLINGER": abs(bb_position - 0.5) * 2,
            "ATR": min(atr_pct / 0.03, 1.0),
            "VOLUME": min(volume_ratio / 2.0, 1.0),
        }
        return [k for k, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

    try:
        # Current signal values in exact training order
        current_values = np.array([[
            rsi,
            macd_diff,
            bb_position,
            bb_width,
            atr_pct,
            vol_10d,
            volume_ratio
        ]])

        # Trained importance weights (static, from training)
        trained_weights = lgbm_model.feature_importances_

        # Normalize current values to [0, 1] range for each feature
        # This measures how "active" or extreme each signal is right now
        normalized = np.abs(current_values[0]) / (np.abs(current_values[0]).sum() + 1e-9)

        # Final rank = trained weight × current signal activity
        # This means a strong trained predictor that's also currently extreme ranks highest
        combined_scores = trained_weights * normalized

        ranked = sorted(
            zip(lgbm_features, combined_scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [FEATURE_LABELS.get(feat, feat.upper()) for feat, _ in ranked]

    except Exception as e:
        print(f"[LightGBM] Importance inference failed: {e}")
        return ["RSI", "MACD", "ATR", "VOLUME", "BOLLINGER"]

# =============================================================================
# INDICATOR COMPUTATION
# =============================================================================

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["rsi"] = RSIIndicator(close=df["close"], window=RSI_WINDOW).rsi()

    macd_obj = MACD(
        close=df["close"],
        window_fast=MACD_FAST,
        window_slow=MACD_SLOW,
        window_sign=MACD_SIGNAL_WINDOW
    )
    df["macd"] = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()

    bb_obj = BollingerBands(close=df["close"], window=BB_WINDOW, window_dev=BB_STD)
    df["bb_upper"] = bb_obj.bollinger_hband()
    df["bb_lower"] = bb_obj.bollinger_lband()

    df["atr"] = AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=ATR_WINDOW
    ).average_true_range()

    return df


def compute_macd_status(recent_window: pd.DataFrame) -> str:
    bullish = (recent_window["macd"] > recent_window["macd_signal"]).sum()
    bearish = (recent_window["macd"] < recent_window["macd_signal"]).sum()
    if bullish >= 2:
        return "BULLISH"
    if bearish >= 2:
        return "BEARISH"
    return "NEUTRAL"


def compute_bollinger_zone(live_price: float, bb_upper: float, bb_lower: float) -> str:
    if not pd.isna(bb_upper) and live_price >= bb_upper:
        return "OVERBOUGHT"
    if not pd.isna(bb_lower) and live_price <= bb_lower:
        return "OVERSOLD"
    return "NEUTRAL"


def compute_volume_signal(df: pd.DataFrame) -> dict:
    if "volume" not in df.columns or df["volume"].isna().all():
        return {"signal": "UNAVAILABLE", "latest": 0, "avg_20d": 0}

    avg_20d = df["volume"].iloc[-20:].mean()
    latest = float(df.iloc[-1]["volume"])

    avg_20d_safe = avg_20d if not pd.isna(avg_20d) else 0.0
    latest_safe = latest if not pd.isna(latest) else 0.0

    signal = (
        "HIGH_CONVICTION"
        if avg_20d_safe > 0 and latest_safe > avg_20d_safe * VOLUME_SPIKE_MULTIPLIER
        else "LOW_CONVICTION"
    )
    return {
        "signal": signal,
        "latest": int(latest_safe),
        "avg_20d": int(avg_20d_safe)
    }


# =============================================================================
# ROUTES
# =============================================================================

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "engine": "AlphaQuant Core API v3",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.post("/api/v1/analyze")
async def analyze_asset(body: AnalyzeRequest):
    try:
        asset_ticker = body.ticker.upper().strip()
        news_context = body.news

        try:
            df = yf.Ticker(asset_ticker).history(period="1y")
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Market data fetch failed: {str(e)}")

        if anomaly_detector is None:
            print("[WARNING] Isolation Forest not loaded — anomaly_flag will return False.")

        if df.empty or len(df) < MIN_HISTORY_BARS:
            raise HTTPException(
                status_code=404,
                detail=f"Ticker '{asset_ticker}' returned insufficient history ({len(df)} bars)."
            )

        df.columns = [col.lower() for col in df.columns]
        df = compute_indicators(df)

        latest_row = df.iloc[-1]
        recent_window = df.iloc[-3:]
        live_price = float(latest_row["close"])

        rsi_val = float(latest_row["rsi"]) if not pd.isna(latest_row["rsi"]) else 50.0
        macd_status = compute_macd_status(recent_window)
        bollinger_zone = compute_bollinger_zone(
            live_price, latest_row["bb_upper"], latest_row["bb_lower"]
        )
        atr_val = float(latest_row["atr"]) if not pd.isna(latest_row["atr"]) else 0.0
        volatility_profile = (
            "HIGH_VOLATILITY"
            if atr_val > (live_price * ATR_VOLATILITY_THRESHOLD)
            else "STABLE_COMPRESSION"
        )
        volume = compute_volume_signal(df)
        macro = get_macro_regime_telemetry()
        options_data = get_options_put_call_ratio()

        # --- Compute LightGBM input features ---
        bb_upper = latest_row["bb_upper"]
        bb_lower = latest_row["bb_lower"]
        bb_width = float(bb_upper - bb_lower) if not (pd.isna(bb_upper) or pd.isna(bb_lower)) else 0.0
        bb_position = float(
            (live_price - bb_lower) / (bb_width + 1e-9)
        ) if bb_width > 0 else 0.5

        macd_val = latest_row["macd"]
        macd_sig = latest_row["macd_signal"]
        macd_diff = float(macd_val - macd_sig) if not (pd.isna(macd_val) or pd.isna(macd_sig)) else 0.0

        returns_series = np.log(df["close"] / df["close"].shift(1))
        vol_10d = float(returns_series.iloc[-10:].std()) if len(df) >= 10 else 0.0

        volume_ma20 = df["volume"].iloc[-20:].mean() if "volume" in df.columns else 1.0
        latest_volume = float(latest_row.get("volume", 0))
        volume_ratio = float(latest_volume / (volume_ma20 + 1e-9)) if not pd.isna(volume_ma20) else 1.0

        atr_pct = atr_val / live_price if live_price > 0 else 0.0

        # --- ML layer ---
        finbert_score = compute_finbert_score(news_context)
        anomaly_flag = compute_anomaly_flag(df)
        feature_ranking = compute_feature_importance(
            rsi_val, macd_diff, bb_position, bb_width,
            atr_pct, vol_10d, volume_ratio
        )

        score = calculate_composite_score(
            rsi_val, macd_status, bollinger_zone, macro, finbert_score, options_data
        )

        return {
            "status": "success",
            "ticker": asset_ticker,
            "live_price": round(live_price, 2),
            "composite_sentiment_score": score,
            "macro_telemetry": {
                **macro,
                "put_call_ratio": options_data["put_call_ratio"],
                "options_regime": options_data["options_regime"]
            },
                "execution_signals": {
                "rsi_value": round(rsi_val, 2),
                "macd_confirmed_status": macd_status,
                "bollinger_zone": bollinger_zone,
                "volume_conviction": volume["signal"]
            },
            "metrics": {
                "average_true_range": round(atr_val, 4),
                "volatility_profile": volatility_profile,
                "latest_volume": volume["latest"],
                "avg_20d_volume": volume["avg_20d"]
            },
            "ml_signals": {
                "finbert_sentiment_score": finbert_score,
                "lstm_sequence_pattern": compute_lstm_pattern(df),
                "feature_importance_ranking": feature_ranking,
                "anomaly_flag": anomaly_flag
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Engine failure: {str(e)}")
    
