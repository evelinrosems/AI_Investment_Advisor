# model_utils.py (Modified for Indian investments and INR)
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import dump, load
import os
from datetime import datetime, timedelta

MODEL_PATH = "models"
os.makedirs(MODEL_PATH, exist_ok=True)

# Default Indian tickers (NSE via Yahoo Finance uses .NS suffix)
DEFAULT_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HDFC.NS", "KOTAKBANK.NS", "LT.NS", "HINDUNILVR.NS", "SBIN.NS"
]

def fetch_price_series(ticker, period_days=365):
    # fetch daily close price series for given ticker
    try:
        end = datetime.utcnow().date()
        start = end - timedelta(days=period_days)
        df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False, threads=False)
        if df.empty:
            return None
        return df['Close'].dropna()
    except Exception:
        return None

def compute_features_for_tickers(tickers, period_days=365):
    # returns DataFrame with simple technical features for each ticker
    rows = []
    for t in tickers:
        s = fetch_price_series(t, period_days=period_days)
        if s is None or len(s) < 60:
            continue
        returns_7 = s.pct_change(7).iloc[-1]
        returns_30 = s.pct_change(30).iloc[-1]
        vol_90 = s.pct_change().rolling(window=90).std().iloc[-1] * np.sqrt(252)
        y = s.iloc[-60:].values
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        recent_mean = s.iloc[-30:].mean()
        rows.append({
            "ticker": t,
            "returns_7": returns_7,
            "returns_30": returns_30,
            "vol_90": vol_90,
            "momentum_slope": slope,
            "recent_mean": recent_mean,
            # simple surrogate target: next-30-day pct change (if available)
            "target": (s.pct_change(30).iloc[-1] if len(s) > 30 else np.nan)
        })
    df = pd.DataFrame(rows).set_index("ticker")
    df = df.dropna()
    return df

def train_and_save_model(tickers=None):
    if tickers is None:
        tickers = DEFAULT_TICKERS
    table = compute_features_for_tickers(tickers)
    if table.shape[0] < 3:
        raise ValueError("Not enough data to train model. Try more tickers or longer period.")
    X = table[["returns_7", "returns_30", "vol_90", "momentum_slope", "recent_mean"]].values
    y = table["target"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    # save model
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_file = os.path.join(MODEL_PATH, f"rf_model_{stamp}.joblib")
    dump({"model": model, "index": table.index.tolist()}, model_file)
    return model_file

def load_latest_model():
    files = [f for f in os.listdir(MODEL_PATH) if f.endswith('.joblib')]
    if not files:
        return None
    latest = sorted(files)[-1]
    from joblib import load
    return load(os.path.join(MODEL_PATH, latest))

def recommend_portfolio(tickers, investment_amount_inr, age, monthly_income, goal_type="retirement"):
    """
    Simple rule-based + model-assisted portfolio suggestion tailored to Indian investors.
    - Converts investment_amount_inr (assumed INR)
    - Age and income used to compute risk profile
    - Recommends buckets: Equities (direct stocks + mutual funds/SIP), Debt (PPF, FD), Gold, Cash
    """
    df = compute_features_for_tickers(tickers)
    if df.empty:
        raise ValueError("No data for provided tickers.")
    # risk score: younger => more equity
    age = int(age)
    if age < 30:
        equity_pct = 0.80
    elif age < 45:
        equity_pct = 0.65
    elif age < 60:
        equity_pct = 0.50
    else:
        equity_pct = 0.35

    # adjust for income (higher income -> slightly more equities)
    if monthly_income is not None:
        try:
            inc = float(monthly_income)
            if inc > 200000:  # very high income
                equity_pct = min(0.9, equity_pct + 0.05)
            elif inc < 30000:
                equity_pct = max(0.4, equity_pct - 0.05)
        except Exception:
            pass

    # goal-based tweak
    if goal_type == "retirement":
        equity_pct = max(0.4, equity_pct - 0.05)
    elif goal_type == "education":
        equity_pct = max(0.5, equity_pct - 0.0)
    elif goal_type == "wealth_creation":
        equity_pct = min(0.9, equity_pct + 0.05)

    equity_amount = investment_amount_inr * equity_pct
    debt_amount = investment_amount_inr * (1 - equity_pct) * 0.7
    gold_amount = investment_amount_inr * (1 - equity_pct) * 0.15
    cash_amount = investment_amount_inr - equity_amount - debt_amount - gold_amount

    # Simple weighting inside equities using model predicted returns if model exists
    model_bundle = load_latest_model()
    weights = {}
    if model_bundle is not None:
        model = model_bundle["model"]
        available = df.index.tolist()
        X = df[["returns_7", "returns_30", "vol_90", "momentum_slope", "recent_mean"]].values
        preds = model.predict(X)
        score = pd.Series(preds, index=available)
        score = score.rank(ascending=False)
        score = score / score.sum()
        # choose top 6 tickers
        chosen = score.sort_values(ascending=False).head(6)
        for t, w in chosen.items():
            weights[t] = float(w / chosen.sum())
    else:
        # fallback: equal weight top 6 by recent performance
        top = df.sort_values("returns_30", ascending=False).head(6)
        w = 1.0 / len(top)
        for t in top.index:
            weights[t] = w

    positions = []
    for ticker, w in weights.items():
        alloc_amount = equity_amount * w
        positions.append({
            "ticker": ticker,
            "allocation_amount_inr": round(float(alloc_amount), 2),
            "allocation_percent_of_total": round(100 * alloc_amount / investment_amount_inr, 2),
            "source": "equity"
        })

    # Suggested instruments (Indian context)
    suggestions = {
        "equities": "Direct stocks (selected above) and diversified large-cap mutual funds or Nifty 50 index funds. Consider SIPs for rupee-cost averaging.",
        "debt": "PPF (long-term, tax-free), Bank FDs, Corporate FDs, Debt mutual funds depending on horizon.",
        "gold": "Sovereign Gold Bonds (SGB) or Gold ETFs rather than physical gold for better liquidity and safety.",
        "tax_saving": "If you want tax benefits: ELSS (Equity Linked Savings Scheme) for Section 80C, PPF for long-term saving, NPS for retirement benefits."
    }

    return {
        "investment_amount_inr": investment_amount_inr,
        "equity_pct": equity_pct,
        "debt_pct": round(1 - equity_pct, 2),
        "positions": positions,
        "debt_suggested_instruments": suggestions["debt"],
        "gold_suggested_instruments": suggestions["gold"],
        "equity_suggested": suggestions["equities"],
        "tax_saving_options": suggestions["tax_saving"]
    }
