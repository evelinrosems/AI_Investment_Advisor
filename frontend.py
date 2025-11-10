# frontend.py (Streamlit UI modified for INR and Indian investment suggestions)
import streamlit as st
import requests
import pandas as pd

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Investment Advisor (India)", layout="centered")

st.title("AI Investment Advisor — India (INR)")

with st.sidebar:
    st.header("Investor profile")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    monthly_income = st.number_input("Monthly income (INR)", min_value=0.0, value=50000.0, step=1000.0)
    goal = st.selectbox("Primary goal", ["retirement", "education", "wealth_creation"])
    invest_amt = st.number_input("Total investible amount (INR)", min_value=1000.0, value=100000.0, step=1000.0)
    custom_tickers = st.text_area("Optional: comma-separated Indian stock tickers (e.g. RELIANCE.NS, INFY.NS)", height=80)

if st.button("Get recommendation"):
    tickers = [t.strip() for t in custom_tickers.split(",") if t.strip()] or None
    payload = {
        "tickers": tickers,
        "investment_amount_inr": invest_amt,
        "age": int(age),
        "monthly_income": float(monthly_income),
        "goal_type": goal
    }
    try:
        resp = requests.post(f"{API_BASE}/recommend", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        st.subheader("Suggested allocation")
        st.write(f"Total investible amount: ₹{data['investment_amount_inr']:,}")
        st.write(f"Equity allocation: {round(data['equity_pct']*100,2)}%")
        st.write(f"Debt allocation: {round(data['debt_pct']*100,2)}%")
        st.write(" ")
        st.subheader("Equity positions (suggested)")
        df = pd.DataFrame(data['positions'])
        if not df.empty:
            df['allocation_amount_inr'] = df['allocation_amount_inr'].apply(lambda x: f"₹{x:,.2f}")
            st.table(df)
        st.subheader("Suggested instruments and notes")
        st.write("- Equity: ", data.get("equity_suggested"))
        st.write("- Debt: ", data.get("debt_suggested_instruments"))
        st.write("- Gold: ", data.get("gold_suggested_instruments"))
        st.write("- Tax-saving options: ", data.get("tax_saving_options"))
    except Exception as e:
        st.error(f"Failed to get recommendation: {e}")
