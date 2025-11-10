# app.py — AI Investment Advisor API (India)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import model_utils
import uvicorn
import os
import traceback

app = FastAPI(title="AI Investment Advisor API (India, INR)")

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Input model for API
class RecommendRequest(BaseModel):
    tickers: Optional[List[str]] = None
    investment_amount_inr: float
    age: int
    monthly_income: Optional[float] = None
    goal_type: Optional[str] = "retirement"  # retirement, education, wealth_creation

@app.post("/recommend")
def recommend(req: RecommendRequest):
    tickers = req.tickers or model_utils.DEFAULT_TICKERS
    try:
        result = model_utils.recommend_portfolio(
            tickers=tickers,
            investment_amount_inr=req.investment_amount_inr,
            age=req.age,
            monthly_income=req.monthly_income,
            goal_type=req.goal_type
        )
        return result
    except Exception as e:
        print("❌ ERROR IN RECOMMENDATION:")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train_model")
def train_model(tickers: Optional[List[str]] = None):
    tickers = tickers or model_utils.DEFAULT_TICKERS
    try:
        path = model_utils.train_and_save_model(tickers=tickers)
        return {"status": "trained", "model_path": path}
    except Exception as e:
        print("❌ ERROR IN TRAINING:")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
