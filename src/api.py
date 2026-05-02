import os
import json
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "src", "data", "outputs")

@app.get("/api/stocks")
def get_stocks():
    if not os.path.exists(OUTPUTS_DIR):
        return []
    
    stocks = []
    for code in os.listdir(OUTPUTS_DIR):
        code_dir = os.path.join(OUTPUTS_DIR, code)
        if os.path.isdir(code_dir) and os.path.exists(os.path.join(code_dir, "metrics.json")):
            stocks.append(code)
    return sorted(stocks)

@app.get("/api/stocks/{code}")
def get_stock_data(code: str):
    code_dir = os.path.join(OUTPUTS_DIR, code)
    if not os.path.exists(code_dir):
        raise HTTPException(status_code=404, detail="Stock not found")
    
    data = {}
    
    metrics_path = os.path.join(code_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            data["metrics"] = json.load(f)
            
    predictions_path = os.path.join(code_dir, "predictions.csv")
    if os.path.exists(predictions_path):
        df = pd.read_csv(predictions_path)
        data["predictions"] = df.to_dict(orient="records")
        
    future_path = os.path.join(code_dir, "future_5days.csv")
    if os.path.exists(future_path):
        df = pd.read_csv(future_path)
        data["future"] = df.to_dict(orient="records")
        
    return data

@app.post("/api/crawl")
def crawl_data():
    try:
        script_path = os.path.join(BASE_DIR, "src", "craw_vn30.py")
        subprocess.run(["python", script_path], check=True, cwd=BASE_DIR)
        return {"status": "success", "message": "Crawl completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
