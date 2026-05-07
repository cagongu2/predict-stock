import os
import json
import pickle
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "src", "data", "outputs")
STOCK_DIR  = os.path.join(BASE_DIR, "src", "data", "raw-data")
START_YEAR = 2022
N_FUTURE   = 5
DEVICE     = torch.device("cpu")


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA_9"]  = df["close"].ewm(span=9).mean().shift()
    df["SMA_5"]  = df["close"].rolling(5).mean().shift()
    df["SMA_10"] = df["close"].rolling(10).mean().shift()
    df["SMA_15"] = df["close"].rolling(15).mean().shift()
    df["SMA_30"] = df["close"].rolling(30).mean().shift()
    close  = df["close"]
    delta  = close.diff().iloc[1:]
    rs     = delta.clip(lower=0).rolling(14).mean() / delta.clip(upper=0).abs().rolling(14).mean()
    df["RSI"] = (100.0 - 100.0 / (1.0 + rs)).reindex(df.index).fillna(0)
    ema12 = close.ewm(span=12, min_periods=12).mean()
    ema26 = close.ewm(span=26, min_periods=26).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, min_periods=9).mean()
    df = df.iloc[33:]
    df.index = range(len(df))
    return df


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
    future_path = os.path.join(code_dir, f"future_{N_FUTURE}days.csv")
    if os.path.exists(future_path):
        df = pd.read_csv(future_path)
        data["future"] = df.to_dict(orient="records")
    return data


@app.post("/api/stocks/{code}/predict")
def predict_stock(code: str):
    """Tạo dự báo tương lai ON-DEMAND cho một mã cụ thể."""
    code_dir      = os.path.join(OUTPUTS_DIR, code)
    meta_path     = os.path.join(code_dir, "meta.json")
    scaler_X_path = os.path.join(code_dir, "scaler_X.pkl")
    scaler_y_path = os.path.join(code_dir, "scaler_y.pkl")
    model_path    = os.path.join(code_dir, "best_lstm.pt")
    csv_path      = os.path.join(STOCK_DIR, f"{code}.csv")

    for p in [meta_path, scaler_X_path, scaler_y_path, model_path, csv_path]:
        if not os.path.exists(p):
            raise HTTPException(status_code=404, detail=f"Thiếu file artifact: {os.path.basename(p)}")

    try:
        with open(meta_path) as f:         meta     = json.load(f)
        with open(scaler_X_path, "rb") as f: scaler_X = pickle.load(f)
        with open(scaler_y_path, "rb") as f: scaler_y = pickle.load(f)

        feature_cols = meta["feature_cols"]
        seq_len      = meta["seq_len"]

        model = LSTMModel(
            input_size=meta["input_size"], hidden_size=meta["hidden_size"],
            num_layers=meta["num_layers"], dropout=meta["dropout"],
        ).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        raw = pd.read_csv(csv_path)
        raw["time"] = pd.to_datetime(raw["time"])
        raw = raw[raw["time"].dt.year >= START_YEAR].copy()
        raw.index = range(len(raw))

        df_feat   = _add_features(raw)
        X_all     = scaler_X.transform(df_feat[feature_cols].values)
        window    = X_all[-seq_len:].copy()
        close_idx = feature_cols.index("close") if "close" in feature_cols else None

        preds_scaled = []
        with torch.no_grad():
            for _ in range(N_FUTURE):
                x_t  = torch.tensor(window[np.newaxis], dtype=torch.float32).to(DEVICE)
                p_sc = model(x_t).cpu().item()
                preds_scaled.append(p_sc)
                new_row = window[-1].copy()
                if close_idx is not None:
                    new_row[close_idx] = p_sc
                window = np.vstack([window[1:], new_row])

        preds_orig   = scaler_y.inverse_transform(
            np.array(preds_scaled).reshape(-1, 1)
        ).flatten()
        last_date    = df_feat["time"].iloc[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=N_FUTURE)

        result = pd.DataFrame({
            "time": future_dates.strftime("%Y-%m-%d"),
            "predicted_close": preds_orig,
        })
        result.to_csv(os.path.join(code_dir, f"future_{N_FUTURE}days.csv"), index=False)

        return {
            "status": "success",
            "code": code,
            "future": result.to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
