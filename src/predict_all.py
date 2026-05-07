"""
Script sinh dự báo 5 ngày tương lai cho TẤT CẢ mã đã được train.
Chạy sau khi đã train mô hình (notebook) và cập nhật raw data (crawl).

Output: src/data/outputs/<CODE>/future_5days.csv
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ─── Cấu hình ────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
STOCK_DIR      = os.path.join(BASE_DIR, "data", "raw-data")
OUTPUT_ROOT    = os.path.join(BASE_DIR, "data", "outputs")
N_FUTURE       = 5
START_YEAR     = 2022
DEVICE         = torch.device("cpu")
# ─────────────────────────────────────────────────────────────────────────────


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm chỉ báo kỹ thuật (không shift target — dùng cho predict)."""
    df = df.copy()
    df["EMA_9"]  = df["close"].ewm(span=9).mean().shift()
    df["SMA_5"]  = df["close"].rolling(5).mean().shift()
    df["SMA_10"] = df["close"].rolling(10).mean().shift()
    df["SMA_15"] = df["close"].rolling(15).mean().shift()
    df["SMA_30"] = df["close"].rolling(30).mean().shift()

    close  = df["close"]
    delta  = close.diff().iloc[1:]
    up     = delta.clip(lower=0)
    down   = delta.clip(upper=0).abs()
    rs     = up.rolling(14).mean() / down.rolling(14).mean()
    rsi    = 100.0 - (100.0 / (1.0 + rs))
    df["RSI"] = rsi.reindex(df.index).fillna(0)

    ema12 = close.ewm(span=12, min_periods=12).mean()
    ema26 = close.ewm(span=26, min_periods=26).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, min_periods=9).mean()

    df = df.iloc[33:]
    df.index = range(len(df))
    return df


def predict_future_for(code: str):
    out_dir = os.path.join(OUTPUT_ROOT, code)
    meta_path     = os.path.join(out_dir, "meta.json")
    scaler_X_path = os.path.join(out_dir, "scaler_X.pkl")
    scaler_y_path = os.path.join(out_dir, "scaler_y.pkl")
    model_path    = os.path.join(out_dir, "best_lstm.pt")
    csv_path      = os.path.join(STOCK_DIR, f"{code}.csv")

    # Kiểm tra đủ artifacts
    for p in [meta_path, scaler_X_path, scaler_y_path, model_path, csv_path]:
        if not os.path.exists(p):
            print(f"  [SKIP] Thiếu file: {p}")
            return False

    with open(meta_path) as f:
        meta = json.load(f)
    with open(scaler_X_path, "rb") as f:
        scaler_X = pickle.load(f)
    with open(scaler_y_path, "rb") as f:
        scaler_y = pickle.load(f)

    feature_cols = meta["feature_cols"]
    seq_len      = meta["seq_len"]

    model = LSTMModel(
        input_size=meta["input_size"],
        hidden_size=meta["hidden_size"],
        num_layers=meta["num_layers"],
        dropout=meta["dropout"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Load raw data
    raw = pd.read_csv(csv_path)
    raw["time"] = pd.to_datetime(raw["time"])
    raw = raw[raw["time"].dt.year >= START_YEAR].copy()
    raw.index = range(len(raw))

    df_feat = add_features(raw)
    X_all   = scaler_X.transform(df_feat[feature_cols].values)
    window  = X_all[-seq_len:].copy()

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

    preds_orig = scaler_y.inverse_transform(
        np.array(preds_scaled).reshape(-1, 1)
    ).flatten()

    last_date    = df_feat["time"].iloc[-1]
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=N_FUTURE
    )

    result = pd.DataFrame({
        "time": future_dates.strftime("%Y-%m-%d"),
        "predicted_close": preds_orig,
    })

    save_path = os.path.join(out_dir, f"future_{N_FUTURE}days.csv")
    result.to_csv(save_path, index=False)
    return result


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    codes = sorted([
        d for d in os.listdir(OUTPUT_ROOT)
        if os.path.isdir(os.path.join(OUTPUT_ROOT, d))
        and os.path.exists(os.path.join(OUTPUT_ROOT, d, "metrics.json"))
    ])

    print(f"Tìm thấy {len(codes)} mã đã train: {codes}")
    print("=" * 60)

    success, failed = 0, []
    for code in codes:
        print(f"  Đang dự báo: {code} ...", end=" ", flush=True)
        try:
            result = predict_future_for(code)
            if result is not False:
                last_pred = result["predicted_close"].iloc[-1]
                print(f"OK  ({N_FUTURE} ngày, giá cuối: {last_pred:.2f})")
                success += 1
            else:
                failed.append(code)
        except Exception as e:
            print(f"LỖI: {e}")
            failed.append(code)

    print("=" * 60)
    print(f"Hoàn thành: {success}/{len(codes)} mã")
    if failed:
        print(f"Mã lỗi: {', '.join(failed)}")
