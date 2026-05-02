"""
Cào dữ liệu lịch sử giá cho các mã VN30 bằng vnstock 3.x
- Dữ liệu đã được điều chỉnh giá (adjusted price)
- Source: VCI
- Output: d:\predict-stock\src\data\raw-data\<SYMBOL>.csv
"""

import sys
import json
import os
from datetime import datetime
from vnstock.api.quote import Quote

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ─── Cấu hình ────────────────────────────────────────────────────────────────
JSON_FILE  = r"d:\predict-stock\src\vn30.json"
OUTPUT_DIR = r"d:\predict-stock\src\data\raw-data"
SOURCE     = "VCI"
START_DATE = "1990-01-01"
# ─────────────────────────────────────────────────────────────────────────────

from dotenv import load_dotenv
load_dotenv()
os.environ["VNSTOCK_API_KEY"] = os.getenv("STOCK_API_KEY", "")
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(JSON_FILE, "r", encoding="utf-8") as f:
    stocks = json.load(f)

end_date = datetime.now().strftime("%Y-%m-%d")
total    = len(stocks)
success  = 0
failed   = []

print(f"Bat dau cao {total} ma VN30 tu vnstock (source={SOURCE})")
print(f"Khoang thoi gian: {START_DATE} -> {end_date}")
print("=" * 60)

for idx, (code, name) in enumerate(stocks.items(), start=1):
    print(f"[{idx:02d}/{total}] {code} ({name})...", end=" ", flush=True)
    try:
        q = Quote(symbol=code, source=SOURCE)
        df = q.history(start=START_DATE, end=end_date, interval="1D")

        csv_path = os.path.join(OUTPUT_DIR, f"{code}.csv")
        df.to_csv(csv_path, index=False)

        t_min = str(df['time'].min())[:10]
        t_max = str(df['time'].max())[:10]
        print(f"OK  {len(df)} dòng  [{t_min} -> {t_max}]")
        success += 1

    except Exception as e:
        print(f"LOI: {e}")
        failed.append(code)

print("=" * 60)
print(f"Hoan thanh: {success}/{total} ma thanh cong")
if failed:
    print(f"cac ma loi: {', '.join(failed)}")