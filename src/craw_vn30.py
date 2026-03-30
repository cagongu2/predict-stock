"""
Cao du lieu lich su gia cho cac ma VN30 bang vnstock 3.x
- Du lieu da duoc dieu chinh gia (adjusted price)
  => Da xu ly pha loang co phieu, chia tach, tra co tuc bang co phieu
- Source: VCI
- Output: d:\\predict-stock\\src\\data\\raw-data\\<SYMBOL>.csv
"""

import sys
import json
import os
from datetime import datetime
from vnstock import Vnstock

# Set UTF-8 stdout để tránh UnicodeEncodeError trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ─── Cau hinh ────────────────────────────────────────────────────────────────
JSON_FILE  = r"d:\predict-stock\src\vn30.json"
OUTPUT_DIR = r"d:\predict-stock\src\data\raw-data"
SOURCE     = "VCI"         # VCI cho du lieu adjusted price
START_DATE = "1990-01-01"
# ─────────────────────────────────────────────────────────────────────────────

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
        stock = Vnstock().stock(symbol=code, source=SOURCE)
        df = stock.quote.history(
            start=START_DATE,
            end=end_date,
            interval="1D"
        )

        csv_path = os.path.join(OUTPUT_DIR, f"{code}.csv")
        df.to_csv(csv_path, index=False)

        t_min = df['time'].min()
        t_max = df['time'].max()
        if hasattr(t_min, 'strftime'):
            t_min = t_min.strftime('%Y-%m-%d')
            t_max = t_max.strftime('%Y-%m-%d')
        print(f"OK  {len(df)} dong  [{t_min} -> {t_max}]")
        success += 1

    except Exception as e:
        print(f"LOI: {e}")
        failed.append(code)

print("=" * 60)
print(f"Hoan thanh: {success}/{total} ma thanh cong")
if failed:
    print(f"Cac ma loi: {', '.join(failed)}")
