import json
import requests
import pandas as pd
import os
import io

json_file = r"d:\predict-stock\src\data\craw\vn30.json"
output_dir = r"d:\predict-stock\src\data\raw"

# Create output dir if not exists
os.makedirs(output_dir, exist_ok=True)

with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

for code in data.keys():
    url = f"https://m.cafef.vn/du-lieu/Ajax/PageNew/DataHistory/PriceHistory.ashx?Type=EXPORT&Symbol={code}&StartDate=&EndDate=&PageIndex=1&PageSize=20"
    print(f"Downloading {code}...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://s.cafef.vn/'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        try:
            # Read excel from binary memory
            df = pd.read_excel(io.BytesIO(response.content))
            csv_path = os.path.join(output_dir, f"{code}.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved {csv_path}")
        except Exception as e:
            print(f"Error parsing {code}: {e}")
    else:
        print(f"Failed to download {code}: Status Code {response.status_code}")
