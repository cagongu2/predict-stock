# Dự án Dự đoán Giá Cổ phiếu VN30 (Stock Prediction)

Dự án này cung cấp một hệ thống từ việc thu thập dữ liệu lịch sử giá cổ phiếu thuộc nhóm VN30 tại thị trường chứng khoán Việt Nam, cho đến việc xây dựng và huấn luyện mô hình học máy (XGBoost) để dự đoán giá cổ phiếu cho ngày tiếp theo.

## Mục tiêu

Xây dựng một đường ống (pipeline) hoàn chỉnh bao gồm:
1. **Thu thập dữ liệu**: Cào dữ liệu lịch sử giá rổ VN30 đã điều chỉnh bằng thư viện `vnstock`.
2. **Tiền xử lý và Feature Engineering**: Thêm các chỉ báo kỹ thuật như đường trung bình động (EMA, SMA), RSI, MACD để lượng hoá xu hướng và động lượng của thị trường.
3. **Mô hình hoá dự đoán**: Sử dụng mô hình `XGBRegressor` dự báo xu hướng và giá đóng cửa của ngày tiếp theo.

## Cấu trúc thư mục

```text
d:\predict-stock\
├── docs/                                          
│   └── XGBoost for stock trend & prices prediction.md  # Tài liệu lý thuyết về Feature Engineering
├── src/                                           
│   ├── data/                                      
│   │   └── raw-data/                                   # Nơi lưu trữ file CSV dữ liệu thô kéo về
│   ├── craw_vn30.py                                    # Script cào dữ liệu lịch sử chứng khoán VN30
│   ├── vn30.json                                       # Định nghĩa danh sách các mã cổ phiếu rổ VN30
│   └── stock-prediction.ipynb                          # Notebook xử lý dữ liệu và huấn luyện mô hình XGBoost
└── README.md                                           # Tài liệu dự án
```

## Các thành phần chính

### 1. Thu thập dữ liệu (`src/craw_vn30.py` & `src/vn30.json`)
- Đọc danh sách các mã cổ phiếu VN30 từ tệp `vn30.json`.
- Sử dụng API của `vnstock` v3.x (nguồn `VCI`) để lấy dữ liệu giá **đã điều chỉnh** (bao gồm xử lý chia tách, cổ tức) cho từng mã chứng khoán từ `01/01/1990` đến thời điểm hiện tại.
- Xuất dữ liệu dưới định dạng CSV và lưu vào thư mục `src/data/raw-data/`.

### 2. Tiền xử lý & Huấn luyện Mô hình (`src/stock-prediction.ipynb`)
Notebook này bao gồm các bước sau (tham khảo chi tiết tại `docs/XGBoost for stock trend & prices prediction.md`):
- **Lọc dữ liệu**: Tải dữ liệu từ tệp CSV (ví dụ: `FPT.csv`) và lọc lấy dữ liệu từ năm **2022** trở đi để tập trung vào các biến động thị trường gần nhất.
- **Feature Engineering**: Tính toán bộ 8 đặc trưng kỹ thuật cốt lõi:
  - **Xu hướng (Trend)**: `EMA_9`, `SMA_5`, `SMA_10`, `SMA_15`, `SMA_30`.
  - **Động lượng (Momentum)**: `RSI` (14 ngày), `MACD` và `MACD_signal`.
- **Tiền xử lý nâng cao**:
  - **Dịch nhãn (Label Shift)**: Chuyển dữ liệu về bài toán dự báo giá ngày mai (`shift(-1)`).
  - **Chuẩn hóa**: Sử dụng `MinMaxScaler` để đưa các chỉ báo về cùng một thang đo trước khi huấn luyện.
- **Chia tập dữ liệu**: Sử dụng chiến lược **TimeSeriesSplit** (Train 70% / Valid 15% / Test 15%) để ngăn chặn rò rỉ dữ liệu từ tương lai.
- **Dự báo (XGBoost)**: Huấn luyện mô hình `XGBRegressor` với bộ tham số được tối ưu hóa qua GridSearchCV, đạt độ lỗi thấp (MSE ~29.9 cho mã `FPT`).

## Cách sử dụng

**Bước 1: Cập nhật dữ liệu** 
Chạy script để phục hồi và cào toàn bộ dữ liệu lịch sử các mã cổ phiếu rổ VN30:
```bash
cd d:\predict-stock\src
python craw_vn30.py
```
*(Kết quả các file CSV sẽ nằm dưới dạng `src/data/raw-data/<MÃ_CỔ_PHIẾU>.csv`)*

**Bước 2: Chạy huấn luyện Mô hình** 
Mở file `src/stock-prediction.ipynb` bằng Jupyter Notebook, JupyterLab hoặc VSCode.
Cấu hình biến `STOCK_CODE` (ví dụ `STOCK_CODE = 'FPT'`) cho cổ phiếu bạn muốn huấn luyện và bắt đầu chạy (Run all) tuần tự các ô trong notebook để xem trực quan hóa và kết quả đánh giá mô hình.

## Lưu ý về thư viện yêu cầu
Để dự án hoạt động, đảm bảo môi trường Python của bạn có các thư viện sau:
- `pandas`, `numpy`
- `vnstock`
- `xgboost` 
- `scikit-learn`
- `plotly`, `matplotlib` (Để trực quan hóa kết quả)
- `statsmodels` (Cho tính năng time series decomposition)
