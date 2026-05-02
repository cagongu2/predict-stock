# Dự án Dự đoán Giá Cổ phiếu VN30 (Stock Prediction)

Dự án này cung cấp một hệ thống từ việc thu thập dữ liệu lịch sử giá cổ phiếu thuộc nhóm VN30 tại thị trường chứng khoán Việt Nam, cho đến việc xây dựng và huấn luyện mô hình học máy để dự đoán giá cổ phiếu cho những ngày tiếp theo. Dự án hiện tại hỗ trợ cả các mô hình cơ bản (XGBoost) và Deep Learning (LSTM), đi kèm với một giao diện Web trực quan.

## Mục tiêu

Xây dựng một đường ống (pipeline) hoàn chỉnh bao gồm:
1. **Thu thập dữ liệu**: Cào dữ liệu lịch sử giá rổ VN30 bằng thư viện `vnstock` v3 (kết hợp `python-dotenv` để quản lý API Key an toàn).
2. **Tiền xử lý và Feature Engineering**: Thêm các chỉ báo kỹ thuật như đường trung bình động (EMA, SMA), RSI, MACD để lượng hoá xu hướng và động lượng của thị trường.
3. **Mô hình hoá dự đoán (LSTM/XGBoost)**: Xây dựng các mô hình dự báo xu hướng và giá đóng cửa của ngày tiếp theo. Đặc biệt có tính năng dự báo liên tiếp 5 ngày tương lai.
4. **Trực quan hoá bằng Web App**: Xây dựng backend (FastAPI) và frontend (React) để theo dõi các số liệu (MSE, MAPE) và biểu đồ giá trực tiếp.

## Cấu trúc thư mục

```text
d:\predict-stock\
├── docs/                                          
│   ├── XGBoost for stock trend & prices prediction.md  # Tài liệu lý thuyết về XGBoost
│   └── LSTM for stock trend & prices prediction.md     # Tài liệu lý thuyết về kiến trúc mạng LSTM
├── fe/                                                 # Thư mục mã nguồn Frontend (React/Vite)
├── src/                                           
│   ├── data/                                      
│   │   ├── raw-data/                                   # Nơi lưu trữ file CSV dữ liệu thô
│   │   └── outputs/                                    # Nơi lưu kết quả dự đoán, file summary, metrics
│   ├── craw_vn30.py                                    # Script cào dữ liệu lịch sử chứng khoán VN30
│   ├── vn30.json                                       # Định nghĩa danh sách các mã cổ phiếu rổ VN30
│   ├── api.py                                          # Backend API (FastAPI) phục vụ dữ liệu cho Frontend
│   └── stock_prediction_LSTM.ipynb                     # Notebook xử lý dữ liệu và huấn luyện mô hình LSTM
├── .env.example                                        # (Nếu có) File cấu hình biến môi trường mẫu
└── README.md                                           # Tài liệu dự án
```

## Các thành phần chính

### 1. Thu thập dữ liệu (`src/craw_vn30.py`)
- **Nguồn dữ liệu**: Sử dụng nguồn `VCI` qua `vnstock` API.
- **Bảo mật**: Lấy `STOCK_API_KEY` từ file `.env` để không bị giới hạn requests.
- **Đầu ra**: Dữ liệu thô lưu tại `src/data/raw-data/`.

### 2. Tiền xử lý & Huấn luyện LSTM (`src/stock_prediction_LSTM.ipynb`)
- **Feature Engineering**: Tính toán bộ 8 đặc trưng kỹ thuật cốt lõi (SMA, EMA, RSI, MACD).
- **Chuẩn bị dữ liệu**: Xây dựng `StockDataset` sử dụng cơ chế cửa sổ trượt (Sliding Window) với `seq_len = 5`.
- **Huấn luyện**: Xây dựng mạng LSTM bằng PyTorch. Áp dụng Early Stopping.
- **Đánh giá & Lưu kết quả**: Xuất các file `predictions.csv`, `future_5days.csv` và `metrics.json` ra thư mục `src/data/outputs/`. 

### 3. Web Dashboard (FastAPI + React)
- **Backend API**: FastAPI (`src/api.py`) cung cấp các endpoints để trả dữ liệu cho frontend và cung cấp nút bấm để trigger lệnh crawl tự động.
- **Frontend**: Ứng dụng React (`fe/`) với giao diện Glassmorphism, dùng `recharts` để vẽ biểu đồ trực quan và hiển thị kết quả mô hình (MSE, MAPE).

## Cách sử dụng

### Bước 1: Cấu hình môi trường
Tạo file `.env` ở thư mục gốc và điền API Key của vnstock:
```env
STOCK_API_KEY=vnstock_của_bạn_vào_đây
```

### Bước 2: Huấn luyện Mô hình
Mở `src/stock_prediction_LSTM.ipynb` và chạy từng cell. Bạn có thể tự động chạy qua toàn bộ danh sách CSV trong thư mục `raw-data` để tự động huấn luyện cho toàn bộ danh sách VN30. Kết quả sau khi chạy sẽ được lưu vào `src/data/outputs/`.

### Bước 3: Chạy Web Dashboard
Bạn cần mở 2 cửa sổ Terminal độc lập:

**Terminal 1 (Chạy Backend - FastAPI):**
```bash
python src/api.py
```

**Terminal 2 (Chạy Frontend - React):**
```bash
cd fe
npm install
npm run dev
```
Sau đó, giữ `Ctrl` và click vào đường link `http://localhost:5173` hiển thị trên Terminal 2 để mở ứng dụng Web.

## Yêu cầu thư viện (Dependencies)
Để dự án hoạt động, hãy đảm bảo môi trường Python của bạn có các thư viện sau:
- **Cơ bản**: `pandas`, `numpy`, `python-dotenv`
- **Lấy dữ liệu**: `vnstock`
- **Mô hình học máy**: `torch` (PyTorch), `xgboost`, `scikit-learn`
- **Trực quan hoá (Notebook)**: `plotly`, `matplotlib`
- **Backend Web**: `fastapi`, `uvicorn`
