# Dự đoán xu hướng và giá cổ phiếu bằng LSTM

> Tài liệu này trình bày chi tiết quá trình chuẩn bị dữ liệu và xây dựng mô hình Deep Learning sử dụng mạng **Long Short-Term Memory (LSTM)** bằng PyTorch để dự đoán giá đóng cửa của cổ phiếu ngày tiếp theo. Mô hình cũng kết hợp với các chỉ báo kỹ thuật để tăng cường khả năng học dữ liệu chuỗi thời gian.

---

## 1. Tổng quan Pipeline

Quy trình xử lý dữ liệu được thiết kế tương tự XGBoost nhưng bổ sung thêm bước tạo chuỗi (Sequences) đặc trưng cho mạng LSTM:

```text
Dữ liệu thô (OHLCV)
        ↓
Lọc dữ liệu (Từ năm 2022)
        ↓
Kỹ thuật đặc trưng (Feature Engineering)
   ├── Moving Averages: EMA_9, SMA_5/10/15/30
   ├── RSI (14 ngày)
   └── MACD + MACD_signal
        ↓
Dịch nhãn (Label Shifting): Close[t] → Close[t+1]
        ↓
Chuẩn hóa dữ liệu (MinMaxScaler cho cả X và y)
        ↓
Tạo chuỗi (Sliding Window): Dùng seq_len ngày (vd: 5 ngày) để dự đoán 1 ngày
        ↓
Chia tập dữ liệu theo thời gian: Train / Validation / Test
        ↓
Huấn luyện (Training) & Đánh giá mạng LSTM (PyTorch)
```

---

## 2. Dữ liệu đầu vào

Mô hình sử dụng dữ liệu lịch sử các mã cổ phiếu trong rổ VN30 (lưu tại `src/data/raw-data/`).

```python
STOCK_DIRECTORY = 'src/data/raw-data'
START_YEAR      = 2022

# Đọc và lọc dữ liệu từ năm 2022 để tập trung vào xu hướng gần đây
df = pd.read_csv(csv_path)
df['time'] = pd.to_datetime(df['time'])
df = df[df['time'].dt.year >= START_YEAR].copy()
```

Dữ liệu bao gồm các cột cơ bản: **Time (Ngày)**, **Open**, **High**, **Low**, **Close**, **Volume**.

---

## 3. Kỹ thuật đặc trưng (Feature Engineering)

Mặc dù LSTM có khả năng tự học đặc trưng chuỗi thời gian, việc cung cấp thêm các chỉ báo kỹ thuật giúp mô hình hội tụ nhanh và dự đoán chính xác hơn. Các chỉ báo tương tự như XGBoost:

1.  **Chỉ báo xu hướng:** EMA_9, SMA_5, SMA_10, SMA_15, SMA_30.
2.  **Chỉ báo động lượng:** RSI (14 ngày).
3.  **Tín hiệu phân kỳ/hội tụ:** MACD, MACD_signal.

```python
# Các đặc trưng được tính toán và target (Close) được dịch lên 1 hàng để dự đoán ngày mai
if shift_target:
    df['close'] = df['close'].shift(-1)   # target = giá đóng cửa ngày sau
    df = df.iloc[33:-1]                   # bỏ NaN đầu & dòng cuối do tính toán SMA/MACD
```

---

## 4. Tiền xử lý dữ liệu cho LSTM

### 4.1 Chuẩn hóa dữ liệu (Scaling)
LSTM rất nhạy cảm với thang đo của dữ liệu, do đó toàn bộ đầu vào `X` và đầu ra `y` cần được đưa về cùng một phạm vi (thường là [0, 1]) bằng `MinMaxScaler`.

### 4.2 Tạo chuỗi (Sliding Window / Sequence Generation)
Khác với các mô hình Machine Learning cổ điển, LSTM yêu cầu dữ liệu đầu vào ở định dạng 3 chiều: `(Batch Size, Sequence Length, Features)`.

*   **Sequence Length (`seq_len`):** Ở đây cấu hình mặc định là `5` ngày. Tức là mô hình sẽ nhìn vào đặc trưng của 5 ngày liên tiếp để dự đoán giá của ngày thứ 6.

```python
class StockDataset(Dataset):
    def __getitem__(self, idx):
        return (self.X[idx : idx + self.seq_len],   # (seq_len, features)
                self.y[idx + self.seq_len])         # scalar (giá trị thực của ngày tiếp theo)
```

### 4.3 Chia tập dữ liệu
Dữ liệu vẫn được chia theo trình tự thời gian (không trộn ngẫu nhiên):
*   **Train:** Phần lớn dữ liệu để học.
*   **Validation:** Đánh giá trong quá trình huấn luyện và sử dụng Early Stopping (dừng sớm nếu mô hình không cải thiện).
*   **Test:** Phần cuối cùng dùng để đánh giá mô hình khách quan nhất.

---

## 5. Mô hình LSTM & Huấn luyện

Mô hình được xây dựng bằng PyTorch với cấu trúc cơ bản gồm:
1.  **Lớp LSTM:** Lớp học chính với `hidden_size=64`, `num_layers=2` và cơ chế Dropout để tránh Overfitting.
2.  **Lớp Linear (Fully Connected):** Nối tiếp lớp LSTM cuối cùng để tính toán giá trị liên tục duy nhất (Giá đóng cửa).

### Cấu hình siêu tham số (Hyperparameters):
*   **SEQ_LEN:** 5
*   **HIDDEN_SIZE:** 64
*   **NUM_LAYERS:** 2
*   **EPOCHS:** 100
*   **BATCH_SIZE:** 32
*   **LR (Learning Rate):** 1e-3
*   **Early Stopping (PATIENCE):** 10 epochs.

Quá trình huấn luyện sử dụng hàm Loss là `MSELoss` (Mean Squared Error) và bộ tối ưu hóa `Adam`.

---

## 6. Đánh giá Kết quả & Dự báo Tương lai

Mô hình được đánh giá trên tập Test bằng các độ đo:
*   **MSE (Mean Squared Error):** Độ lỗi bình phương trung bình.
*   **MAPE (Mean Absolute Percentage Error):** Sai số phần trăm tuyệt đối trung bình, cho phép đánh giá trực quan hơn. 

Ví dụ với mã cổ phiếu **VRE**, mô hình đạt được `MAPE = 0.0503 (5.03%)`, cho thấy mô hình dự đoán khá tốt. 
Hơn nữa, mô hình hỗ trợ dự báo giá liên tiếp nhiều ngày trong tương lai bằng cách đưa kết quả dự đoán của ngày hôm nay vào chuỗi đầu vào để dự báo cho ngày mai (hàm `predict_future`).

---

## Kết luận

Mạng LSTM cung cấp một giải pháp mạnh mẽ để theo dõi tính phụ thuộc thời gian (temporal dependencies) của giá cổ phiếu. Việc kết hợp với các chỉ báo kỹ thuật truyền thống giúp mô hình không chỉ nắm bắt tính chất chuỗi mà còn các tín hiệu giao dịch quan trọng, mở ra hướng ứng dụng tin cậy để dự đoán chứng khoán trong thực tiễn.
