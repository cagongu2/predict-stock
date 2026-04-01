# Dự đoán xu hướng và giá cổ phiếu bằng XGBoost

> Tài liệu này trình bày chi tiết quá trình chuẩn bị dữ liệu và xây dựng mô hình `XGBRegressor` để dự đoán giá đóng cửa của cổ phiếu ngày tiếp theo, sử dụng các chỉ báo kỹ thuật làm đặc trưng (features).

---

## 1. Tổng quan Pipeline

Quy trình xử lý dữ liệu được thiết kế để chuyển đổi dữ liệu chuỗi thời gian thô thành dạng bảng (tabular) phù hợp cho mô hình XGBoost:

```
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
Xử lý giá trị thiếu (NaN) & Chuẩn hóa (Scaling)
        ↓
Chia tập dữ liệu (TimeSeriesSplit): Train 70% / Validation 15% / Test 15%
        ↓
Huấn luyện & Đánh giá mô hình XGBoost
```

---

## 2. Dữ liệu đầu vào

Mô hình sử dụng dữ liệu lịch sử của cổ phiếu `FPT` (hoặc các mã khác trong danh mục VN30) được lưu trữ tại `src/data/raw-data/`.

```python
STOCK_CODE = 'FPT'
df = pd.read_csv(os.path.join(STOCK_DIRECTORY, STOCK_CODE + '.csv'))
df['time'] = pd.to_datetime(df['time'])

# Lọc dữ liệu từ năm 2022 để tập trung vào xu hướng gần đây
df = df[df['time'].dt.year >= 2022].copy()
df.index = range(len(df))
```

Dữ liệu bao gồm các cột cơ bản: **Time (Ngày)**, **Open**, **High**, **Low**, **Close**, **Volume**.

---

## 3. Kỹ thuật đặc trưng (Feature Engineering)

XGBoost là một mô hình học máy dạng bảng, nó không tự nhận biết được tính tuần tự của thời gian. Do đó, chúng ta cần tính toán các chỉ báo kỹ thuật để "mã hóa" thông tin về xu hướng và động lượng vào các đặc trưng đầu vào.

### 3.1 Chỉ báo xu hướng (Moving Averages)

Sử dụng các đường trung bình động với các khung thời gian khác nhau (5, 10, 15, 30 ngày) để bắt trọn xu hướng ngắn và trung hạn.

```python
df['EMA_9'] = df['Close'].ewm(9).mean().shift()
df['SMA_5'] = df['Close'].rolling(5).mean().shift()
df['SMA_10'] = df['Close'].rolling(10).mean().shift()
df['SMA_15'] = df['Close'].rolling(15).mean().shift()
df['SMA_30'] = df['Close'].rolling(30).mean().shift()
```

> **Lưu ý quan trọng:** Dùng `.shift()` để tránh rò rỉ dữ liệu (*data leakage*). Giá trị đặc trưng tại ngày $t$ chỉ được phép sử dụng thông tin của các ngày từ $t-1$ trở về trước.

### 3.2 Chỉ báo động lượng (RSI & MACD)

*   **RSI (Relative Strength Index):** Đo lường tốc độ và sự thay đổi của biến động giá để xác định các điều kiện quá mua (overbought) hoặc quá bán (oversold).
*   **MACD (Moving Average Convergence Divergence):** Cho thấy mối liên hệ giữa hai đường trung bình động của giá cổ phiếu.

```python
# RSI Calculation
def relative_strength_idx(df, n=14):
    close = df['Close']
    delta = close.diff()
    pricesUp = delta.copy(); pricesUp[pricesUp < 0] = 0
    pricesDown = delta.copy(); pricesDown[pricesDown > 0] = 0
    rollUp = pricesUp.rolling(n).mean()
    rollDown = pricesDown.abs().rolling(n).mean()
    rs = rollUp / rollDown
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

df['RSI'] = relative_strength_idx(df).fillna(0)

# MACD Calculation
EMA_12 = pd.Series(df['Close'].ewm(span=12, min_periods=12).mean())
EMA_26 = pd.Series(df['Close'].ewm(span=26, min_periods=26).mean())
df['MACD'] = pd.Series(EMA_12 - EMA_26)
df['MACD_signal'] = pd.Series(df.MACD.ewm(span=9, min_periods=9).mean())
```

---

## 4. Tiền xử lý dữ liệu

### 4.1 Dịch nhãn (Label Shifting)

Để dự đoán giá ngày mai, chúng ta dịch cột giá đóng cửa (`Close`) ngược lên một hàng.

```python
df['Close'] = df['Close'].shift(-1)
```

Lúc này, mỗi hàng dữ liệu chứa các đặc trưng của ngày hôm nay và mục tiêu (target) là giá của ngày mai.

### 4.2 Chia tập dữ liệu

Sử dụng phương pháp chia theo thời gian (không trộn ngẫu nhiên) để đảm bảo tính khách quan:
*   **Train (70%):** Dùng để huấn luyện mô hình.
*   **Validation (15%):** Dùng để tinh chỉnh tham số (Hyperparameter tuning).
*   **Test (15%):** Dùng để đánh giá hiệu năng cuối cùng trên dữ liệu "chưa từng thấy".

---

## 5. Mô hình XGBoost & Kết quả

Mô hình `XGBRegressor` được huấn luyện với bộ tham số tối ưu (tìm được qua GridSearchCV hoặc RandomSearch).

### Đặc trưng đầu vào (8 Features):
1.  **EMA_9**: Xu hướng ngắn hạn (trọng số lũy thừa).
2.  **SMA_5/10/15/30**: Các đường trung bình động đơn giản.
3.  **RSI**: Động lượng thị trường.
4.  **MACD**: Tín hiệu phân kỳ/hội tụ.
5.  **MACD_signal**: Đường tín hiệu hỗ trợ.

### Đánh giá hiệu năng:
Dựa trên kết quả thực tế cho mã `FPT`, mô hình đạt được độ lỗi trung bình bình phương (**Mean Squared Error - MSE**) xấp xỉ **29.9**. Điều này cho thấy mô hình có khả năng bám sát biến động giá thực tế khá tốt trong điều kiện thị trường ổn định.

---

## Kết luận

Việc kết hợp đa dạng các chỉ báo kỹ thuật (Xu hướng + Động lượng) giúp mô hình XGBoost nắm bắt được các quy luật phức tạp của thị trường chứng khoán Việt Nam giai đoạn 2022-2025. Tài liệu này đóng vai trò là hướng dẫn chuẩn để triển khai cho các mã cổ phiếu khác trong VN30.