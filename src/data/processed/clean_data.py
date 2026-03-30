import os
import pandas as pd

input_dir = r"d:\predict-stock\src\data\raw-data"
output_dir = r"d:\predict-stock\src\data\processed\cleaned"

# Tạo thư mục chứa dữ liệu đã xử lý nếu chưa có
os.makedirs(output_dir, exist_ok=True)

drop_columns = ["KLThoaThuan", "GtThoaThuan", "GiaDieuChinh", "ThayDoi"]

# Duyệt qua từng file trong danh sách
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_dir, filename)
        
        try:
            # Đọc dữ liệu
            df = pd.read_csv(file_path)
            
            # Loại bỏ các cột không cần thiết
            cols_to_drop = [col for col in drop_columns if col in df.columns]
            df.drop(columns=cols_to_drop, inplace=True)
            
            # Lưu file mới 
            output_path = os.path.join(output_dir, filename)
            df.to_csv(output_path, index=False)
            
            print(f"Đã xử lý và lưu: {output_path}")
        except Exception as e:
            print(f"Lỗi khi xử lý file {filename}: {e}")
