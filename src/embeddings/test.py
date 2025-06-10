import os
import pickle
from extract_embeddings import extract_all_embeddings
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Đường dẫn tương đối từ src/embeddings
pkl_file = os.path.join(base_dir, "data", "embeddings", "all_embeddings.pkl")

print("Thư mục làm việc hiện tại:", os.getcwd())
print("Đường dẫn file:", os.path.abspath(pkl_file))

with open(pkl_file, "rb") as f:
    data = pickle.load(f)

print("Nội dung file .pkl:")
print(data)