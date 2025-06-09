# import numpy as np
# from numpy.linalg import norm
#
# def cosine_similarity(vec1, vec2):
#     return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
#
# def load_embeddings(npz_file):
#     return dict(np.load(npz_file, allow_pickle=True))
#
# def find_most_similar(embedding, embeddings_dict, threshold=0.7):
#     best_match = None
#     highest_sim = -1
#
#     for filename, saved_emb in embeddings_dict.items():
#         sim = cosine_similarity(embedding, saved_emb)
#         if sim > highest_sim:
#             best_match = filename
#             highest_sim = sim
#
#     if highest_sim >= threshold:
#         return best_match, highest_sim
#     else:
#         return "Unknown", highest_sim
#
# # Ví dụ sử dụng
# if __name__ == "__main__":
#     from deepface import DeepFace
#
#     # Load embeddings đã lưu
#     embeddings = load_embeddings("D:/ComputerVisionHaUI_copy/data/embeddings/person1_embeddings.npz")
#
#     # Trích xuất embedding từ ảnh mới
#     test_img = "D:\ComputerVisionHaUI_copy\data\processed\person1/5.jpg"
#     rep = DeepFace.represent(img_path=test_img, model_name='Facenet', enforce_detection=False)[0]["embedding"]
#
#     # So sánh
#     result, score = find_most_similar(rep, embeddings)
#     print(f"Kết quả nhận diện: {result} (Độ tương đồng: {score:.4f})")
#
#
#
import os
import numpy as np
from numpy.linalg import norm
from deepface import DeepFace

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def load_embeddings(npz_file):
    return dict(np.load(npz_file, allow_pickle=True))

def find_most_similar(embedding, embeddings_dict, threshold=0.7):
    best_match = None
    highest_sim = -1

    for filename, saved_emb in embeddings_dict.items():
        sim = cosine_similarity(embedding, saved_emb)
        if sim > highest_sim:
            best_match = filename
            highest_sim = sim

    if highest_sim >= threshold:
        return best_match, highest_sim
    else:
        return "Unknown", highest_sim

# === Xử lý chính: KHÔNG nằm trong if __name__ == "__main__" ===
person_name = input("Nhập tên người dùng [VD: person1(_embeddings_npz)]: ").strip()
if not person_name:
    print("[X] Tên không hợp lệ.")
    exit()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

embedding_path = os.path.join(project_root, "data", "embeddings", f"{person_name}_embeddings.npz")
if not os.path.exists(embedding_path):
    print(f"[X] Không tìm thấy file embedding: {embedding_path}")
    exit()

image_dir = os.path.join(project_root, "data", "processed", person_name)
if not os.path.exists(image_dir):
    print(f"[X] Thư mục ảnh không tồn tại: {image_dir}")
    exit()

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
if not image_files:
    print(f"[X] Không có ảnh trong thư mục: {image_dir}")
    exit()

test_img_path = os.path.join(image_dir, image_files[0])
print(f"[i] Đang sử dụng ảnh: {test_img_path}")

# Load và xử lý
embeddings = load_embeddings(embedding_path)

try:
    rep = DeepFace.represent(img_path=test_img_path, model_name='Facenet', enforce_detection=False)[0]["embedding"]
except Exception as e:
    print(f"[X] Lỗi khi trích xuất embedding: {e}")
    exit()

result, score = find_most_similar(rep, embeddings)
print(f"\nKết quả nhận diện: {result}")
print(f"Độ tương đồng: {score:.4f}")

# if __name__ == "__main__":
#     run_face_recognition(person_name)

