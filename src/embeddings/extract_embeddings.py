import os
import numpy as np
from deepface import DeepFace

person_name = input("Nhập tên người dùng đã lưu ảnh ở data/processed: ").strip()
if not person_name:
    print("[X] Tên không hợp lệ.")
    exit()

# === Thiết lập đường dẫn ===
# Sử dụng đường dẫn tuyệt đối dựa theo vị trí file
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
input_folder = os.path.join(project_root, 'data', 'processed', person_name)
output_folder = os.path.join(project_root, 'data', 'embeddings', f"{person_name}_embeddings.npz")


def extract_embeddings_from_folder(input_folder, output_npz_path, model_name="Facenet"):
    if not os.path.exists(input_folder):
        print(f"[!] Thư mục không tồn tại: {input_folder}")
        return

    embeddings = {}
    success_count = 0
    failed_count = 0

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(input_folder, filename)
        try:
            results = DeepFace.represent(img_path=image_path, model_name=model_name, enforce_detection=False)
            if results:
                emb = np.array(results[0]["embedding"])
                embeddings[filename] = emb
                success_count += 1
                print(f"[✓] {filename}")
        except Exception as e:
            failed_count += 1
            print(f"[✗] {filename} — Lỗi: {e}")

    if embeddings:
        os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
        np.savez(output_npz_path, **embeddings)
        print(f"\n[✓] Đã lưu {success_count} embedding vào: {output_npz_path}")
        if failed_count > 0:
            print(f"[!] Bỏ qua {failed_count} ảnh do lỗi.")
    else:
        print("[!] Không có embedding nào được tạo.")

if __name__ == "__main__":
    extract_embeddings_from_folder(input_folder, output_folder, model_name="Facenet")

