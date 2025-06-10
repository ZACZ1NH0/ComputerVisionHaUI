# import os
# import numpy as np
# import pickle
# from deepface import DeepFace
# from tqdm import tqdm
# import sys
# base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(f'{base_dir}/src')  # Thêm đúng thư mục chứa module
# from config.config import PROCESSED_PATH, EMBEDDINGS_PATH
#
# def extract_all_embeddings(processed_root=PROCESSED_PATH, save_path=EMBEDDINGS_PATH, model_name="Facenet"):
#
#     all_embeddings = []
#     labels = []
#
#     # Danh sách thư mục người dùng
#     persons = [p for p in os.listdir(processed_root) if os.path.isdir(os.path.join(processed_root, p))]
#     for person in tqdm(persons, desc="🔍 Đang xử lý người"):
#         person_path = os.path.join(processed_root, person)
#         image_files = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
#
#         for img_file in tqdm(image_files, desc=f"📸 {person}", leave=False):
#             img_path = os.path.join(person_path, img_file)
#             try:
#                 embedding = DeepFace.represent(img_path=img_path, model_name=model_name)[0]["embedding"]
#                 all_embeddings.append(embedding)
#                 labels.append(person)
#             except Exception as e:
#                 print(f"⚠️ Lỗi khi xử lý ảnh {img_file} của {person}: {e}")
#
#     # Lưu kết quả
#     with open(save_path, "wb") as f:
#         pickle.dump({"embeddings": all_embeddings, "labels": labels}, f)
#     print(f"\n✅ Đã lưu {len(all_embeddings)} embeddings vào '{save_path}'")
#
# # Cho phép chạy trực tiếp
# if __name__ == "__main__":
#     extract_all_embeddings()
#
import os
import numpy as np
import pickle
from tqdm import tqdm
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Thêm đường dẫn đến thư mục chứa config
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(f'{base_dir}/src')
from config.config import PROCESSED_PATH, EMBEDDINGS_PATH


# Hàm tạo mô hình CNN (sử dụng ResNet50 làm ví dụ)
def build_cnn_model(input_shape=(224, 224, 3), embedding_size=128):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(embedding_size, activation=None)(x)  # Embedding layer
    model = Model(inputs=base_model.input, outputs=x)

    # Đóng băng các layer của ResNet50 nếu không muốn fine-tune
    for layer in base_model.layers:
        layer.trainable = False
    return model


# Hàm tiền xử lý ảnh
def preprocess_image(img_path, target_size=(224, 224)):
    try:
        img = load_img(img_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Thêm batch dimension
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)  # Chuẩn hóa cho ResNet
        return img_array
    except Exception as e:
        print(f"⚠️ Lỗi khi tiền xử lý ảnh {img_path}: {e}")
        return None


# Hàm trích xuất embeddings
def extract_all_embeddings(processed_root=PROCESSED_PATH, save_path=EMBEDDINGS_PATH):
    # Khởi tạo mô hình CNN
    model = build_cnn_model()

    all_embeddings = []
    labels = []

    # Danh sách thư mục người dùng
    persons = [p for p in os.listdir(processed_root) if os.path.isdir(os.path.join(processed_root, p))]
    for person in tqdm(persons, desc="🔍 Đang xử lý người"):
        person_path = os.path.join(processed_root, person)
        image_files = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for img_file in tqdm(image_files, desc=f"📸 {person}", leave=False):
            img_path = os.path.join(person_path, img_file)
            img_array = preprocess_image(img_path)
            if img_array is not None:
                try:
                    embedding = model.predict(img_array)[0]  # Trích xuất embedding
                    all_embeddings.append(embedding)
                    labels.append(person)
                except Exception as e:
                    print(f"⚠️ Lỗi khi xử lý ảnh {img_file} của {person}: {e}")

    # Lưu kết quả
    with open(save_path, "wb") as f:
        pickle.dump({"embeddings": all_embeddings, "labels": labels}, f)
    print(f"\n✅ Đã lưu {len(all_embeddings)} embeddings vào '{save_path}'")


# Cho phép chạy trực tiếp
if __name__ == "__main__":
    extract_all_embeddings()