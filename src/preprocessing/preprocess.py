# import cv2
# import os
# from ultralytics import YOLO
# import sys
# # === Lấy tên người dùng ===
# person_name = input("Nhập tên người dùng đã lưu ảnh ở data/raw: ").strip()
# if not person_name:
#     print("[X] Tên không hợp lệ.")
#     exit()
# base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(f'{base_dir}/src')
# # === Thiết lập đường dẫn ===
# input_folder = os.path.join(base_dir, 'data', 'raw', person_name)
# output_folder = os.path.join(base_dir, 'data', 'processed', person_name)
# os.makedirs(output_folder, exist_ok=True)

# # === Load mô hình YOLOv8 ===
# model = YOLO('yolov8n.pt')  # dùng model nhẹ để demo

# def is_valid_image(image_path):
#     try:
#         img = cv2.imread(image_path)
#         return img is not None and img.shape[0] > 0 and img.shape[1] > 0
#     except:
#         return False

# def detect_and_crop_face(image, conf_threshold=0.5, iou_threshold=0.5):
#     results = model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
#     boxes = results[0].boxes.xyxy
#     if len(boxes) > 0:
#         # chọn box có diện tích lớn nhất
#         max_area = 0
#         best_box = None
#         for box in boxes:
#             x1, y1, x2, y2 = map(int, box)
#             area = (x2 - x1) * (y2 - y1)
#             if area > max_area:
#                 max_area = area
#                 best_box = (x1, y1, x2, y2)
#         if best_box:
#             x1, y1, x2, y2 = best_box
#             # Crop vùng trung tâm của box (ví dụ: 60% vùng giữa)
#             w = x2 - x1
#             h = y2 - y1
#             cx = x1 + w // 2
#             cy = y1 + h // 2
#             crop_w = int(w * 0.6)
#             crop_h = int(h * 0.6)
#             new_x1 = max(cx - crop_w // 2, 0)
#             new_y1 = max(cy - crop_h // 2, 0)
#             new_x2 = min(cx + crop_w // 2, image.shape[1])
#             new_y2 = min(cy + crop_h // 2, image.shape[0])
#             return image[new_y1:new_y2, new_x1:new_x2]
#     return None

# def resize_face(face_img, size=(160, 160)):
#     return cv2.resize(face_img, size)

# # === Duyệt và xử lý ảnh ===
# image_list = os.listdir(input_folder)
# image_list = [f for f in image_list if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# for idx, filename in enumerate(image_list):
#     img_path = os.path.join(input_folder, filename)

#     if not is_valid_image(img_path):
#         print(f"[X] Ảnh lỗi: {filename}")
#         continue

#     image = cv2.imread(img_path)
#     face = detect_and_crop_face(image)

#     if face is None:
#         print(f"[!] Không tìm thấy khuôn mặt trong: {filename}")
#         continue

#     resized_face = resize_face(face)
#     save_path = os.path.join(output_folder, f'{idx}.jpg')
#     cv2.imwrite(save_path, resized_face)
#     print(f"[✓] Đã xử lý: {filename} → {save_path}")
