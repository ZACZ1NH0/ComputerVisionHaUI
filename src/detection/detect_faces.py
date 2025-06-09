import cv2
import numpy as np
from ultralytics import YOLO
import time
import os

from src.recognition.recognize_faces import load_reference_embeddings, recognize_faces


MODEL_PATH = 'yolov11n-face.pt'

try:
    print(f"Đang tải mô hình YOLO từ '{MODEL_PATH}', vui lòng đợi...")
    # Khởi tạo đối tượng YOLO từ file model. Đây là bước tốn thời gian nhất và chỉ nên làm một lần.
    YOLO_MODEL = YOLO(MODEL_PATH)
    print("Tải mô hình YOLO thành công.")
except Exception as e:
    print(f"Lỗi nghiêm trọng khi tải mô hình: {e}")
    print("Hãy chắc chắn rằng tệp model có trong thư mục. Chương trình sẽ không thể hoạt động.")
    YOLO_MODEL = None


def detect_and_process_faces(image_input, flag: int):
    if YOLO_MODEL is None:
        return None, "Lỗi: Mô hình YOLO chưa được tải thành công."

    if flag not in [1, 2, 3, 4]:
        return None, f"Lỗi: Tham số cờ 'flag' không hợp lệ."

    if isinstance(image_input, str):
        img_original = cv2.imread(image_input)
        if img_original is None:
            return None, f"Lỗi: Không thể đọc ảnh từ '{image_input}'."
    elif isinstance(image_input, np.ndarray):
        img_original = image_input.copy()
    else:
        return None, "Lỗi: Đầu vào không phải là đường dẫn hoặc mảng NumPy."

    results = YOLO_MODEL(img_original, verbose=False)
    boxes = results[0].boxes.cpu().numpy()


    dict_coords = {}
    dict_cropped_faces = {}
    image_with_boxes = img_original.copy() if flag in [2, 4] else None


    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        person_key = f"person_{i + 1}"

        if flag in [1, 2, 4]:
            dict_coords[person_key] = np.array([x1, y1, x2, y2])
        if flag in [2, 4]:
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if flag in [3, 4]:
            cropped_face = img_original[max(0, y1):y2, max(0, x1):x2]
            dict_cropped_faces[person_key] = cropped_face

    if flag == 1: return dict_coords
    if flag == 2: return dict_coords, image_with_boxes
    if flag == 3: return dict_cropped_faces
    if flag == 4: return dict_coords, image_with_boxes, dict_cropped_faces


def process_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở file video tại '{video_path}'")
        return

    print("Đang xử lý video... Nhấn 'q' trên cửa sổ video để thoát.")
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Đã xử lý xong video.")
            break

        result = detect_and_process_faces(frame, flag=2)

        if isinstance(result, tuple):
            _, drawn_frame = result
        else:
            drawn_frame = frame

        processing_time = time.time() - start_time
        fps = 1 / processing_time if processing_time > 0 else 0

        cv2.putText(drawn_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Video Face Detection', drawn_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# def process_webcam():
#     """Hàm ứng dụng: Dùng hàm `detect_and_process_faces` để xử lý hình ảnh từ webcam."""
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Lỗi: Không thể kết nối với webcam.")
#         return
#
#     print("Đã kết nối webcam. Nhấn 'q' trên cửa sổ video để thoát.")
#     while True:
#         start_time = time.time()
#         ret, frame = cap.read()
#         if not ret:
#             print("Lỗi: Mất kết nối với webcam.")
#             break
#
#         frame = cv2.flip(frame, 1)
#
#         result = detect_and_process_faces(frame, flag=2)
#
#         if isinstance(result, tuple):
#             _, drawn_frame = result
#         else:
#             drawn_frame = frame
#
#         processing_time = time.time() - start_time
#         fps = 1 / processing_time if processing_time > 0 else 0
#         cv2.putText(drawn_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#
#         cv2.imshow('Webcam Face Detection', drawn_frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()

def process_webcam():
    """Kết hợp phát hiện + nhận diện khuôn mặt realtime từ webcam"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Lỗi: Không thể kết nối với webcam.")
        return

    print("Webcam đang chạy... Nhấn 'q' để thoát.")

    reference_embeddings = load_reference_embeddings()

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Lỗi: Mất kết nối với webcam.")
            break

        frame = cv2.flip(frame, 1)

        result = detect_and_process_faces(frame, flag=4)
        if result is None:
            continue

        coords_dict, frame_with_boxes, faces_dict = result

        recog_results = recognize_faces(faces_dict, reference_embeddings)

        # Vẽ kết quả lên khung
        for person_id, (name, score) in recog_results.items():
            if person_id not in coords_dict:
                continue

            x1, y1, x2, y2 = coords_dict[person_id]
            label = f"{name} ({score:.1f}%)" if name != "unknown" else "Unknown"

            # Khung xanh + nhãn
            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_with_boxes, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Tính FPS
        processing_time = time.time() - start_time
        fps = 1 / processing_time if processing_time > 0 else 0
        cv2.putText(frame_with_boxes, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Webcam Face Recognition", frame_with_boxes)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    print("NHẬN DIỆN KHUÔN MẶT TỪ WEBCAM")
    if YOLO_MODEL is None:
        print("Mô hình chưa được tải. Vui lòng kiểm tra lại đường dẫn model.")
        return

    process_webcam()


if __name__ == "__main__":
    main()


