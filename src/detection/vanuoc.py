# --- PHẦN 1: IMPORT CÁC THƯ VIỆN CẦN THIẾT ---
import cv2  # Thư viện OpenCV, chuyên dùng cho các tác vụ xử lý ảnh và video.
import numpy as np  # Thư viện NumPy, dùng để làm việc hiệu quả với dữ liệu dạng mảng (ảnh cũng là một dạng mảng).
from ultralytics import YOLO  # Import lớp YOLO từ thư viện của Ultralytics để sử dụng mô hình.
import time  # Thư viện Time, dùng để đo thời gian xử lý và tính toán FPS (khung hình trên giây).

# --- PHẦN 2: TẢI MODEL VÀ CÁC THIẾT LẬP BAN ĐẦU ---

# Đường dẫn tới file model đã được huấn luyện. Bạn có thể thay đổi đường dẫn này.
# Gợi ý: Dùng model 'n' (nano) như 'yolov8n-face.pt' sẽ cho tốc độ nhanh hơn trên video/webcam.
MODEL_PATH = 'yolov11n-face.pt'

# Sử dụng khối try...except để bắt lỗi nếu không tìm thấy file model hoặc file bị hỏng.
try:
    # In thông báo cho người dùng biết quá trình tải model đang diễn ra.
    print(f"Đang tải mô hình YOLO từ '{MODEL_PATH}', vui lòng đợi...")
    # Khởi tạo đối tượng YOLO từ file model. Đây là bước tốn thời gian nhất và chỉ nên làm một lần.
    YOLO_MODEL = YOLO(MODEL_PATH)
    # Thông báo khi tải thành công.
    print("Tải mô hình YOLO thành công.")
except Exception as e:
    # Nếu có lỗi (ví dụ: sai đường dẫn, file hỏng), in ra thông báo lỗi chi tiết.
    print(f"Lỗi nghiêm trọng khi tải mô hình: {e}")
    print("Hãy chắc chắn rằng tệp model có trong thư mục. Chương trình sẽ không thể hoạt động.")
    # Gán YOLO_MODEL là None để các hàm sau biết rằng model chưa sẵn sàng.
    YOLO_MODEL = None


# --- PHẦN 3: CÁC HÀM XỬ LÝ ---

def detect_and_process_faces(image_input, flag: int):
    """
    Hàm cốt lõi: Phát hiện và xử lý các khuôn mặt trong một ảnh hoặc một khung hình.

    Args:
        image_input (str hoặc np.ndarray): Đầu vào có thể là:
                                           - Một chuỗi (str) chứa đường dẫn tới file ảnh.
                                           - Một mảng NumPy (np.ndarray) đại diện cho ảnh (ví dụ: một khung hình từ video).
        flag (int): Một con số để điều khiển chức năng và kết quả trả về.
                    - 1: Chỉ lấy tọa độ các khuôn mặt.
                    - 2: Lấy tọa độ VÀ ảnh đã được vẽ các ô vuông quanh khuôn mặt.
                    - 3: Chỉ lấy các ảnh khuôn mặt đã được cắt ra.
                    - 4: Lấy tất cả các kết quả trên.
    Returns:
        Kết quả trả về phụ thuộc vào 'flag'. Nếu có lỗi, trả về (None, "Thông báo lỗi").
    """
    # --- Bước 3.1: Kiểm tra các điều kiện đầu vào ---

    # Kiểm tra xem model đã được tải thành công ở bước trên chưa.
    if YOLO_MODEL is None:
        return None, "Lỗi: Mô hình YOLO chưa được tải thành công."

    # Kiểm tra xem tham số 'flag' có hợp lệ không.
    if flag not in [1, 2, 3, 4]:
        return None, f"Lỗi: Tham số cờ 'flag' không hợp lệ."

    # Kiểm tra loại dữ liệu đầu vào.
    if isinstance(image_input, str):  # Nếu đầu vào là một chuỗi (đường dẫn file)
        img_original = cv2.imread(image_input)  # Dùng OpenCV để đọc ảnh từ đường dẫn.
        if img_original is None:  # Nếu không đọc được ảnh (ví dụ: đường dẫn sai).
            return None, f"Lỗi: Không thể đọc ảnh từ '{image_input}'."
    elif isinstance(image_input, np.ndarray):  # Nếu đầu vào đã là một ảnh (mảng NumPy).
        img_original = image_input.copy()  # Tạo một bản sao để không làm thay đổi ảnh gốc.
    else:  # Nếu đầu vào không phải chuỗi hay mảng NumPy.
        return None, "Lỗi: Đầu vào không phải là đường dẫn hoặc mảng NumPy."

    # --- Bước 3.2: Thực hiện nhận diện ---

    # Đây là dòng lệnh quan trọng nhất: đưa ảnh vào model để xử lý.
    # verbose=False để model không in ra các thông tin thừa trong quá trình chạy.
    results = YOLO_MODEL(img_original, verbose=False)
    # Lấy kết quả các hộp giới hạn (bounding boxes) và chuyển sang dạng mảng NumPy trên CPU cho dễ xử lý.
    boxes = results[0].boxes.cpu().numpy()

    # --- Bước 3.3: Chuẩn bị các biến để lưu kết quả ---

    dict_coords = {}  # Dictionary để lưu tọa độ các khuôn mặt.
    dict_cropped_faces = {}  # Dictionary để lưu các ảnh khuôn mặt đã được cắt.
    # Nếu flag yêu cầu vẽ ảnh (2 hoặc 4), tạo một bản sao của ảnh gốc để vẽ lên đó.
    image_with_boxes = img_original.copy() if flag in [2, 4] else None

    # --- Bước 3.4: Lặp qua từng khuôn mặt đã phát hiện để xử lý ---

    # `enumerate` để lấy cả chỉ số (i) và đối tượng (box) trong mỗi vòng lặp.
    for i, box in enumerate(boxes):
        # Lấy tọa độ (x1, y1) là góc trên bên trái và (x2, y2) là góc dưới bên phải.
        # `map(int, ...)` để chuyển các tọa độ từ số thực sang số nguyên.
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Tạo một key duy nhất cho mỗi người, ví dụ: "person_1", "person_2",...
        person_key = f"person_{i + 1}"

        # Tùy theo giá trị của 'flag', thực hiện các hành động tương ứng.
        if flag in [1, 2, 4]:  # Nếu cần lấy tọa độ.
            dict_coords[person_key] = np.array([x1, y1, x2, y2])
        if flag in [2, 4]:  # Nếu cần vẽ ô vuông.
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vẽ hình chữ nhật màu xanh lá.
        if flag in [3, 4]:  # Nếu cần cắt khuôn mặt.
            # Cắt một vùng chữ nhật từ ảnh gốc dựa trên tọa độ đã phát hiện.
            # `max(0, ...)` để đảm bảo tọa độ không bị âm, tránh lỗi.
            cropped_face = img_original[max(0, y1):y2, max(0, x1):x2]
            dict_cropped_faces[person_key] = cropped_face

    # --- Bước 3.5: Trả về kết quả cuối cùng theo 'flag' ---
    if flag == 1: return dict_coords
    if flag == 2: return dict_coords, image_with_boxes
    if flag == 3: return dict_cropped_faces
    if flag == 4: return dict_coords, image_with_boxes, dict_cropped_faces


def process_video(video_path: str):
    """Hàm ứng dụng: Dùng hàm `detect_and_process_faces` để xử lý một file video."""
    # Mở file video bằng OpenCV. `cap` là đối tượng video capture.
    cap = cv2.VideoCapture(video_path)
    # Kiểm tra xem video có được mở thành công không.
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở file video tại '{video_path}'")
        return

    print("Đang xử lý video... Nhấn 'q' trên cửa sổ video để thoát.")
    # Bắt đầu vòng lặp vô hạn để đọc từng khung hình của video.
    while cap.isOpened():
        # Ghi lại thời gian bắt đầu xử lý khung hình.
        start_time = time.time()
        # Đọc một khung hình. `ret` là True nếu đọc thành công, `frame` là chính khung hình đó.
        ret, frame = cap.read()
        # Nếu `ret` là False, có nghĩa là đã hết video.
        if not ret:
            print("Đã xử lý xong video.")
            break

        # Gọi hàm xử lý cốt lõi cho khung hình hiện tại. Dùng flag=2 để vẽ ô vuông.
        result = detect_and_process_faces(frame, flag=2)

        # Kiểm tra kết quả trả về. Nếu là tuple, nghĩa là xử lý thành công.
        if isinstance(result, tuple):
            _, drawn_frame = result  # Chỉ lấy ảnh đã vẽ, bỏ qua dict tọa độ.
        else:
            drawn_frame = frame  # Nếu có lỗi, hiển thị khung hình gốc.

        # Tính toán thời gian xử lý và FPS (số khung hình trên giây).
        processing_time = time.time() - start_time
        fps = 1 / processing_time if processing_time > 0 else 0

        # Viết chữ FPS lên góc trên bên trái của khung hình.
        cv2.putText(drawn_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Hiển thị khung hình đã xử lý trong một cửa sổ có tên 'Video Face Detection'.
        cv2.imshow('Video Face Detection', drawn_frame)

        # Chờ 1 mili giây. Nếu trong lúc đó người dùng nhấn phím 'q', vòng lặp sẽ dừng.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Sau khi vòng lặp kết thúc, giải phóng đối tượng video.
    cap.release()
    # Đóng tất cả các cửa sổ của OpenCV.
    cv2.destroyAllWindows()


def process_webcam():
    """Hàm ứng dụng: Dùng hàm `detect_and_process_faces` để xử lý hình ảnh từ webcam."""
    # Mở webcam mặc định của máy tính. `0` là chỉ số của webcam đầu tiên.
    cap = cv2.VideoCapture(0)
    # Kiểm tra xem webcam có kết nối được không.
    if not cap.isOpened():
        print("Lỗi: Không thể kết nối với webcam.")
        return

    print("Đã kết nối webcam. Nhấn 'q' trên cửa sổ video để thoát.")
    # Bắt đầu vòng lặp để lấy hình ảnh từ webcam.
    while True:
        start_time = time.time()
        # Đọc một khung hình từ webcam.
        ret, frame = cap.read()
        if not ret:
            print("Lỗi: Mất kết nối với webcam.")
            break

        # Lật ảnh webcam theo chiều ngang (trục y, số 1).
        # Điều này tạo hiệu ứng giống như nhìn vào gương, tự nhiên hơn cho người dùng.
        frame = cv2.flip(frame, 1)

        # Gọi hàm xử lý cốt lõi, logic y hệt như xử lý video.
        result = detect_and_process_faces(frame, flag=2)

        if isinstance(result, tuple):
            _, drawn_frame = result
        else:
            drawn_frame = frame

        # Tính toán và hiển thị FPS.
        processing_time = time.time() - start_time
        fps = 1 / processing_time if processing_time > 0 else 0
        cv2.putText(drawn_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Hiển thị kết quả lên màn hình.
        cv2.imshow('Webcam Face Detection', drawn_frame)

        # Chờ nhấn phím 'q' để thoát.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng webcam và đóng cửa sổ.
    cap.release()
    cv2.destroyAllWindows()