import cv2
import os
import time

person_name = input("Nhập tên thư mục người dùng (ví dụ: person_thanh): ").strip()
if not person_name:
    print("[X] Tên không được để trống.")
    exit()

try:
    num_images = int(input("Nhập số lượng ảnh cần chụp: "))
    if num_images <= 0:
        raise ValueError
except ValueError:
    print("[X] Số lượng ảnh không hợp lệ.")
    exit()

try:
    delay = float(input("Nhập độ trễ giữa các ảnh (giây): "))
    if delay <= 0:
        raise ValueError
except ValueError:
    print("[X] Độ trễ không hợp lệ.")
    exit()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  
raw_folder = os.path.join(project_root, 'data', 'raw')
save_folder = os.path.join(raw_folder, person_name)

os.makedirs(save_folder, exist_ok=True)
print(f"[✓] Ảnh sẽ lưu tại: {save_folder}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[X] Không thể mở webcam.")
    exit()

existing_images = [f for f in os.listdir(save_folder) if f.endswith(('.jpg', '.png'))]
count = len(existing_images)

print("\n[INFO] Đang chụp... Nhấn 'q' để thoát sớm.\n")

while count < num_images:
    ret, frame = cap.read()
    if not ret:
        print("[X] Không thể đọc frame từ webcam.")
        break

    save_path = os.path.join(save_folder, f"{count}.jpg")
    cv2.imwrite(save_path, frame)
    print(f"[{count+1}/{num_images}] Đã lưu: {save_path}")
    count += 1

    cv2.imshow("Capture - Nhấn 'q' để thoát", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[!] Đã thoát sớm.")
        break

    time.sleep(delay)

cap.release()
cv2.destroyAllWindows()
print(f"[✓] Hoàn thành. Tổng ảnh đã lưu: {count}")