# analyze.py
import csv
from datetime import datetime
from collections import defaultdict, Counter

LOG_FILE = "face_log.csv"

def log_face(name, score):
    """Ghi nhận diện khuôn mặt vào file CSV."""
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, f"{score:.2f}"])

def read_log():
    """Đọc log nhận diện từ file CSV."""
    logs = []
    try:
        with open(LOG_FILE, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) == 3:
                    logs.append({'time': row[0], 'name': row[1], 'score': float(row[2])})
    except FileNotFoundError:
        pass
    return logs

def get_statistics():
    """Trả về thống kê số lần nhận diện theo tên."""
    logs = read_log()
    counter = Counter()
    for entry in logs:
        counter[entry['name']] += 1
    return dict(counter)

def clear_log():
    """Xoá file log nếu cần reset."""
    open(LOG_FILE, 'w').close()
