import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import cv2
import time
import os, sys
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(f'{base_dir}/src')
# Giả sử bạn đã có các hàm này
from recognition.recognize_faces import load_reference_embeddings, extract_embedding, recognize_faces
from detection.detect_faces import detect_and_process_faces

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition GUI")

        self.video_source = None
        self.cap = None
        self.running = False

        # Load embeddings
        self.reference_embeddings, self.reference_labels = load_reference_embeddings()

        # Tạo canvas hiển thị video
        self.canvas = tk.Label(self.root)
        self.canvas.pack()

        # Nút điều khiển
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Start Webcam", command=self.start_webcam).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Browse Video", command=self.browse_video).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Stop", command=self.stop).pack(side=tk.LEFT, padx=5)

    def start_webcam(self):
        self.stop()
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.process_frame()

    def browse_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
        if file_path:
            self.stop()
            self.cap = cv2.VideoCapture(file_path)
            self.running = True
            self.process_frame()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def process_frame(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            return

        frame = cv2.flip(frame, 1)

        result = detect_and_process_faces(frame, flag=4)
        if result:
            coords_dict, frame_with_boxes, faces_dict = result

            recog_results = {}
            for face_id, face_img in faces_dict.items():
                embedding = extract_embedding(face_img)
                if embedding is not None:
                    name, score = recognize_faces(embedding, self.reference_embeddings, self.reference_labels)
                else:
                    name, score = "Unknown", 0.0
                recog_results[face_id] = (name, score)

            for face_id, (name, score) in recog_results.items():
                if face_id in coords_dict:
                    x1, y1, x2, y2 = coords_dict[face_id]
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    label = f"{name} ({score:.1f}%)" if name != "Unknown" else "Unknown"
                    cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_with_boxes, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            frame_with_boxes = frame

        # Hiển thị ảnh
        img = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (800, 600))
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        self.canvas.imgtk = imgtk
        self.canvas.config(image=imgtk)

        # Tiếp tục đọc frame
        self.root.after(10, self.process_frame)


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop)
    root.mainloop()
