import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import time
import os, sys
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(f'{base_dir}/src')
from recognition.recognize_faces import load_reference_embeddings, extract_embedding, recognize_faces
from detection.detect_faces import detect_and_process_faces
from config.config import THRESHOLD

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Webcam Face Recognition")
        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Không thể mở webcam")
            return

        self.reference_embeddings, self.reference_labels = load_reference_embeddings()
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Không lấy được frame từ webcam")
            self.root.after(10, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        result = detect_and_process_faces(frame, flag=4)

        if result is not None:
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
                if face_id not in coords_dict:
                    continue
                x1, y1, x2, y2 = coords_dict[face_id]
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                label = f"{name} ({score:.1f}%)" if name != "Unknown" else "Unknown"

                cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, 2)
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(frame_with_boxes, (x1, y1 - text_size[1] - 10),
                             (x1 + text_size[0], y1), color, -1)
                cv2.putText(frame_with_boxes, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        else:
            frame_with_boxes = frame

        # Convert frame to ImageTk format
        rgb_frame = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.root.after(10, self.update_frame)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
