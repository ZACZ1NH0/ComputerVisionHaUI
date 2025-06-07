import cv2
import numpy as np
from typing import Tuple, Dict, List
import time

def draw_face_box(frame: np.ndarray, 
                 box: Tuple[int, int, int, int], 
                 label: str, 
                 confidence: float,
                 color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Vẽ khung và nhãn cho khuôn mặt
    
    Args:
        frame: Frame cần vẽ
        box: Tọa độ (x1, y1, x2, y2) của khung
        label: Nhãn của khuôn mặt
        confidence: Độ tin cậy của nhận diện
        color: Màu của khung (BGR)
        
    Returns:
        Frame đã vẽ
    """
    x1, y1, x2, y2 = box
    
    # Vẽ khung
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Tạo text với nhãn và độ tin cậy
    text = f"{label} ({confidence:.2f})"
    
    # Tính toán vị trí text
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    text_x = x1
    text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
    
    # Vẽ background cho text
    cv2.rectangle(frame, 
                 (text_x, text_y - text_size[1] - 4),
                 (text_x + text_size[0], text_y + 4),
                 color, -1)
    
    # Vẽ text
    cv2.putText(frame, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame

def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """
    Vẽ FPS lên frame
    
    Args:
        frame: Frame cần vẽ
        fps: FPS hiện tại
        
    Returns:
        Frame đã vẽ
    """
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

def calculate_iou(box1: Tuple[int, int, int, int], 
                 box2: Tuple[int, int, int, int]) -> float:
    """
    Tính IoU (Intersection over Union) giữa 2 khung
    
    Args:
        box1: Khung thứ nhất (x1, y1, x2, y2)
        box2: Khung thứ hai (x1, y1, x2, y2)
        
    Returns:
        Giá trị IoU
    """
    # Tính tọa độ giao nhau
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Tính diện tích giao nhau
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Tính diện tích của mỗi khung
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Tính diện tích hợp
    union = box1_area + box2_area - intersection
    
    # Trả về IoU
    return intersection / union if union > 0 else 0

def update_tracking(tracking_results: Dict,
                   face_id: str,
                   box: Tuple[int, int, int, int],
                   label: str,
                   confidence: float,
                   timestamp: float) -> Dict:
    """
    Cập nhật kết quả tracking
    
    Args:
        tracking_results: Dict chứa kết quả tracking
        face_id: ID của khuôn mặt
        box: Tọa độ khung
        label: Nhãn của khuôn mặt
        confidence: Độ tin cậy
        timestamp: Thời điểm phát hiện
        
    Returns:
        Dict tracking_results đã cập nhật
    """
    if face_id not in tracking_results:
        tracking_results[face_id] = {
            'label': label,
            'first_seen': timestamp,
            'last_seen': timestamp,
            'appearances': 1,
            'boxes': [box],
            'confidences': [confidence]
        }
    else:
        tracking_results[face_id]['last_seen'] = timestamp
        tracking_results[face_id]['appearances'] += 1
        tracking_results[face_id]['boxes'].append(box)
        tracking_results[face_id]['confidences'].append(confidence)
        
    return tracking_results
