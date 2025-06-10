import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data")
PROCESSED_PATH = os.path.join(DATA_PATH, "processed") 
EMBEDDINGS_PATH = os.path.join( DATA_PATH, "embeddings", "all_embeddings.pkl")
VIDEO_PATH = os.path.join(DATA_PATH, "videos", "taylor.mp4") 
RESULTS_PATH = os.path.join(DATA_PATH, "results") 
TEST = os.path.join(DATA_PATH, "raw")
THRESHOLD = 0.6  # Ngưỡng so sánh embeddings
VIDEO_RESOLUTION = (640, 480)
FACE_DETECTION_CONFIDENCE = 0.54

