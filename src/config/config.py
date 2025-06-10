# DATA_PATH = "data/"
# PROCESSED_PATH = DATA_PATH + "processed/"
# EMBEDDINGS_PATH = DATA_PATH + "embeddings/embeddings.pkl"
# TEST = DATA_PATH +  "raw/"
# VIDEO_PATH = DATA_PATH + "videos/input_video.mp4"
# RESULTS_PATH = DATA_PATH + "results/"
# THRESHOLD = 0.6  # Ngưỡng so sánh embeddings
# VIDEO_RESOLUTION = (640, 480)
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_PATH = os.path.join(BASE_DIR, "data")
PROCESSED_PATH = os.path.join(DATA_PATH, "processed")
EMBEDDINGS_PATH = os.path.join(DATA_PATH, "embeddings", "all_embeddings.pkl")
VIDEO_PATH = os.path.join(DATA_PATH, "videos", "input_video.mp4")
RESULTS_PATH = os.path.join(DATA_PATH, "results")
TEST = os.path.join(DATA_PATH, "raw")
THRESHOLD = 0.6
VIDEO_RESOLUTION = (640, 480)
