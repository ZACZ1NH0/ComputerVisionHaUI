import numpy as np
from scipy.spatial.distance import cosine
from config import THRESHOLD

def compare_embeddings(input_embedding, stored_embeddings, stored_labels, threshold=THRESHOLD):

    min_dist = float("inf")
    best_match = "Unknown"

    for emb, label in zip(stored_embeddings, stored_labels):
        dist = cosine(input_embedding, emb)
        if dist < min_dist:
            min_dist = dist
            best_match = label

    if min_dist > threshold:
        best_match = "Unknown"

    return best_match, min_dist
