import numpy as np
from scipy.spatial.distance import cosine

def cosine_sim(a, b):
    return 1 - cosine(a, b)

def euclidean_l2(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.linalg.norm(a - b)
