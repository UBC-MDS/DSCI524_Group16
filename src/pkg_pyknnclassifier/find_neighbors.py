from collections import Counter
import numpy as np
import pandas as pd


def find_neighbors(labeled_arraies, unlabeled_array, k):
    distances = []
    for labeled_array in labeled_arraies:
        distance = calculate_distance(labeled_array, unlabeled_array)
        distances.append(distance)
    distances = np.array(distances)
    indices = np.argsort(distances)[:k]

    return indices