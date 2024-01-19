from collections import Counter
from pkg_pyknnclassifier.calculate_distance import calculate_distance
import numpy as np
import pandas as pd


def find_neighbors(labeled_arraies, unlabeled_array, k):
    """
    Finds the indices of the 'k' nearest neighbors in a collection of labeled arrays 
    to a given unlabeled array. 

    This function computes the distance between each labeled array and the 
    unlabeled array using the 'calculate_distance' function. It then selects the 'k' 
    labeled arrays that are closest to the unlabeled array.

    Parameters
    ----------
    labeled_arrays : array-like
        An iterable (list, NumPy array, etc.) of labeled arrays. Each array represents 
        an individual data point in the feature space.
    
    unlabeled_array : array-like
        The array representing the unlabeled data point for which the neighbors 
        are to be found. It should have the same length as each array in labeled_arrays.
    
    k : int
        The number of nearest neighbors to find. 'k' must be a positive integer, and 
        it should not exceed the number of arrays in labeled_arrays.

    Returns
    -------
    indices : numpy.ndarray
        An array of indices of the 'k' nearest neighbors from the labeled_arrays.

    Notes
    -----
    - The distance measurement used depends on the 'calculate_distance' function's definition.
    - This function assumes that all arrays (both labeled and unlabeled) are of 
      the same dimensionality and are compatible for distance calculations.
    """
    distances = []
    for labeled_array in labeled_arraies:
        distance = calculate_distance(labeled_array, unlabeled_array)
        distances.append(distance)
    distances = np.array(distances)
    indices = np.argsort(distances)[:k]

    return indices