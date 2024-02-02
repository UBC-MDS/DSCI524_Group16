import numpy as np
from pkg_pyknnclassifier.calculate_distance import calculate_distance

def find_neighbors(labeled_arrays, unlabeled_array, k):
    """
    Finds the indices of the 'k' nearest neighbors in a collection of labeled arrays
    to a given unlabeled array.

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
    """

    # Check if 'k' is a positive integer and does not exceed the number of labeled arrays
    if not isinstance(k, int) or k <= 0 or k > len(labeled_arrays):
        raise ValueError("'k' must be a positive integer and should not exceed the number of labeled arrays.")


    distances = []

    # Calculate the distance between the unlabeled array and each labeled array
    for labeled_array in labeled_arrays:
        distance = calculate_distance(labeled_array, unlabeled_array)
        distances.append(distance)
    distances = np.array(distances)

    # Get the indices of the k nearest neighbors
    indices = np.argsort(distances)[:k]

    return indices
