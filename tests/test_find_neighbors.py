import numpy as np
from pkg_pyknnclassifier.find_neighbors import find_neighbors
from pkg_pyknnclassifier.calculate_distance import calculate_distance

def test_find_neighbors_valid_input():
    """Test find_neighbors with valid input."""
    labeled_arrays = [
        np.array([1, 2]),
        np.array([2, 3]), 
        np.array([3, 4])
    ]
    unlabeled_array = np.array([2, 2])
    k = 2

    distances = [calculate_distance(labeled_array, unlabeled_array) for labeled_array in labeled_arrays]
    expected_indices = np.argsort(distances)[:k]
    indices = find_neighbors(labeled_arrays, unlabeled_array, k)
    assert np.array_equal(indices, expected_indices), f"Expected indices {expected_indices} but got {indices}"

def test_find_neighbors_invalid_k():
    """Test find_neighbors with invalid 'k'."""
    labeled_arrays = [
        np.array([1, 2]),
        np.array([2, 3]), 
        np.array([3, 4])
    ]
    unlabeled_array = np.array([2, 2])
    k = -10  # Use a negative integer for 'k'
    try:
        find_neighbors(labeled_arrays, unlabeled_array, k)
    except ValueError as e:
        assert str(e) == "k should be positive integer"
        assert False, "Expected ValueError not raised"

"""
def test_find_neighbors_inconsistent_dimensions():
    #Test find_neighbors with inconsistent dimensions in labeled arrays.
    labeled_arrays = [
        np.array([1, 2]),
        np.array([2, 3, 4])  # Use an array of different length
    ]
    unlabeled_array = np.array([2, 2])
    k = 1
    try:
        find_neighbors(labeled_arrays, unlabeled_array, k)
    except ValueError as e:
        assert str(e) == "All labeled arrays must have the same dimensionality."
    else:
        assert False, "Expected ValueError not raised"
"""
