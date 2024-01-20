from pkg_pyknnclassifier.calculate_distance import calculate_distance
import numpy as np

obs_1 = [1, 2, 0.5, -2]
obs_2 = [0, 1/3, -3, 2.1]

def test_calculate_distance_chebyshev():
    """Test if the functions calculates the chebyshev distance correctly"""

    calculated_chebyshev = calculate_distance(obs_1, obs_2, method = "Chebyshev")
    expected_chebyshev = np.max(np.array([1, 5/3, 3.5, 4.1]))

    assert calculated_chebyshev == expected_chebyshev, "Chebyshev distance calculate incorrectly!"


def test_calculate_distance_euclidean():
    """Test if the functions calculates the euclidean distance correctly"""

    calculated_euclidean = calculate_distance(obs_1, obs_2)
    expected_euclidean = np.linalg.norm(np.array(obs_1) - np.array(obs_2))

    assert calculated_euclidean == expected_euclidean, "Euclidean distance calculate incorrectly!"


def test_calculate_distance_manhattan():
    """Test if the functions calculates the manhattan distance correctly"""

    calculated_manhattan = calculate_distance(obs_1, obs_2, method = "Manhattan")
    expected_manhattan = np.sum(np.array([1, 5/3, 3.5, 4.1]))

    assert calculated_manhattan == expected_manhattan, "Manhattan distance calculate incorrectly!"

def test_calculate_distance_right_length():
    try:
        calculate_distance([1, 2, 0.5, -2], [0, 1/3, -3, 2.1, 2])
    except ValueError:
        print("Two observations need to have the same number of features!")

def test_calculate_distance_right_metric():
    try:
        calculate_distance(obs_1,obs_2, method = "AU")
    except ValueError:
        print("The metrics have to come from Euclidean, Manhattan, or Chebyshev!")