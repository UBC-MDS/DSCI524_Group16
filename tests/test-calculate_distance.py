from pkg_pyknnclassifier.calculate_distance import calculate_distance
import numpy as np


def test_calculate_distance_check_dimensions():
    """Test if the two data points are of the same dimensions"""
    list1 = [1, 2, 3]
    list2 = [2, 3]

    try:
        calculate_distance(list1, list2)
    except ValueError as e:
        assert str(e) == "Failed length test"
    else:
        assert False, "Expected ValueError not raised"

def test_calculate_distance_check_method():
    """Test if the method is correct"""
    obs_1 = [1, 2, 0.5, -2]
    obs_2 = [0, 1 / 3, -3, 2.1]

    try:
        calculate_distance(obs_1, obs_2, method="invalid")
    except ValueError as e:
        assert str(e) == "invalid method"
    else:
        assert False, "Expected ValueError not raised"

def test_calculate_distance_chebyshev():
    """Test if the functions calculates the chebyshev distance correctly"""

    obs_1 = [1, 2, 0.5, -2]
    obs_2 = [0, 1 / 3, -3, 2.1]

    calculated_chebyshev = calculate_distance(obs_1, obs_2, method="Chebyshev")
    expected_chebyshev = np.max(np.array([1, 5 / 3, 3.5, 4.1]))

    assert (
        calculated_chebyshev == expected_chebyshev
    ), "Chebyshev distance calculate incorrectly!"


def test_calculate_distance_euclidean():
    """Test if the functions calculates the euclidean distance correctly"""

    obs_1 = [1, 2, 0.5, -2]
    obs_2 = [0, 1 / 3, -3, 2.1]

    calculated_euclidean = calculate_distance(obs_1, obs_2, method="Euclidean")
    expected_euclidean = np.linalg.norm(np.array(obs_1) - np.array(obs_2))

    assert (
        calculated_euclidean == expected_euclidean
    ), "Euclidean distance calculate incorrectly!"


def test_calculate_distance_manhattan():
    """Test if the functions calculates the manhattan distance correctly"""

    obs_1 = [0, 1, 0, 1]
    obs_2 = [0, 1, 0, 1]

    calculated_manhattan = calculate_distance(obs_1, obs_2, method="Manhattan")
    expected_manhattan = 0

    assert (
        calculated_manhattan == expected_manhattan
    ), "Manhattan distance calculate incorrectly!"


def test_calculate_distance_default():
    """Test if the function calculates the default (Euclidean) distance correctly"""

    obs_1 = [1, 2, 0.5, -2]
    obs_2 = [0, 1 / 3, -3, 2.1]

    calculated_default = calculate_distance(obs_1, obs_2)
    expected_default = np.linalg.norm(np.array(obs_1) - np.array(obs_2))

    assert (
        calculated_default == expected_default
    ), "Default (Euclidean) distance calculation incorrect!"

def test_calculate_distance_non_list_input():
    """Test if the function handles non-list input correctly"""
    obs_1 = "not a list"
    obs_2 = [0, 1 / 3, -3, 2.1]

    try:
        calculate_distance(obs_1, obs_2)
    except TypeError as e:
        assert str(e) == "Input must be a list or numpy array"
    else:
        assert False, "Expected TypeError not raised"

def test_calculate_distance_string_input():
    """Test if the function handles non-list input correctly"""
    obs_1 = [0, 1 / 3, -3, 2.1]
    obs_2 = "not a list"

    try:
        calculate_distance(obs_1, obs_2)
    except TypeError as e:
        assert str(e) == "Input must be a list or numpy array"
    else:
        assert False, "Expected TypeError not raised"
