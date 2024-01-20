import numpy as np

def calculate_distance(obs_1, obs_2, method = "Euclidean"):
    """
    This function calculates the Euclidean distance between two observations for the KNN model to find the similarity score.
    
    Parameters
    ----------
    obs_1 : array
        An array containing the features of the first observation.
    obs_2 : array
        An array containing the features of the second observation.

    Returns
    -------
    float
        Float representing a euclidean distance between two observations.

    Examples
    --------
    obs_1 = [0.81, 0.2, -0.86, 0.08]
    obs_2 = [-0.39, 0.24, -0.77, 0.17]
    dist = calculate_distance(obs_1, obs_2)
    print(f"Euclidean Distance between two observations is {dist}")
    """

    if method == "Euclidean":
        distance = (np.sum((np.array(obs_1) - np.array(obs_2))**2))**0.5
        return distance
    
    elif method == "Manhattan":
        distance = np.sum(np.abs(np.array(obs_1) - np.array(obs_2)))
        return distance
    
    elif method == "Chebyshev":
        distance = np.max(np.abs(np.array(obs_1) - np.array(obs_2)))
        return distance
    
    else:
        raise ValueError(f"Invalid method: {method}. Possible method: 'Euclidean', 'Manhattan', 'Chebyshev'")