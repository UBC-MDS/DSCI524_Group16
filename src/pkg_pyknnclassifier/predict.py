from collections import Counter
import numpy as np
import pandas as pd


def predict(train_X, train_y, unlabel_df, pred_method, k):
    """
    This function predicts the labels of the unlabled observations based on the similarity score calculated from Euclidean distance.
    
    Parameters
    ----------
    train_X : pd.DataFrame
        The data frame containing labeled observations, but without the label.

    train_y : numpy.array
        The array containing labels in the training dataset

    unlabel_df : pd.DataFrame
        The data frame containing unlabeld observations.

    pred_method : str
        'soft' or 'hard'.

    k : int
        The number of nearest neighbors to consider for making predictions.

    Returns
    -------
    array
        An array containing predicted labels for the observations.

    Examples
    --------
    df = pd.DataFrame({'A':[0.5, 0.2, 0.4],
                       'B':[0.3, 0.2, 0.5]})
    predict(df)                
    """
    predictions = []
    X_array = train_X.values
    unlabel_array = unlabel_df.values
    for data_point in unlabel_array:
        neighbors_idxs = find_neighbors(X_array, unlabel_array, k)
        neighbor_labels = train_y[neighbors_idxs]
        cnt = Counter(neighbor_labels)

        if pred_method == 'hard':
            label = cnt.most_common()[0][0]
            predictions.append(label)
        if pred_method == 'soft':
            prob = cnt.most_common()[0][1] / (cnt.most_common()[0][1] + cnt.most_common()[1][1])
            predictions.append(prob)
    return np.array(predictions)