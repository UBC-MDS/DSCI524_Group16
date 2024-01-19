import pandas as pd

def data_loading(str_of_path, target_column):
    """ Load data from a file path and split into features and target.

    Parameters
    ----------    
    str_of_path : str 
        The file path to the dataset.
    target_column : str
        The name of the column to be used as the target variable.
    
    Returns
    ---------- 
    pd.DataFrame, pd.Series
        The features as a DataFrame and the target as a Series.

    Examples
    --------
    >>> # Loading from a CSV file and selecting target column
    >>> features, target = data_loading('path/to/your/data.csv', 'target_column_name')
    """
    if isinstance(str_of_path, str):
        data = pd.read_csv(str_of_path)
        train_y = data[target_column]
        train_X = data.drop(columns=[target_column])
        return train_X, train_y
    else:
        raise ValueError("Input must be a string representing the file path.")

