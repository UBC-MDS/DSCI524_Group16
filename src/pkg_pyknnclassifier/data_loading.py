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
    
    # Check if the input file path is a string
    if not isinstance(str_of_path, str):
        raise ValueError("Input must be a string representing the file path.")
    
    try:
    # Attempt to read the CSV file into a DataFrame
        data = pd.read_csv(str_of_path)
        train_y = data[target_column]
        train_X = data.drop(columns=[target_column])
    except FileNotFoundError:
        # Raise an error if the file is not found
        raise ValueError(f"File not found: {str_of_path}")
    except KeyError: 
        # Raise an error if the target_column is not found in the DataFrame
        raise ValueError(f"Target column '{target_column}' not found in the DataFrame.")
        
    # Return the features and target  
    return train_X, train_y


