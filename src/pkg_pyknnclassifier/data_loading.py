def data_loading(str_of_path):
    """ Load data from a file path.

    Parameters
    ----------    
    str_of_path: str 
        The file path to the dataset.
    
    Returns
    ---------- 
    pd.DataFrame
        The loaded data as a DataFrame.

    Examples
    --------
    >>> # Loading from a CSV file
    >>> df = data_loading('path/to/your/data.csv')
    """
    if isinstance(str_of_path, str):
        return pd.read_csv(str_of_path)
    else:
        raise ValueError("Input must be a file path or a DataFrame.")

