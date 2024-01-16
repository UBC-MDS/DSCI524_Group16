from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd

def scaling(df, impute_strategy, scale_method):
    """ Apply imputation and scaling to the given data.

    Parameters
    ----------    
    df: pd.DataFrame 
        The DataFrame to be preprocessed.
        
    impute_strategy: str
        The strategy for imputation, 'mean', 'median', 'most_frequent', or 'constant'.
        
    scale_method: str
        The scaling method, either 'StandardScaler' or 'MinMaxScaler'.
        
    Returns
    ---------- 
    pd.DataFrame
        The scaled DataFrame.

    Examples
    --------
    >>> df_scaled = scaling(df, impute_strategy='mean', scale_method='MinMaxScaler')
    """
    
    # Impute missing values
    imputer = SimpleImputer(strategy=impute_strategy)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Scale data
    if scale_method == 'StandardScaler':
        scaler = StandardScaler()
    elif scale_method == 'MinMaxScaler':
        scaler = MinMaxScaler()
    else:
        raise ValueError("scale_method must be 'StandardScaler' or 'MinMaxScaler'.")

    # Apply scaling
    df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df.columns)

    return df_scaled
