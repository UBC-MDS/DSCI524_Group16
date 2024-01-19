import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from pkg_pyknnclassifier.scaling import scaling

def test_invalid_impute_strategy():
    """Test scaling with invalid impute strategy."""
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    invalid_strategy = "invalid_strategy"
    try:
        scaling(df, invalid_strategy, 'StandardScaler')
    except ValueError as e:
        assert str(e) == "Can only use these strategies: ['mean', 'median', 'most_frequent', 'constant']"

def test_invalid_scale_method():
    """Test scaling with invalid scale method."""
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    invalid_scale_method = "invalid_scale"
    try:
        scaling(df, 'mean', invalid_scale_method)
    except ValueError as e:
        assert str(e) == "scale_method must be 'StandardScaler' or 'MinMaxScaler'."
