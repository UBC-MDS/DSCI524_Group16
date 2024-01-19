import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from pkg_pyknnclassifier.scaling import scaling

def test_valid_imputation_and_scaling():
    """Test scaling with valid imputation and scale method."""
    df = pd.DataFrame({
        'A': [1, np.nan, 3],
        'B': [4, 5, np.nan]
    })

    # Test mean imputation with StandardScaler
    scaled_df = scaling(df, 'mean', 'StandardScaler')
    assert not scaled_df.isnull().any().any(), "Imputation failed, NaN values found"
    assert scaled_df.mean().sum() < 1e-6, "StandardScaler scaling failed to center data"

    # Test median imputation with MinMaxScaler
    scaled_df = scaling(df, 'median', 'MinMaxScaler')
    assert not scaled_df.isnull().any().any(), "Imputation failed, NaN values found"
    assert scaled_df.min().sum() == 0, "MinMaxScaler scaling failed to set min to 0"

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
