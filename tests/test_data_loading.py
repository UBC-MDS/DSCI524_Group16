from pkg_pyknnclassifier.data_loading import data_loading
from sklearn.datasets import make_classification
import pandas as pd

def generate_toy_dataset(n_samples=100, save_path=None):
    """Generate a simple toy dataset and save it to a file."""
    
    X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2, n_redundant=0, n_classes=2, random_state=42)

    X_df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
    y_df = pd.Series(y, name='Target')

    if save_path:
        X_df.assign(Target=y_df).to_csv(save_path, index=False)

    return X_df, y_df

X, y = generate_toy_dataset(n_samples=30, save_path='/toy_dataset.csv')


def test_data_loading_invalid_path():
    """Test data loading with invalid file path."""
    
    path = "/toy_dataset.csv"
    target = "target_column"
    
    try:
        data_loading(path, target)
    except ValueError as e:
        assert str(e) == "Input must be a string representing the file path."
    else:
        assert False, "Expected ValueError not raised for invalid file path"
        

