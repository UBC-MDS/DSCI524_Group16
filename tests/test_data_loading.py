from pkg_pyknnclassifier.data_loading import data_loading
import pandas as pd

def test_data_loading_invalid_path():
    """Test data loading with invalid file path."""
    
    path = "../data/non_existent_file.csv"  # Use a clearly non-existent file for clarity
    target = "Target"
    
    try:
        data_loading(path, target)
    except ValueError as e:
        assert str(e) == f"File not found: {path}"
    else:
        assert False, "Expected ValueError not raised for invalid file path"
        
def test_data_loading_non_string_path():
    """Test data loading with a non-string file path."""
    
    path = 123  # Use a non-string input, like an integer
    target = "Target"
    
    try:
        data_loading(path, target)
    except ValueError as e:
        assert str(e) == "Input must be a string representing the file path."
    else:
        assert False, "Expected ValueError not raised for non-string file path"
        
def test_data_loading_valid_input():
    """Test data loading with valid file path and target column."""
    
    path = "./data/toy_dataset.csv" 
    target = "Target"
    
    try:
        features, target = data_loading(path, target)
        assert isinstance(features, pd.DataFrame), "Features should be a DataFrame"
        assert isinstance(target, pd.Series), "Target should be a Series"
    except Exception as e:
        assert False, f"Unexpected error occurred: {e}"


def test_data_loading_nonexistent_target_column():
    """Test data loading with non-existent target column."""
    
    
    path = "./data/toy_dataset.csv" 
    target = "invalidtarget"
    
    try:
        features, target = data_loading(path, target)
    except ValueError as e:
        assert str(e) == f"Target column '{target}' not found in the DataFrame."
    else:
        assert False, "Expected ValueError not raised"

