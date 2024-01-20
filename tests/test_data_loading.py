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

