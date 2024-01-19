from pkg_pyknnclassifier.data_loading import data_loading

def test_data_loading_invalid_path():
    """Test data loading with invalid file path."""
    path = "non_existent_file.csv"
    target = "target_column"
    
    try:
        data_loading(path, target)
    except ValueError as e:
        assert str(e) == "Input must be a string representing the file path."
    else:
        assert False, "Expected ValueError not raised for invalid file path"
        

