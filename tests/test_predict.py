from pkg_pyknnclassifier.predict import predict
import pandas as pd
import numpy as np

train_X = pd.DataFrame({'A': [0.1, 0.2, 0.3], 'B': [0.2, 0.3, 0.4]})
train_y = np.array(['class1', 'class2', 'class1'])
unlabel_df = pd.DataFrame({'A': [0.15, 0.25], 'B': [0.25, 0.35]})
k = 1
expected = np.array(['class1', 'class2'])


def test_hard_prediction():
    """Test for making hard prediction"""
    pred_method = 'hard'
    predictions = predict(train_X, train_y, unlabel_df, pred_method, k)
    assert np.array_equal(predictions, expected), "Hard predictions are incorrect."
    
def test_soft_prediction():
    """Test for making soft prediction"""
    pred_method = 'soft'
    expected = np.array([1, 1])  
    predictions = predict(train_X, train_y, unlabel_df, pred_method, k)
    assert np.allclose(predictions, expected), "Soft predictions are incorrect."

def test_soft_prediction_k_larger_than_one():
    """Test to varify the function works for different k value"""
    k = 2
    pred_method = 'soft'
    expected = np.array([0.5, 0.5])  # Placeholder values, adjust as necessary
    
    # Perform predictions
    predictions = predict(train_X, train_y, unlabel_df, pred_method, k)
    
    # Check if the predictions match expected
    assert np.allclose(predictions, expected), f"Expected {expected} but got {predictions}"


def test_predict_invalid_train_dimensions():
    """Test predict with train_X and train_y of different lengths."""
    train_X = pd.DataFrame({'Feature1': [1, 2, 3], 'Feature2': [4, 5, 6]})
    train_y = np.array([7, 8])  # One less element than train_X
    unlabel_df = pd.DataFrame({'Feature1': [1, 2], 'Feature2': [3, 4]})
    
    try:
        predict(train_X, train_y, unlabel_df)
    except ValueError as e:
        assert str(e) == "train_X and train_y must have the same number of rows."
    else:
        assert False, "Expected ValueError not raised"

def test_predict_invalid_method():
    """Test predict with invalid pred_method."""
    train_X = pd.DataFrame({'Feature1': [1, 2, 3], 'Feature2': [4, 5, 6]})
    train_y = np.array([7, 8, 9])
    unlabel_df = pd.DataFrame({'Feature1': [1, 2], 'Feature2': [3, 4]})
    pred_method = "invalid"  # Invalid method
    
    try:
        predict(train_X, train_y, unlabel_df, pred_method)
    except ValueError as e:
        assert str(e) == "pred_method must be either 'hard' or 'soft'."
    else:
        assert False, "Expected ValueError not raised"

def test_predict_invalid_k():
    """Test predict with invalid k."""
    train_X = pd.DataFrame({'Feature1': [1, 2, 3], 'Feature2': [4, 5, 6]})
    train_y = np.array([7, 8, 9])
    unlabel_df = pd.DataFrame({'Feature1': [1, 2], 'Feature2': [3, 4]})
    k = 4  # More than the number of labeled examples
    
    try:
        predict(train_X, train_y, unlabel_df, k=k)
    except ValueError as e:
        assert str(e) == "k must be positive and less than or equal to the number of labeled examples."
    else:
        assert False, "Expected ValueError not raised"

