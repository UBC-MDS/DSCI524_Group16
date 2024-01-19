from pkg_pyknnclassifier.predict import predict
import pandas as pd
import numpy as np

train_X = pd.DataFrame({'A': [0.1, 0.2, 0.3], 'B': [0.2, 0.3, 0.4]})
train_y = np.array(['class1', 'class2', 'class1'])
unlabel_df = pd.DataFrame({'A': [0.15, 0.25], 'B': [0.25, 0.35]})
k = 1
expected = np.array(['class1', 'class2'])


def test_hard_prediction():
    pred_method = 'hard'
    predictions = predict(train_X, train_y, unlabel_df, pred_method, k)
    assert np.array_equal(predictions, expected), "Hard predictions are incorrect."
    
def test_soft_prediction():
    pred_method = 'soft'
    expected = np.array([1, 1])  
    predictions = predict(train_X, train_y, unlabel_df, pred_method, k)
    assert np.allclose(predictions, expected), "Soft predictions are incorrect."

def test_soft_prediction_k_larger_than_one():
    k = 2
    pred_method = 'soft'
    expected = np.array([0.5, 0.5])  # Placeholder values, adjust as necessary
    
    # Perform predictions
    predictions = predict(train_X, train_y, unlabel_df, pred_method, k)
    
    # Check if the predictions match expected
    assert np.allclose(predictions, expected), f"Expected {expected} but got {predictions}"
