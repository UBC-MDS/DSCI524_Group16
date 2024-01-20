import numpy as np
from pkg_pyknnclassifier.evaluate import evaluate
true_labels = [0, 1, 1, 0, 1, 0, 1, 0]
predicted_labels = [0, 1, 1, 0, 1, 1, 0, 1]

def test_accuracy():
    """Test of the accuracy computation"""
    acc_real = 5/8
    acc_func = evaluate(true_labels, predicted_labels, 'accuracy') 
    tol = 1e-8
    assert np.abs(acc_real - acc_func) < tol , "wrong calculation of accuracy"

def test_default_metric():
    """Test computation when the metric parameter is not provided."""
    true_labels = [0, 1, 1, 0]
    predicted_labels = [0, 1, 1, 0]
    acc_real = 1
    acc_func = evaluate(true_labels, predicted_labels) 
    tol = 1e-8
    assert np.abs(acc_real - acc_func) < tol , "wrong calculation of accuracy"

def test_precision():
    """Test precision computation."""
    precision_real = 3/5
    precision_func = evaluate(true_labels, predicted_labels, 'precision')
    tol = 1e-8
    assert np.abs(precision_real - precision_func) < tol, "Wrong calculation of precision"

def test_recall():
    """Test recall computation."""
    recall_real = 3/4
    recall_func = evaluate(true_labels, predicted_labels, 'recall')
    tol = 1e-8
    assert np.abs(recall_real - recall_func) < tol, "Wrong calculation of recall"

def test_f1_score():
    """Test F1 score computation."""
    f1_real = (2 * 3/4 * 3/5) / (3/4 + 3/5)
    f1_func = evaluate(true_labels, predicted_labels, 'f1')
    tol = 1e-8
    assert np.abs(f1_real - f1_func) < tol, "Wrong calculation of F1 score"

def test_invalid_metric():
    """Test that the function raises a ValueError for an invalid metric."""
    invalid_metric = 'invalid_metric'
    try:
        evaluate(true_labels, predicted_labels, metric=invalid_metric)
    except ValueError as e:
        assert str(e) == f"Invalid metric: {invalid_metric}. Possible values: 'accuracy', 'precision', 'recall', 'f1'"
    else:
        assert False, "Expected ValueError not raised"

def test_precision_zero_denominator():
    """Test precision computation when the denominator is zero."""
    true_labels = [0, 0]
    predicted_labels = [1, 1]
    precision_result = evaluate(true_labels, predicted_labels, 'precision')
    assert precision_result == 0, "Precision should be 0 when the denominator is zero"

def test_recall_zero_denominator():
    """Test recall computation when the denominator is zero."""
    true_labels = [0, 0]
    predicted_labels = [1, 1]
    recall_result = evaluate(true_labels, predicted_labels, 'recall')
    assert recall_result == 0, "Recall should be 0 when the denominator is zero"

def test_f1_zero_denominator():
    """Test recall computation when the denominator is zero."""
    true_labels = [0, 0]
    predicted_labels = [1, 1]
    f1_result = evaluate(true_labels, predicted_labels, 'f1')
    assert f1_result == 0, "F1 should be 0 when the denominator is zero"

def test_empty_labels():
    """Test computation when both predicted and true labels are empty."""
    true_labels = []
    predicted_labels = []
    try:
        evaluate(true_labels, predicted_labels, 'accuracy')
    except ValueError as e:
        assert str(e) == "Both predicted and true labels are empty."
    else:
        assert False, "Expected ValueError not raised"

def test_different_lengths():
    """Test computation when predicted and true labels have different lengths."""
    true_labels = [0, 1, 1, 0]
    predicted_labels = [0, 1, 1, 0, 1]
    try:
        evaluate(true_labels, predicted_labels, 'accuracy')
    except ValueError as e:
        assert str(e) == "Predicted and true labels must have the same length."
    else:
        assert False, "Expected ValueError not raised"