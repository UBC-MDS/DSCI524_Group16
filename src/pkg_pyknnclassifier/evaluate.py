def evaluate(y_true, y_pred, metric='accuracy'):
    """
    This function calculates evaluation metrics such as accuracy, precision, recall, and F1 score
    for a k-NN model based on true labels and predicted labels. The default metrics to return is 
    accuracy.

    Parameters:
    - y_true (list or array): True labels.
    - y_pred (list or array): Predicted labels.
    - metric (str, optional): Metric to compute. Default is 'accuracy'.
        Possible values: 'accuracy', 'precision', 'recall', 'f1'.

    Returns:
    - float: Value of the specified metric.

    Examples:
    true_labels = [0, 1, 1, 0, 1, 0, 1, 0]
    predicted_labels = [0, 1, 1, 0, 1, 1, 0, 1]
    accuracy_result = evaluate_knn_manual(true_labels, predicted_labels, metric='accuracy')
    print("Accuracy:", accuracy_result)
    precision_result = evaluate_knn_manual(true_labels, predicted_labels, metric='precision')
    print("Precision:", precision_result)
    """