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
    if metric == 'accuracy':
        correct_pred = sum((y_pred == y_true).astype(int))
        return correct_pred/ len(y_true)
    
    elif metric in ['precision', 'recall', 'f1']:
        true_pos, pred_pos, real_pos = 0, 0, 0
        precision, recall = 0, 0
        for pred, true in zip(y_pred, y_true):
            if pred == true == 1:
                true_pos += 1
            if pred == 1:
                pred_pos += 1
            if true == 1:
                real_pos += 1
        if pred_pos == 0:
            precision = 0
        else:
            precision = true_pos/pred_pos
        if real_pos == 0:
            recall = 0
        else:
            recall =  true_pos/real_pos
        if metric == 'precision':
            return precision
        elif metric == 'recall':
            return recall
        else:
            if precision == recall == 0:
                return 0
            else:
                return 2*precision*recall/(precision+recall)
