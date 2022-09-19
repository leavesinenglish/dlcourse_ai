def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(prediction)):
        if prediction[i] and ground_truth[i]:
            tp += 1
        if prediction[i] and not ground_truth[i]:
            fp += 1
        if not prediction[i] and not ground_truth[i]:
            tn += 1
        if not prediction[i] and ground_truth[i]:
            fn += 1
    tp = tp / len(prediction)
    tn = tn / len(prediction)
    fn = fn / len(prediction)
    fp = fp / len(prediction)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    t = 0
    for i in range(len(prediction)):
        if prediction[i] == ground_truth[i]:
            t += 1
    return t / len(prediction)
