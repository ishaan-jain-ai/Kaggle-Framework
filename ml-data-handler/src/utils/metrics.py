def accuracy(y_true, y_pred):
    return sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred)) / len(y_true)

def precision(y_true, y_pred):
    true_positive = sum(1 for y_t, y_p in zip(y_true, y_pred) if y_t == 1 and y_p == 1)
    predicted_positive = sum(1 for y_p in y_pred if y_p == 1)
    return true_positive / predicted_positive if predicted_positive > 0 else 0.0

def recall(y_true, y_pred):
    true_positive = sum(1 for y_t, y_p in zip(y_true, y_pred) if y_t == 1 and y_p == 1)
    actual_positive = sum(1 for y_t in y_true if y_t == 1)
    return true_positive / actual_positive if actual_positive > 0 else 0.0

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0