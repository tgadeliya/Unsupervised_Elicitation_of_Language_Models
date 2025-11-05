

def accuracy(preds, true_labels):
    """Calculate accuracy and map unparsable outputs to 0"""
    correct = 0
    for pred, label in zip(preds, true_labels):
        correct += 1 if pred == label else 0
    return (correct / (len(preds))) * 100
