def compute_accuracy(preds, labels):
    return (preds == labels).sum() / float(len(labels))

def compute_scores(preds, labels):

    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1_score = 2 * precision * recall / (precision + recall)

    scores = {
        "True Positive": tp,
        "False Positive": fp,
        "True Negative": tn,
        "False Negative": fn,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score
    }

    return scores

def eval_model(model, train_data, train_labels, val_data, val_labels):
    model.fit(train_data, train_labels)
    val_preds = model.predict(val_data)
    val_scores = compute_scores(val_preds, val_labels)
    return val_scores

def save_submission(preds, filename):
    import pandas as pd
    df = pd.DataFrame(preds, columns=['label'])
    df.to_csv(filename, index=False)