import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_metrics(y_pred, y_true, sessions=None, tolerance=0):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    if sessions is not None:
        sessions = np.array(sessions)

    classes = np.unique(np.concatenate([y_true, y_pred]))

    y_pred_tolerant = y_pred.copy()

    if tolerance > 0:
        for i in range(len(y_pred)):
            if sessions is not None:
                same_session_indices = np.where(sessions == sessions[i])[0]
                
                if i == same_session_indices[same_session_indices <= i][0]:
                    start = same_session_indices[same_session_indices <= i][0]
                else:
                    start = max(same_session_indices[same_session_indices <= i][-1] - tolerance, same_session_indices[same_session_indices <= i][0])
                
                if i == same_session_indices[same_session_indices >= i][-1]:
                    end = same_session_indices[same_session_indices >= i][-1]
                else:
                    end = min(same_session_indices[same_session_indices >= i][0] + tolerance + 1, same_session_indices[same_session_indices >= i][-1] + 1)
            else:
                start = max(0, i - tolerance)
                end = min(len(y_true), i + tolerance + 1)
            
            if y_pred[i] in y_true[start:end]:
                y_pred_tolerant[i] = y_true[i]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)

    accuracy_tolerant = accuracy_score(y_true, y_pred_tolerant)
    precision_tolerant = precision_score(y_true, y_pred_tolerant, labels=classes, average="macro", zero_division=0)
    recall_tolerant = recall_score(y_true, y_pred_tolerant, labels=classes, average="macro", zero_division=0)
    f1_tolerant = f1_score(y_true, y_pred_tolerant, labels=classes, average="macro", zero_division=0)

    return {
        "test_accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy_tolerant": accuracy_tolerant,
        "precision_tolerant": precision_tolerant,
        "recall_tolerant": recall_tolerant,
        "f1_tolerant": f1_tolerant
    }

