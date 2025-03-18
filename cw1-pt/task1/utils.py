from logistic_model import logistic_fun, count_p
import numpy as np

def generate_data(n, M, D, seed=None):
    if seed is not None:
        np.random.seed(seed)  
    else:
        np.random.seed() 
    X = np.random.uniform(-5, 5, (n, D))
    p = count_p(M, D)
    w = np.array([(-1) ** (p - k) * np.sqrt(k) / p for k in range(p, 0, -1)])
    
    y_true = np.array([logistic_fun(w, M, X[i]) for i in range(n)])
    y_noise = y_true + np.random.normal(0, 1.0, n) 
    
    y_true = (y_true >= 0.5).astype(int)  
    t = (y_noise >= 0.5).astype(int)  
    
    return X, t, y_true

def compute_accuracy(y_pred, y_true):
    y_pred = np.round(y_pred)  
    return np.mean(y_pred == y_true)

def compute_f1_score(y_true, y_pred):
    y_pred = np.round(y_pred)  
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def report_metrics(y_train_pred, y_test_pred, t_train, y_test_true, y_train_true):
    train_acc = compute_accuracy(y_train_pred, t_train)
    test_acc = compute_accuracy(y_test_pred, y_test_true)
    true_train_acc = compute_accuracy(y_train_pred, y_train_true)

    f1_train = compute_f1_score(y_train_true, y_train_pred)
    f1_test = compute_f1_score(y_test_true, y_test_pred)

    print(f"Train Accuracy (on noisy labels): {train_acc:.3f}, Test Accuracy: {test_acc:.3f}")
    print(f"Train Accuracy (on true labels): {true_train_acc:.3f}")
    print(f"F1 Score (Train): {f1_train:.3f}, F1 Score (Test): {f1_test:.3f}")

    if f1_train > f1_test and train_acc - test_acc > 0.005:
        print("Train F1-score is higher than Test F1-score, indicating possible overfitting.")
    else:
        print("Train and Test F1-scores are similar, indicating good generalization.")
    
