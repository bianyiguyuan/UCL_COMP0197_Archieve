from logistic_model import logistic_fun, count_p
import numpy as np

def generate_data(n, M, D, seed=None):
    """
    Generate synthetic data for logistic regression with polynomial features.

    Parameters:
    n : int
        Number of samples to generate.
    M : int
        Polynomial degree.
    D : int
        Number of input features.
    seed : int, optional
        Random seed for reproducibility.

    Returns:
    X : np.ndarray
        Generated input features of shape (n, D).
    t : np.ndarray
        Noisy target labels (binary) of shape (n,).
    y_true : np.ndarray
        True target labels (binary) before noise addition of shape (n,).
    """
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
    """
    Compute classification accuracy.

    Parameters:
    y_pred : np.ndarray
        Predicted labels of shape (n,).
    y_true : np.ndarray
        Ground-truth labels of shape (n,).

    Returns:
    float
        Accuracy score between 0 and 1.
    """
    y_pred = np.round(y_pred)  
    return np.mean(y_pred == y_true)

def compute_f1_score(y_true, y_pred):
    """
    Compute F1-score for binary classification.

    Parameters:
    y_true : np.ndarray
        Ground-truth labels of shape (n,).
    y_pred : np.ndarray
        Predicted labels of shape (n,).

    Returns:
    float
        F1-score between 0 and 1.
    """
    y_pred = np.round(y_pred)  
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def report_metrics(y_train_pred, y_test_pred, t_train, y_test_true, y_train_true):
    """
    Print accuracy and F1-score for training and testing datasets.

    Parameters:
    y_train_pred : np.ndarray
        Predicted labels for training data of shape (n_train,).
    y_test_pred : np.ndarray
        Predicted labels for test data of shape (n_test,).
    t_train : np.ndarray
        Noisy training labels of shape (n_train,).
    y_test_true : np.ndarray
        True labels for test data of shape (n_test,).
    y_train_true : np.ndarray
        True labels for training data of shape (n_train,).

    Returns:
    None
    """
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
    
