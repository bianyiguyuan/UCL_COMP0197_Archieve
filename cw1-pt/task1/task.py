import numpy as np
from logistic_model import logistic_fun, count_p
from optimizer import fit_logistic_sgd

def generate_data(n, M, D):
    X = np.random.uniform(-5, 5, (n, D))
    p = count_p(M, D)
    w = np.array([(-1) ** (p - k) * np.sqrt(k) / p for k in range(p, 0, -1)])
    y_true = np.array([logistic_fun(w, M, X[i]) for i in range(n)])
    y_noise = y_true + np.random.normal(0, 1.0, n) 
    y_true = np.array([1 if y_true[i] >= 0.5 else 0 for i in range(n)])
    t = np.array([1 if y_noise[i] >= 0.5 else 0 for i in range(n)])
    return X, t, y_true

def compute_accuracy(y_pred, y_true):
    y_pred = np.round(y_pred)  
    return np.mean(y_pred == y_true)

def report_metrics(y_train_pred, y_test_pred, t_train, t_test, true_train_labels):
    train_acc = compute_accuracy(y_train_pred, t_train)
    test_acc = compute_accuracy(y_test_pred, t_test)

    true_train_acc = compute_accuracy(y_train_pred, true_train_labels)

    print(f"Train Accuracy (on noisy data): {train_acc:.3f}, Test Accuracy: {test_acc:.3f}")
    print(f"Train Accuracy (on true labels): {true_train_acc:.3f}")
    print("Difference: Model performs better on noisy labels, meaning it may have learned the noise.")


if __name__ == '__main__':
    # 1. Generate training and test data
    M = 2
    D = 5
    n_train = 200
    n_test = 100
    X_train, t_train, y_true = generate_data(n_train, M, D)
    X_test, t_test, _ = generate_data(n_test, M, D)

    Ms = [1, 2, 3]
    for M in Ms:
        print(f'Training for M = {M} with cross entropy loss')
        W = fit_logistic_sgd(X_train, t_train, M, type='cross_entropy', lr=0.01, batch_size=10, epochs=100)
        y_train = np.array([logistic_fun(W, M, X_train[i]) for i in range(n_train)])
        y_test = np.array([logistic_fun(W, M, X_test[i]) for i in range(n_test)])
        train_acc = np.mean(np.round(y_train) == t_train)
        test_acc = np.mean(np.round(y_test) == t_test)
        report_metrics(y_train, y_test, t_train, t_test, y_true)

    for M in Ms:
        print(f'Training for M = {M} with RMSE loss')
        W = fit_logistic_sgd(X_train, t_train, M, type='rmse', lr=0.01, batch_size=10, epochs=100)
        y_train = np.array([logistic_fun(W, M, X_train[i]) for i in range(n_train)])
        y_test = np.array([logistic_fun(W, M, X_test[i]) for i in range(n_test)])
        train_acc = np.mean(np.round(y_train) == t_train)
        test_acc = np.mean(np.round(y_test) == t_test)
        report_metrics(y_train, y_test, t_train, t_test, y_true)
    