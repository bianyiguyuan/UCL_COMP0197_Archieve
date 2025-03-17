import numpy as np
from logistic_model import logistic_fun, count_p
from optimizer import fit_logistic_sgd

def generate_data(n, M, D):
    X = np.random.uniform(-5, 5, (n, D))
    p = count_p(M, D)
    w = np.array([(-1) ** (p - k) * np.sqrt(k) / p for k in range(p, 0, -1)])
    y = np.array([logistic_fun(w, M, X[i]) for i in range(n)]) + np.random.normal(0, 1.0, n)
    t = np.array([1 if y[i] >= 0.5 else 0 for i in range(n)])
    return X, t

if __name__ == '__main__':
    # 1. Generate training and test data
    M = 2
    D = 5
    n_train = 200
    n_test = 100
    X_train, t_train = generate_data(n_train, M, D)
    X_test, t_test = generate_data(n_test, M, D)

    Ms = [1, 2, 3]
    for M in Ms:
        print(f'Training for M = {M} with cross entropy loss')
        W = fit_logistic_sgd(X_train, t_train, M, type='cross_entropy', lr=0.01, batch_size=10, epochs=100)
        y_train = np.array([logistic_fun(W, M, X_train[i]) for i in range(n_train)])
        y_test = np.array([logistic_fun(W, M, X_test[i]) for i in range(n_test)])
        train_acc = np.mean(np.round(y_train) == t_train)
        test_acc = np.mean(np.round(y_test) == t_test)
        print(f'Train Accuracy: {train_acc}, Test Accuracy: {test_acc}')

    for M in Ms:
        print(f'Training for M = {M} with RMSE loss')
        W = fit_logistic_sgd(X_train, t_train, M, type='rmse', lr=0.01, batch_size=10, epochs=100)
        y_train = np.array([logistic_fun(W, M, X_train[i]) for i in range(n_train)])
        y_test = np.array([logistic_fun(W, M, X_test[i]) for i in range(n_test)])
        train_acc = np.mean(np.round(y_train) == t_train)
        test_acc = np.mean(np.round(y_test) == t_test)
        print(f'Train Accuracy: {train_acc}, Test Accuracy: {test_acc}')