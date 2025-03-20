from logistic_model import logistic_fun
from optimizer import fit_logistic_sgd
from utils import generate_data, report_metrics
import numpy as np

if __name__ == '__main__':
    M = 2
    D = 5
    n_train = 200
    n_test = 100
    X_train, t_train, y_train_true = generate_data(n_train, M, D)
    X_test, _, y_test_true = generate_data(n_test, M, D)

    Ms = [1, 2, 3]
    
    for M in Ms:
        print("*"*50)
        print(f'Training for M = {M} with cross entropy loss')
        W = fit_logistic_sgd(X_train, t_train, M, type='cross_entropy', lr=0.001, batch_size=32, epochs=100)
        
        y_train_pred = np.array([logistic_fun(W, M, X_train[i]) for i in range(n_train)])
        y_test_pred = np.array([logistic_fun(W, M, X_test[i]) for i in range(n_test)])
        
        report_metrics(y_train_pred, y_test_pred, t_train, y_test_true, y_train_true)

    for M in Ms:
        print("*"*50)
        print(f'Training for M = {M} with RMSE loss')
        W = fit_logistic_sgd(X_train, t_train, M, type='rmse', lr=0.001, batch_size=32, epochs=100)
        
        y_train_pred = np.array([logistic_fun(W, M, X_train[i]) for i in range(n_train)])
        y_test_pred = np.array([logistic_fun(W, M, X_test[i]) for i in range(n_test)])
        
        report_metrics(y_train_pred, y_test_pred, t_train, y_test_true, y_train_true)

    print("*"*50)
    print("Extra metric chosen: F1-score balances precision and recall well, making it more suitable when class imbalance exists.")
    print("F1-score is also a better indicator of overfitting than accuracy because it balances precision and recall.")
    print("If the training F1-score is significantly higher than the test F1-score, the model may have overfitted to the training data nosisy labels.")