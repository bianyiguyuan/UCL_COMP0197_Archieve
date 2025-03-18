import torch
import torch.optim as optim
import numpy as np
from logistic_model import polynomial_features, count_p, logistic_fun
from optimizer import fit_logistic_sgd
from utils import report_metrics
from loss_function import myCrossEntropy
from utils import generate_data, compute_f1_score, compute_accuracy
from collections import Counter

def fit_learnable_M(X, t, M_max=5, lr=0.001, batch_size= 10, epochs=100):
    n = X.shape[0]
    D = X.shape[1]
    poly_features = {M: np.array([polynomial_features(X[i], M) for i in range(n)]) for M in range(1, M_max+1)}
    alpha = torch.nn.Parameter(torch.ones(M_max, dtype=torch.float32,requires_grad=True))
    W = {M: torch.nn.Parameter(torch.randn(count_p(M, D), dtype=torch.float32) * 0.05) for M in range(1, M_max+1)}
    loss_func = myCrossEntropy()
    optimizer = optim.SGD([alpha] + list(W.values()), lr=lr)
    for epoch in range(epochs):
        for i in range(0, n, batch_size):
            X_batch = {M: torch.tensor(poly_features[M][i:i+batch_size], dtype=torch.float32) for M in range(1, M_max+1)}
            t_batch = torch.tensor(t[i:i+batch_size], dtype=torch.float32)
            alpah_softmax = torch.nn.functional.softmax(alpha, dim=0)
            y_pred = sum([alpah_softmax[M-1] * torch.sigmoid(X_batch[M] @ W[M]) for M in range(1, M_max+1)])
            loss = loss_func(y_pred, t_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, M weights: {alpah_softmax.detach().numpy()}')
    
    return alpha.detach().numpy(), {M: W[M].detach().numpy() for M in range(1, M_max+1)}

if __name__ == '__main__':
    n_train = 200
    n_test = 100
    M = 2
    D = 5
    X_train, t_train, y_train_true = generate_data(n_train, M, D)
    X_test, _, y_test_true = generate_data(n_test, M, D)
    M_results = []
    num_trials = 5

    for i in range(num_trials):
        print(f'Trial {i+1}/{num_trials}:')
        alpha_softmax, W_dict = fit_learnable_M(X_train, t_train, M_max=5, lr=0.05, batch_size=64, epochs=100)
        best_M = np.argmax(alpha_softmax) + 1  
        M_results.append(best_M)

    M_counter = Counter(M_results)
    best_M= min(M_counter, key=lambda x: (-M_counter[x], x))  

    W = fit_logistic_sgd(X_train, t_train, best_M, type='cross_entropy', lr=0.001, batch_size=32, epochs=100)
        
    y_train_pred = np.array([logistic_fun(W, best_M, X_train[i]) for i in range(n_train)])
    y_test_pred = np.array([logistic_fun(W, best_M, X_test[i]) for i in range(n_test)])

    train_acc = compute_accuracy(y_train_pred, t_train)
    test_acc = compute_accuracy(y_test_pred, y_test_true)

    f1_train = compute_f1_score(y_train_true, y_train_pred)
    f1_test = compute_f1_score(y_test_true, y_test_pred)

    if f1_test < f1_train and best_M > 1 and train_acc-test_acc > 0.01:
        print('f1_test < f1_train')
        print(f'Overfitting on M: {best_M}, evaluating on M: {best_M-1}')
        best_M -= 1
        W = fit_logistic_sgd(X_train, t_train, best_M, type='cross_entropy', lr=0.001, batch_size=32, epochs=100)

        y_train_pred = np.array([logistic_fun(W, best_M, X_train[i]) for i in range(n_train)])
        y_test_pred = np.array([logistic_fun(W, best_M, X_test[i]) for i in range(n_test)])

    print(f'Optimal M selected: {best_M}')
    report_metrics(y_train_pred, y_test_pred, t_train, y_test_true, y_train_true)

    

    


