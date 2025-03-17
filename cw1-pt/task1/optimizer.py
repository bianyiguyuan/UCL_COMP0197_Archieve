import torch
import torch.optim as optim
import numpy as np
from logistic_model import polynomial_features
from loss_function import myCrossEntropy, myRootMeanSquare

def fit_logistic_sgd(X, t, M, type='cross_entropy', lr=0.01, batch_size=10, epochs=100):
    n = X.shape[0]  
    poly_features = np.array([polynomial_features(X[i], M) for i in range(n)])
    W = torch.randn(poly_features.shape[1], requires_grad=True)
    loss_func = myCrossEntropy() if type == 'cross_entropy' else myRootMeanSquare()
    optimizer = optim.SGD([W], lr=lr)
    for epoch in range(epochs):
        for i in range(0, n, batch_size):
            X_batch = torch.tensor(poly_features[i:i+batch_size])
            t_batch = torch.tensor(t[i:i+batch_size])
            y_pred = torch.sigmoid(X_batch @ W)
            loss = loss_func(y_pred, t_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch%10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

    return W