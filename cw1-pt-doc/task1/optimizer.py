import torch
import torch.optim as optim
import numpy as np
from logistic_model import polynomial_features
from loss_function import myCrossEntropy, myRootMeanSquare

def fit_logistic_sgd(X, t, M, type='cross_entropy', lr= 0.001, batch_size=10, epochs=100):
    """
    Trains a logistic regression model using stochastic gradient descent (SGD).

    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (N, D), where N is the number of samples and D is the feature dimension.
    t : np.ndarray
        Target labels of shape (N,), where N is the number of samples.
    M : int
        Degree of polynomial features.
    type : str, optional
        Type of loss function to use, either 'cross_entropy' or 'root_mean_square' (default is 'cross_entropy').
    lr : float, optional
        Learning rate for SGD (default is 0.001).
    batch_size : int, optional
        Batch size for mini-batch training (default is 10).
    epochs : int, optional
        Number of training epochs (default is 100).

    Returns:
    --------
    np.ndarray
        Trained model weights of shape (P,), where P is the number of polynomial features.
    """

    n = X.shape[0]  
    poly_features = np.array([polynomial_features(X[i], M) for i in range(n)])
    
    W = torch.nn.Parameter(torch.randn(poly_features.shape[1], dtype=torch.float32) * 0.05)
    loss_func = myCrossEntropy() if type == 'cross_entropy' else myRootMeanSquare()
    optimizer = optim.SGD([W], lr=lr)

    for epoch in range(epochs):
        for i in range(0, n, batch_size):
            X_batch = torch.tensor(poly_features[i:i+batch_size], dtype=torch.float32)
            t_batch = torch.tensor(t[i:i+batch_size], dtype=torch.float32)

            y_pred = torch.sigmoid(X_batch @ W)
            loss = loss_func(y_pred, t_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

    return W.detach().numpy()
