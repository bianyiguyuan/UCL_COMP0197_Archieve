import numpy as np
from math import comb
from itertools import combinations_with_replacement

def polynomial_features(x, M):
    """
    Generate polynomial features up to degree M for input vector x.

    Parameters:
    -----------
    x : np.ndarray
        Input feature vector of shape (D,), where D is the number of features.
    M : int
        Maximum polynomial degree.

    Returns:
    --------
    np.ndarray
        A 1D numpy array containing polynomial features of shape (P,), where P is determined by count_p(M, D).
    """
    D = len(x)
    features = [1] 
    for m in range(1, M+1):
        for combination in combinations_with_replacement(range(D), m):
            features.append(np.prod([x[i] for i in combination]))
    return np.array(features)

def count_p(M, D):
    """
    Compute the number of polynomial features for given degree M and dimension D.

    Parameters:
    -----------
    M : int
        Maximum polynomial degree.
    D : int
        Number of input features.

    Returns:
    --------
    int
        The number of polynomial features including the bias term.
    """
    return sum(comb(D + m - 1, m) for m in range(M + 1))

def logistic_fun(w, M, x):
    """
    Compute the logistic (sigmoid) function for polynomial features.

    Parameters:
    -----------
    w : np.ndarray
        Weight vector of shape (P,), where P is the number of polynomial features.
    M : int
        Maximum polynomial degree.
    x : np.ndarray
        Input feature vector of shape (D,), where D is the number of original features.

    Returns:
    --------
    float
        The logistic function value in range (0, 1).
    """
    poly = polynomial_features(x, M)
    z = np.clip(np.dot(w, poly), -700, 700)  # Prevent overflow
    return 1 / (1 + np.exp(-z))