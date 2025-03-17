import numpy as np
from math import comb
from itertools import combinations_with_replacement

def polynomial_features(x, M):
    D = len(x)
    features = [1]

    for m in range(1, M+1):
        for combination in combinations_with_replacement(range(D), m):
            features.append(np.prod([x[i] for i in combination]))
    return np.array(features)

def count_p(M, D):
    return sum(comb(D + m - 1, m) for m in range(M + 1))

def logistic_fun(w, M, x):
    poly = polynomial_features(x, M)
    z = np.clip(np.dot(w, poly), -700, 700)
    return 1 / (1 + np.exp(-z))