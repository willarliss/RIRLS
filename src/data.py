import numpy as np

def generate_sample(theta, n, intercept=True, seed=None):

    rng = np.random.default_rng(seed)

    m = theta.shape[1]

    X = rng.normal(size=(n, m))
    if intercept:
        X[:, 0] = 1

    P = np.hstack([X@theta.T, np.ones((n, 1))])
    P = np.exp(P) / np.exp(P).sum(1, keepdims=True)

    Y = np.vstack([rng.multinomial(1, p) for p in P])

    return Y, P, X
