import numpy as np


inv = np.linalg.pinv
# inv = np.linalg.inv
norm = np.linalg.norm


def irls(X, Y, theta0=None, max_iter=100, eps=1e-9, actual=None):

    n, m = X.shape
    _, k = Y.shape

    if theta0 is None:
        theta0 = np.zeros((k-1, m))
    else:
        assert theta0.shape == (k-1, m)
        theta0 = theta0.copy()

    errors, diffs = [], []

    theta = theta0.copy()
    for rep in range(max_iter):
        P = np.exp(X@theta.T)
        P = P / (P.sum(1, keepdims=True) + 1)
        for i in range(k-1):
            Yi, Pi = Y[:, i], P[:, i]
            Wi = np.diag(Pi * (1-Pi))
            theta[i, :] += inv(X.T @ Wi @ X) @ X.T @ (Yi-Pi)

        if actual is not None:
            errors.append(norm(theta-actual, 'fro'))
        diff = norm(theta-theta0, 'fro')
        diffs.append(diff)
        if diff < eps:
            break

        theta0 = theta.copy()

    return {'estim': theta, 'convergence': diffs, 'errors': errors, 'iters': rep+1}


def rec_irls(X, Y, theta0=None, icovs0=None, actual=None):

    n, m = X.shape
    _, k = Y.shape

    if theta0 is None:
        theta0 = np.zeros((k-1, m))
    else:
        assert theta0.shape == (k-1, m)
    if icovs0 is None:
        icovs = [np.eye(m) for _ in range(k-1)]
        # icovs = [np.eye(m)*1000 for _ in range(k-1)]
    else:
        assert len(icovs0) == k-1
        assert all(icov.shape == (m,m) for icov in icovs0)
        icovs = [icov.copy() for icov in icovs0]

    errors, diffs = [], []

    theta = theta0.copy()
    for t in range(n):
        Xt, Yt = X[[t], :].T, Y[[t], :].T
        Pt = np.exp(theta @ Xt)
        Pt = Pt / (Pt.sum(0, keepdims=True) + 1)
        for i in range(k-1):
            Yti, Pti = Yt[i, :], Pt[i, :]
            Wti = Pti * (1-Pti)
            icovs[i] -= (Wti * icovs[i] @ Xt @ Xt.T @ icovs[i]) / (1 + Wti * Xt.T @ icovs[i] @ Xt)
            theta[[i], :] += ( icovs[i] @ Xt * (Yti-Pti) ).T

        if actual is not None:
            errors.append(norm(theta-actual, 'fro'))
        diffs.append(norm(theta-theta0, 'fro'))

        theta0 = theta.copy()

    return {'estim': theta, 'convergence': diffs, 'errors': errors, 'iters': n}


def rec_irls_agg(X, Y, theta0=None, icov0=None, actual=None):

    n, m = X.shape
    _, k = Y.shape

    if theta0 is None:
        theta0 = np.zeros((k-1, m))
    else:
        assert theta0.shape == (k-1, m)
    if icov0 is None:
        icov = np.eye(m)
        # icov = np.eye(m)*1000
    else:
        assert icov0.shape == (m,m)
        icov = icov0.copy()

    errors, diffs = [], []

    theta = theta0.copy()
    for t in range(n):
        Xt, Yt = X[[t], :].T, Y[[t], :].T
        Pt = np.exp(theta @ Xt)
        Pt = Pt / (Pt.sum(0, keepdims=True) + 1)
        Wt = Pt.T @ (1-Pt) / (k-1)
        icov -= (Wt * icov @ Xt @ Xt.T @ icov) / (1 + Wt * Xt.T @ icov @ Xt)
        theta += ( icov @ Xt @ (Yt[:k-1]-Pt).T ).T
        # Xt, Yt = X[[t], :].T, Y[[t], :].T #
        # Pt = np.exp( np.vstack([theta @ Xt, 1]) ) #
        # Pt = Pt / Pt.sum(0, keepdims=True) #
        # Wt = Pt.T @ (1-Pt) / (k-1) #
        # icov -= (Wt * icov @ Xt @ Xt.T @ icov) / (1 + Wt * Xt.T @ icov @ Xt) #
        # theta += ( icov @ Xt @ (Yt-Pt)[:k-1].T ).T #

        if actual is not None:
            errors.append(norm(theta-actual, 'fro'))
        diffs.append(norm(theta-theta0, 'fro'))

        theta0 = theta.copy()

    return {'estim': theta, 'convergence': diffs, 'errors': errors, 'iters': n}


def rec_ls(X, Y, theta0=None, icov0=None, actual=None):

    n, m = X.shape
    _, k = Y.shape

    if theta0 is None:
        theta0 = np.zeros((k, m))
    else:
        assert theta0.shape == (k, m)
    if icov0 is None:
        icov = np.eye(m)
        # icov = np.eye(m)*1000
    else:
        assert icov0.shape == (m,m)
        icov = icov0.copy()

    errors, diffs = [], []

    theta = theta0.copy()
    for t in range(n):
        Xt, Yt = X[[t], :].T, Y[[t], :].T
        icov -= (icov @ Xt @ Xt.T @ icov) / (1 + Xt.T @ icov @ Xt)
        theta += ( icov @ Xt @ (Yt - theta@Xt).T ).T

        if actual is not None:
            errors.append(norm(theta-actual, 'fro'))
        diffs.append(norm(theta-theta0, 'fro'))

        theta0 = theta.copy()

    return {'estim': theta, 'convergence': diffs, 'errors': errors, 'iters': n}
