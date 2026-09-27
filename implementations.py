"""Core machine-learning methods required for EPFL CS-433 project 1."""

import numpy as np


def _mse_loss(y, tx, w):
    """Return 1/(2n) times the squared error."""
    error = y - tx @ w
    return np.array((error @ error) / (2.0 * y.shape[0]))


def _mse_gradient(y, tx, w):
    """Return the gradient of the mean squared error."""
    error = y - tx @ w
    return -(tx.T @ error) / y.shape[0]


def _sigmoid(t):
    """Numerically stable sigmoid."""
    t = np.clip(t, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-t))


def _logistic_loss(y, tx, w):
    """Return the average negative log-likelihood for labels in {0, 1}."""
    scores = tx @ w
    return np.array(np.mean(np.logaddexp(0.0, scores) - y * scores))


def _logistic_gradient(y, tx, w):
    """Return the gradient of the average negative log-likelihood."""
    pred = _sigmoid(tx @ w)
    return tx.T @ (pred - y) / y.shape[0]


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using full-batch gradient descent."""
    y = np.asarray(y, dtype=float)
    tx = np.asarray(tx, dtype=float)
    w = np.asarray(initial_w, dtype=float).copy()

    for _ in range(max_iters):
        w = w - gamma * _mse_gradient(y, tx, w)

    return w, _mse_loss(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent with batch size 1."""
    y = np.asarray(y, dtype=float)
    tx = np.asarray(tx, dtype=float)
    w = np.asarray(initial_w, dtype=float).copy()
    n_samples = y.shape[0]

    for iteration in range(max_iters):
        index = iteration % n_samples
        row = tx[index : index + 1]
        target = y[index : index + 1]
        w = w - gamma * _mse_gradient(target, row, w)

    return w, _mse_loss(y, tx, w)


def least_squares(y, tx):
    """Least squares regression solved from the normal equations."""
    y = np.asarray(y, dtype=float)
    tx = np.asarray(tx, dtype=float)
    gram = tx.T @ tx
    rhs = tx.T @ y

    try:
        w = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(gram) @ rhs

    return w, _mse_loss(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression solved from the regularized normal equations."""
    y = np.asarray(y, dtype=float)
    tx = np.asarray(tx, dtype=float)
    n_samples, n_features = tx.shape
    gram = tx.T @ tx
    penalty = 2.0 * n_samples * lambda_ * np.eye(n_features)
    rhs = tx.T @ y

    try:
        w = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(gram + penalty) @ rhs

    return w, _mse_loss(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using full-batch gradient descent."""
    y = np.asarray(y, dtype=float)
    tx = np.asarray(tx, dtype=float)
    w = np.asarray(initial_w, dtype=float).copy()

    for _ in range(max_iters):
        w = w - gamma * _logistic_gradient(y, tx, w)

    return w, _logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using full-batch gradient descent."""
    y = np.asarray(y, dtype=float)
    tx = np.asarray(tx, dtype=float)
    w = np.asarray(initial_w, dtype=float).copy()

    for _ in range(max_iters):
        gradient = _logistic_gradient(y, tx, w) + 2.0 * lambda_ * w
        w = w - gamma * gradient

    return w, _logistic_loss(y, tx, w)
