"""Compare the six methods of implementations.py on the validation split.

This script produces the numbers of the baseline table of the report. It uses
the same stratified split and preprocessing as ``run.py`` and ``experiments.py``
and reports, for each method, the validation F1 score with the default
threshold (score >= 0) and with a threshold tuned on the validation set.
"""

from pathlib import Path
import time

import numpy as np

from implementations import (
    least_squares,
    logistic_regression,
    mean_squared_error_gd,
    mean_squared_error_sgd,
    reg_logistic_regression,
    ridge_regression,
)
from project_utils import (
    Preprocessor,
    choose_threshold,
    classification_metrics,
    load_dataset,
    stratified_split,
)

DATA_DIR = Path(__file__).resolve().parent / "dataset"
VALIDATION_FRACTION = 0.2
SEED = 123

# Hyperparameters of the baselines (report them in the paper).
# Step sizes are not fixed by hand: they are derived from the smoothness of the
# loss (see safe_step_sizes) so that gradient descent cannot diverge.
GD_ITERS = 500
SGD_ITERS = 50_000
RIDGE_LAMBDA = 0.01
LOGISTIC_ITERS = 500
LOGISTIC_LAMBDA = 1e-4


def largest_eigenvalue(tx, n_iters=50, seed=0):
    """Estimate the largest eigenvalue of tx.T @ tx / n by power iteration."""
    v = np.random.default_rng(seed).normal(size=tx.shape[1])
    for _ in range(n_iters):
        v = tx.T @ (tx @ v) / tx.shape[0]
        v /= np.linalg.norm(v)
    return float(v @ (tx.T @ (tx @ v)) / tx.shape[0])


def safe_step_sizes(tx):
    """Return step sizes that guarantee convergence of GD, SGD and logistic GD.

    The MSE loss is L-smooth with L the largest eigenvalue of tx.T @ tx / n, so
    GD converges for gamma < 2 / L. The logistic loss is (L / 4)-smooth. For SGD
    the relevant constant is the largest squared norm of a single sample.
    """
    smoothness = largest_eigenvalue(tx)
    max_row_norm = float(np.max(np.sum(tx**2, axis=1)))
    return {
        "gd": 1.0 / smoothness,
        "sgd": 0.5 / max_row_norm,
        "logistic": 4.0 / smoothness,
    }


def f1_scores(scores, y_true):
    """Return the validation F1 at threshold 0 and at the tuned threshold."""
    default_predictions = np.where(scores >= 0.0, 1, -1)
    f1_default = classification_metrics(y_true, default_predictions)["f1"]
    _, tuned_metrics = choose_threshold(scores, y_true)
    return f1_default, tuned_metrics["f1"]


def main():
    """Train the six baselines and print the table rows."""
    x_train, _, y_train, _, _, feature_names = load_dataset(DATA_DIR)
    print("Data loaded", flush=True)
    train_idx, validation_idx = stratified_split(
        y_train,
        validation_fraction=VALIDATION_FRACTION,
        seed=SEED,
    )

    preprocessor = Preprocessor(categorical_threshold=20).fit(
        x_train[train_idx],
        feature_names,
    )
    # Adapt this line if your Preprocessor method has another name, and add a
    # column of ones here if transform() does not already include the bias.
    tx_train = preprocessor.transform(x_train[train_idx])
    tx_val = preprocessor.transform(x_train[validation_idx])
    print(f"Preprocessing done: {tx_train.shape[1]} features", flush=True)

    y_tr = y_train[train_idx]  # labels in {-1, 1} for regression models
    y_tr01 = (y_tr == 1).astype(float)  # labels in {0, 1} for logistic models
    y_val = y_train[validation_idx]
    w0 = np.zeros(tx_train.shape[1])
    gammas = safe_step_sizes(tx_train)
    print("Step sizes:", {k: f"{v:.3g}" for k, v in gammas.items()}, flush=True)

    # SGD visits samples in order, so shuffle them once with a fixed seed.
    order = np.random.default_rng(SEED).permutation(tx_train.shape[0])

    models = {
        "Least squares": lambda: least_squares(y_tr, tx_train),
        "MSE GD": lambda: mean_squared_error_gd(
            y_tr, tx_train, w0, GD_ITERS, gammas["gd"]
        ),
        "MSE SGD": lambda: mean_squared_error_sgd(
            y_tr[order], tx_train[order], w0, SGD_ITERS, gammas["sgd"]
        ),
        "Ridge regression": lambda: ridge_regression(y_tr, tx_train, RIDGE_LAMBDA),
        "Logistic regression": lambda: logistic_regression(
            y_tr01, tx_train, w0, LOGISTIC_ITERS, gammas["logistic"]
        ),
        "Reg. logistic regression": lambda: reg_logistic_regression(
            y_tr01, tx_train, LOGISTIC_LAMBDA, w0, LOGISTIC_ITERS, gammas["logistic"]
        ),
    }

    print(f"{'Method':<26} {'F1 (t=0)':>9} {'F1 (tuned)':>11}", flush=True)
    for name, train in models.items():
        start = time.time()
        print(f"Training {name}...", flush=True)
        w, _ = train()
        f1_default, f1_tuned = f1_scores(tx_val @ w, y_val)
        print(
            f"{name:<26} {f1_default:>9.3f} {f1_tuned:>11.3f}"
            f"   ({time.time() - start:.0f} s)",
            flush=True,
        )
        # Row ready to paste into the LaTeX table.
        print(f"    LaTeX: {name} & {f1_default:.3f} & {f1_tuned:.3f} \\\\")


if __name__ == "__main__":
    main()