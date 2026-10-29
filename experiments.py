"""Reproduce the weighted-ridge hyperparameter comparison.

This script uses the same deterministic validation split and preprocessing as
``run.py``. It is intentionally separate from the final submission pipeline so
that ``python run.py`` remains a single, predictable command.
"""

from pathlib import Path

import numpy as np

from project_utils import (
    Preprocessor,
    choose_threshold,
    fit_weighted_ridge,
    load_dataset,
    predict_scores,
    stratified_split,
)

DATA_DIR = Path(__file__).resolve().parent / "dataset"
LAMBDAS = (0.001, 0.003, 0.005, 0.01)
POSITIVE_WEIGHTS = (1.0, 4.0, 8.0)
VALIDATION_FRACTION = 0.2
SEED = 123


def main():
    """Print validation F1 for the candidate weighted-ridge models."""
    x_train, _, y_train, _, _, feature_names = load_dataset(DATA_DIR)
    train_idx, validation_idx = stratified_split(
        y_train,
        validation_fraction=VALIDATION_FRACTION,
        seed=SEED,
    )
    preprocessor = Preprocessor(categorical_threshold=20).fit(
        x_train[train_idx],
        feature_names,
    )

    results = []
    for positive_weight in POSITIVE_WEIGHTS:
        for lambda_ in LAMBDAS:
            weights = fit_weighted_ridge(
                preprocessor,
                x_train[train_idx],
                y_train[train_idx],
                lambda_=lambda_,
                positive_weight=positive_weight,
            )
            scores = predict_scores(
                preprocessor,
                weights,
                x_train[validation_idx],
            )
            threshold, metrics = choose_threshold(scores, y_train[validation_idx])
            results.append((metrics["f1"], lambda_, positive_weight, threshold))
            print(
                f"lambda={lambda_:.3g} positive_weight={positive_weight:.1f} "
                f"threshold={threshold:.6f} f1={metrics['f1']:.5f}"
            )

    best_f1, best_lambda, best_weight, best_threshold = max(results)
    print(
        "Best configuration: "
        f"lambda={best_lambda:.3g}, positive_weight={best_weight:.1f}, "
        f"threshold={best_threshold:.6f}, f1={best_f1:.5f}"
    )


if __name__ == "__main__":
    main()
