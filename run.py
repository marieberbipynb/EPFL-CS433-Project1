"""Train the final model and create submission.csv.

The full pipeline is deterministic and uses only NumPy plus the standard
library. It first estimates a decision threshold on a fixed validation split,
then retrains the selected weighted ridge model on the complete training set.
"""

from pathlib import Path

import numpy as np

from project_utils import (
    Preprocessor,
    choose_threshold,
    classification_metrics,
    fit_weighted_ridge,
    load_dataset,
    predict_scores,
    stratified_split,
    write_submission,
)


DATA_DIR = Path(__file__).resolve().parent / "dataset"
OUTPUT_PATH = Path(__file__).resolve().parent / "submission.csv"

CATEGORICAL_THRESHOLD = 20
LAMBDA = 0.01
POSITIVE_WEIGHT = 4.0
VALIDATION_FRACTION = 0.2
SEED = 123


def main():
    """Run validation, final training, and prediction export."""
    x_train, x_test, y_train, _, test_ids, feature_names = load_dataset(DATA_DIR)
    train_idx, validation_idx = stratified_split(
        y_train,
        validation_fraction=VALIDATION_FRACTION,
        seed=SEED,
    )

    validation_preprocessor = Preprocessor(
        categorical_threshold=CATEGORICAL_THRESHOLD
    ).fit(x_train[train_idx], feature_names)
    validation_weights = fit_weighted_ridge(
        validation_preprocessor,
        x_train[train_idx],
        y_train[train_idx],
        lambda_=LAMBDA,
        positive_weight=POSITIVE_WEIGHT,
    )
    validation_scores = predict_scores(
        validation_preprocessor,
        validation_weights,
        x_train[validation_idx],
    )
    threshold, threshold_metrics = choose_threshold(
        validation_scores,
        y_train[validation_idx],
    )
    validation_predictions = np.where(validation_scores >= threshold, 1, -1)
    validation_metrics = classification_metrics(
        y_train[validation_idx],
        validation_predictions,
    )

    print("Validation metrics")
    for key, value in validation_metrics.items():
        print(f"  {key}: {value:.5f}")
    print(f"  threshold: {threshold:.6f}")
    print(f"  selected_threshold_f1: {threshold_metrics['f1']:.5f}")

    final_preprocessor = Preprocessor(categorical_threshold=CATEGORICAL_THRESHOLD).fit(
        x_train,
        feature_names,
    )
    final_weights = fit_weighted_ridge(
        final_preprocessor,
        x_train,
        y_train,
        lambda_=LAMBDA,
        positive_weight=POSITIVE_WEIGHT,
    )
    test_scores = predict_scores(final_preprocessor, final_weights, x_test)
    test_predictions = np.where(test_scores >= threshold, 1, -1)
    write_submission(test_ids, test_predictions, OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
