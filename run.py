"""Train the final NumPy-only ensemble and create submission.csv."""

from pathlib import Path

import numpy as np

from numpy_boosting import HistogramGradientBoosting, QuantileBinner
from project_utils import ID_COLUMNS, LEAKAGE_COLUMNS, load_dataset, write_submission

DATA_DIR = Path(__file__).resolve().parent / "dataset"
OUTPUT_PATH = Path(__file__).resolve().parent / "submission.csv"
MODEL_SEEDS = (123, 321)
N_ESTIMATORS = 250
POSITIVE_RATE = 0.127


def standardize(scores):
    """Put independently trained model scores on a common scale."""
    return (scores - scores.mean()) / scores.std()


def predict_top_fraction(scores, positive_rate):
    """Return exactly the requested fraction of highest scores as positive."""
    if not 0.0 < positive_rate < 1.0:
        raise ValueError("positive_rate must be strictly between 0 and 1.")
    positive_count = int(round(positive_rate * scores.size))
    positive_indices = np.argpartition(scores, -positive_count)[-positive_count:]
    predictions = np.full(scores.size, -1, dtype=np.int8)
    predictions[positive_indices] = 1
    return predictions


def main():
    """Fit two categorical boosting models and write the final submission."""
    x_train, x_test, y_train, _, test_ids, feature_names = load_dataset(DATA_DIR)
    excluded = ID_COLUMNS | LEAKAGE_COLUMNS
    keep = np.array([name not in excluded for name in feature_names], dtype=bool)
    boosting_train = x_train[:, keep]
    boosting_test = x_test[:, keep]

    print("Fitting histogram bins", flush=True)
    binner = QuantileBinner(max_bins=32, categorical_threshold=20).fit(boosting_train)
    train_bins = binner.transform(boosting_train)
    test_bins = binner.transform(boosting_test)

    model_scores = []
    for model_number, seed in enumerate(MODEL_SEEDS, start=1):
        print(
            f"Training model {model_number}/{len(MODEL_SEEDS)} (seed={seed})",
            flush=True,
        )

        def progress(iteration, _, __):
            if iteration % 10 == 0:
                print(f"  trees={iteration}/{N_ESTIMATORS}", flush=True)

        model = HistogramGradientBoosting(
            n_estimators=N_ESTIMATORS,
            learning_rate=0.06,
            max_depth=3,
            l2=20.0,
            min_samples_leaf=300,
            max_features=160,
            positive_weight=1.5,
            categorical_splits=True,
            learn_missing_direction=True,
            feature_sampling="tree",
            seed=seed,
        ).fit(
            train_bins,
            y_train,
            binner.n_bins,
            categorical_features=binner.categorical_features,
            callback=progress,
        )
        model_scores.append(standardize(model.decision_function(test_bins)))

    ensemble_scores = np.mean(model_scores, axis=0)
    predictions = predict_top_fraction(ensemble_scores, POSITIVE_RATE)
    write_submission(test_ids, predictions, OUTPUT_PATH)
    print(
        f"Wrote {OUTPUT_PATH} ({np.sum(predictions == 1)} positive predictions)",
        flush=True,
    )


if __name__ == "__main__":
    main()
