"""Evaluate the NumPy-only histogram boosting model on a fixed split."""

import os
from pathlib import Path

import numpy as np

from numpy_boosting import HistogramGradientBoosting, QuantileBinner
from project_utils import (
    ID_COLUMNS,
    Preprocessor,
    choose_threshold,
    classification_metrics,
    fit_weighted_ridge,
    load_dataset,
    predict_scores,
    stratified_split,
)


DATA_DIR = Path(__file__).resolve().parent / "dataset"
SPLIT_SEED = int(os.environ.get("SPLIT_SEED", "123"))
MODEL_SEED = int(os.environ.get("MODEL_SEED", str(SPLIT_SEED)))
N_ESTIMATORS = int(os.environ.get("N_ESTIMATORS", "360"))
MAX_BINS = int(os.environ.get("MAX_BINS", "32"))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", "2"))
MAX_FEATURES = int(os.environ.get("MAX_FEATURES", "160"))
MIN_SAMPLES_LEAF = int(os.environ.get("MIN_SAMPLES_LEAF", "300"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.06"))
L2 = float(os.environ.get("L2", "20.0"))
POSITIVE_WEIGHT = float(os.environ.get("POSITIVE_WEIGHT", "1.5"))
CATEGORICAL_SPLITS = os.environ.get("CATEGORICAL_SPLITS", "0") == "1"
LEARN_MISSING_DIRECTION = os.environ.get("LEARN_MISSING_DIRECTION", "0") == "1"
FEATURE_SAMPLING = os.environ.get("FEATURE_SAMPLING", "tree")
RUN_RIDGE = os.environ.get("RUN_RIDGE", "1") == "1"
SAVE_SCORES = os.environ.get("SAVE_SCORES")


def main():
    x_train, _, y_train, _, _, feature_names = load_dataset(DATA_DIR)
    keep = np.array(
        [name not in ID_COLUMNS for name in feature_names],
        dtype=bool,
    )
    boosting_x = x_train[:, keep]
    train_idx, validation_idx = stratified_split(y_train, 0.2, SPLIT_SEED)
    binner = QuantileBinner(max_bins=MAX_BINS).fit(boosting_x[train_idx])
    train_bins = binner.transform(boosting_x[train_idx])
    validation_bins = binner.transform(boosting_x[validation_idx])
    print(
        f"split={SPLIT_SEED} model_seed={MODEL_SEED} trees={N_ESTIMATORS} "
        f"depth={MAX_DEPTH} "
        f"bins={MAX_BINS} features={MAX_FEATURES} leaf={MIN_SAMPLES_LEAF} "
        f"lr={LEARNING_RATE} l2={L2} weight={POSITIVE_WEIGHT} "
        f"categorical={CATEGORICAL_SPLITS} "
        f"missing_direction={LEARN_MISSING_DIRECTION} "
        f"feature_sampling={FEATURE_SAMPLING}",
        flush=True,
    )

    history = []

    def report(iteration, _, validation_scores):
        if iteration % 10:
            return
        threshold, metrics = choose_threshold(
            validation_scores,
            y_train[validation_idx],
        )
        history.append((metrics["f1"], iteration))
        print(
            f"trees={iteration:3d} f1={metrics['f1']:.5f} "
            f"accuracy={metrics['accuracy']:.5f} threshold={threshold:.5f}",
            flush=True,
        )

    model = HistogramGradientBoosting(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        l2=L2,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        positive_weight=POSITIVE_WEIGHT,
        categorical_splits=CATEGORICAL_SPLITS,
        learn_missing_direction=LEARN_MISSING_DIRECTION,
        feature_sampling=FEATURE_SAMPLING,
        seed=MODEL_SEED,
    ).fit(
        train_bins,
        y_train[train_idx],
        binner.n_bins,
        categorical_features=binner.categorical_features,
        validation_data=(validation_bins, y_train[validation_idx]),
        callback=report,
    )
    best_f1, best_iteration = max(history)
    print(f"best checkpoint: trees={best_iteration}, f1={best_f1:.5f}")
    boosting_scores = model.decision_function(
        validation_bins,
        n_estimators=best_iteration,
    )
    threshold, _ = choose_threshold(boosting_scores, y_train[validation_idx])
    predictions = np.where(boosting_scores >= threshold, 1, -1)
    print("boosting", classification_metrics(y_train[validation_idx], predictions))

    if SAVE_SCORES:
        output = Path(SAVE_SCORES)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            validation_indices=validation_idx,
            validation_scores=boosting_scores,
            validation_labels=y_train[validation_idx],
            best_iteration=best_iteration,
        )
        print(f"saved validation scores to {output}")

    if not RUN_RIDGE:
        return

    ridge_preprocessor = Preprocessor().fit(x_train[train_idx], feature_names)
    ridge_weights = fit_weighted_ridge(
        ridge_preprocessor,
        x_train[train_idx],
        y_train[train_idx],
        lambda_=0.005,
        positive_weight=4.0,
    )
    ridge_scores = predict_scores(
        ridge_preprocessor,
        ridge_weights,
        x_train[validation_idx],
    )
    normalized_ridge = (ridge_scores - ridge_scores.mean()) / ridge_scores.std()
    normalized_boosting = (
        boosting_scores - boosting_scores.mean()
    ) / boosting_scores.std()
    for ridge_fraction in np.linspace(0.0, 1.0, 11):
        blended = (
            ridge_fraction * normalized_ridge
            + (1.0 - ridge_fraction) * normalized_boosting
        )
        threshold, metrics = choose_threshold(blended, y_train[validation_idx])
        print(
            f"ridge_fraction={ridge_fraction:.1f} f1={metrics['f1']:.5f} "
            f"accuracy={metrics['accuracy']:.5f} threshold={threshold:.5f}"
        )


if __name__ == "__main__":
    main()
