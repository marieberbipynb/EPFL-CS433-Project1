"""Ablation study of the final gradient boosting ensemble of ``run.py``.

All models are trained on the same stratified 80/20 split (seed 123) and
evaluated on the validation part with the same decision rule as ``run.py``:
the top ``POSITIVE_RATE`` fraction of scores is predicted positive.

To keep the run time reasonable, each ablation removes one component from a
*single* boosting model (seed 123) and is compared with the full single model.
The effect of the two-seed ensemble is reported separately.
"""

from pathlib import Path
import time

import numpy as np

from numpy_boosting import HistogramGradientBoosting, QuantileBinner
from project_utils import (
    ID_COLUMNS,
    choose_threshold,
    classification_metrics,
    load_dataset,
    stratified_split,
)
from run import MODEL_SEEDS, N_ESTIMATORS, POSITIVE_RATE, predict_top_fraction
from run import standardize

DATA_DIR = Path(__file__).resolve().parent / "dataset"
VALIDATION_FRACTION = 0.2
SPLIT_SEED = 123

# Must match the HistogramGradientBoosting call in run.py.
BOOSTING_PARAMS = dict(
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
)


def f1_top_fraction(scores, y_true, rate=POSITIVE_RATE):
    """F1 when the top `rate` fraction of scores is predicted positive."""
    return classification_metrics(y_true, predict_top_fraction(scores, rate))["f1"]


def binned_data(x_tr, x_val, keep):
    """Fit the quantile binner on the training part and bin both parts."""
    binner = QuantileBinner(max_bins=32, categorical_threshold=20).fit(x_tr[:, keep])
    return binner, binner.transform(x_tr[:, keep]), binner.transform(x_val[:, keep])


def train_scores(binner, bins_tr, y_tr, bins_val, seed=123, **overrides):
    """Train one boosting model and return it with its validation scores."""
    params = {**BOOSTING_PARAMS, **overrides, "seed": seed}
    start = time.time()

    def progress(iteration, _, __):
        if iteration % 50 == 0:
            print(f"    trees={iteration}/{params['n_estimators']}", flush=True)

    model = HistogramGradientBoosting(**params).fit(
        bins_tr,
        y_tr,
        binner.n_bins,
        categorical_features=binner.categorical_features,
        callback=progress,
    )
    print(f"    trained in {time.time() - start:.0f} s", flush=True)
    return model, model.decision_function(bins_val)


def report(label, f1):
    """Print one result line and the matching LaTeX table row."""
    print(f"{label:<45} F1 = {f1:.3f}", flush=True)
    print(f"    LaTeX: {label} & {f1:.3f} \\\\", flush=True)


def main():
    """Run the ablations and print the table rows."""
    x_train, _, y_train, _, _, feature_names = load_dataset(DATA_DIR)
    print("Data loaded", flush=True)
    train_idx, validation_idx = stratified_split(
        y_train, validation_fraction=VALIDATION_FRACTION, seed=SPLIT_SEED
    )
    x_tr, y_tr = x_train[train_idx], y_train[train_idx]
    x_val, y_val = x_train[validation_idx], y_train[validation_idx]

    keep = np.array([name not in ID_COLUMNS for name in feature_names], dtype=bool)
    binner, bins_tr, bins_val = binned_data(x_tr, x_val, keep)

    # --- full model and two-seed ensemble (same setting as run.py)
    print("Full model, seed", MODEL_SEEDS[0], flush=True)
    model, single_scores = train_scores(
        binner, bins_tr, y_tr, bins_val, seed=MODEL_SEEDS[0]
    )
    ensemble = [standardize(single_scores)]
    for seed in MODEL_SEEDS[1:]:
        print("Full model, seed", seed, flush=True)
        _, scores = train_scores(binner, bins_tr, y_tr, bins_val, seed=seed)
        ensemble.append(standardize(scores))
    ensemble_scores = np.mean(ensemble, axis=0)

    print("\n=== Final pipeline")
    report("Full pipeline (2-seed ensemble)", f1_top_fraction(ensemble_scores, y_val))
    report("single model (seed 123)", f1_top_fraction(single_scores, y_val))
    _, tuned = choose_threshold(ensemble_scores, y_val)
    report("tuned threshold instead of top fraction", tuned["f1"])

    # --- free diagnostics: number of trees and predicted positive rate
    print("\n=== Number of trees (single model)")
    checkpoints = [n for n in (50, 100, 150, 200) if n < N_ESTIMATORS]
    for n_trees in checkpoints + [N_ESTIMATORS]:
        scores = model.decision_function(bins_val, n_estimators=n_trees)
        print(f"  {n_trees:>4} trees: F1 = {f1_top_fraction(scores, y_val):.3f}")

    print("\n=== Predicted positive rate (ensemble)")
    for rate in (0.09, 0.11, 0.127, 0.14, 0.16):
        f1 = f1_top_fraction(ensemble_scores, y_val, rate)
        print(f"  rate {rate:.3f}: F1 = {f1:.3f}")

    # --- ablations, each compared with the single full model
    print("\n=== Ablations (single model, seed 123)")
    ablations = {
        "without class weights ($w_+=1$)": dict(positive_weight=1.0),
        "without categorical splits": dict(categorical_splits=False),
        "without learned missing direction": dict(learn_missing_direction=False),
    }
    for label, overrides in ablations.items():
        print(label, flush=True)
        _, scores = train_scores(binner, bins_tr, y_tr, bins_val, **overrides)
        report(label, f1_top_fraction(scores, y_val))

    print("keeping identifier and weight columns", flush=True)
    all_columns = np.ones(len(feature_names), dtype=bool)
    binner_all, bins_tr_all, bins_val_all = binned_data(x_tr, x_val, all_columns)
    _, scores = train_scores(binner_all, bins_tr_all, y_tr, bins_val_all)
    report("keeping identifier and weight columns", f1_top_fraction(scores, y_val))


if __name__ == "__main__":
    main()