"""Validation counterpart of ``run.py``.

The model is the same as in ``run.py`` (same excluded columns, boosting
parameters, seeds and top-fraction rule) but is trained on the training part
of the fixed stratified 80/20 split (seed 123) and evaluated on the validation
part. The script also prints the split-count feature importance.
"""

from collections import Counter
from pathlib import Path
import time

import numpy as np

from ablation import BOOSTING_PARAMS
from numpy_boosting import HistogramGradientBoosting, QuantileBinner
from project_utils import (
    ID_COLUMNS,
    LEAKAGE_COLUMNS,
    classification_metrics,
    load_dataset,
    stratified_split,
)
from run import MODEL_SEEDS, POSITIVE_RATE, predict_top_fraction, standardize

DATA_DIR = Path(__file__).resolve().parent / "dataset"
VALIDATION_FRACTION = 0.2
SPLIT_SEED = 123
TOP_K = 10


def split_counts(trees):
    """Count the distinct split nodes of every tree, per feature index."""
    counts = Counter()
    for tree in trees:
        splits = set()
        for rules, _ in tree:
            for feature, left_bins, _ in rules:
                splits.add((feature, left_bins.tobytes()))
        counts.update(feature for feature, _ in splits)
    return counts


def main():
    """Train the run.py model on 80% of the data and evaluate it on 20%."""
    x_train, _, y_train, _, _, feature_names = load_dataset(DATA_DIR)
    print("Data loaded", flush=True)
    train_idx, validation_idx = stratified_split(
        y_train, validation_fraction=VALIDATION_FRACTION, seed=SPLIT_SEED
    )

    excluded = ID_COLUMNS | LEAKAGE_COLUMNS
    keep = np.array([name not in excluded for name in feature_names], dtype=bool)
    kept_names = [name for name, k in zip(feature_names, keep) if k]
    x_tr, y_tr = x_train[train_idx][:, keep], y_train[train_idx]
    x_val, y_val = x_train[validation_idx][:, keep], y_train[validation_idx]
    print(f"{len(kept_names)} features (excluded: {sorted(LEAKAGE_COLUMNS)})")

    binner = QuantileBinner(max_bins=32, categorical_threshold=20).fit(x_tr)
    bins_tr, bins_val = binner.transform(x_tr), binner.transform(x_val)

    scores, models = [], []
    for seed in MODEL_SEEDS:
        start = time.time()
        print(f"Training seed {seed}", flush=True)

        def progress(iteration, _, __):
            if iteration % 50 == 0:
                print(f"    trees={iteration}", flush=True)

        model = HistogramGradientBoosting(**BOOSTING_PARAMS, seed=seed).fit(
            bins_tr,
            y_tr,
            binner.n_bins,
            categorical_features=binner.categorical_features,
            callback=progress,
        )
        print(f"    trained in {time.time() - start:.0f} s", flush=True)
        models.append(model)
        scores.append(standardize(model.decision_function(bins_val)))

    predictions = predict_top_fraction(np.mean(scores, axis=0), POSITIVE_RATE)
    metrics = classification_metrics(y_val, predictions)
    tp = np.sum((predictions == 1) & (y_val == 1))
    fp = np.sum((predictions == 1) & (y_val == -1))

    print("\n=== Validation (2-seed ensemble, top fraction)")
    for key in ("f1", "precision", "recall", "accuracy"):
        print(f"  {key:<10} {metrics[key]:.3f}")
    print(f"  false alarms per true positive: {fp / tp:.2f}")

    counts = split_counts(models[0].trees)
    total = sum(counts.values())
    print(f"\n=== Most used features (seed {MODEL_SEEDS[0]}, {total} splits)")
    for rank, (feature, count) in enumerate(counts.most_common(TOP_K), start=1):
        print(f"  {rank:>2}  {kept_names[feature]:<10} {count:>5} {count / total:6.1%}")


if __name__ == "__main__":
    main()

""" Output of the code last time, when it as run
(.venv) C:\MA3\ML\project_1_\EPFL-CS433-Project1>c:\MA3\ML\project_1_\.venv\Scripts\python.exe c:/MA3/ML/project_1_/EPFL-CS433-Project1/validate.py
Data loaded
312 features (excluded: ['HAREHAB1'])
Training seed 123
    trees=50
    trees=100
    trees=150
    trees=200
    trees=250
    trained in 1819 s
Training seed 321
    trees=50
    trees=100
    trees=150
    trees=200
    trees=250
    trained in 2096 s

=== Validation (2-seed ensemble, top fraction)
  f1         0.440
  precision  0.373
  recall     0.536
  accuracy   0.879
  false alarms per true positive: 1.68

=== Most used features (seed 123, 1704 splits)
   1  _AGEG5YR      97   5.7%
   2  GENHLTH       90   5.3%
   3  SEX           74   4.3%
   4  CVDSTRK3      59   3.5%
   5  JOINPAIN      47   2.8%
   6  EMPLOY1       41   2.4%
   7  _AGE80        40   2.3%
   8  DIABAGE2      39   2.3%
   9  FC60_         35   2.1%
  10  MAXVO2_       34   2.0%
  
  """
