"""Split-count feature importance of the final gradient boosting model.

The script trains one boosting model with the same settings as ``run.py`` on
the full training set and counts, for every feature, how many tree nodes split
on it. Every ``CHECK_EVERY`` trees it prints the current top features, and it
stops training as soon as the top-``TOP_K`` set has stayed the same for
``STABLE_CHECKS`` consecutive checks (or after ``n_estimators`` trees).
"""

from collections import Counter
from pathlib import Path

import numpy as np

from ablation import BOOSTING_PARAMS
from numpy_boosting import HistogramGradientBoosting, QuantileBinner
from project_utils import ID_COLUMNS, load_dataset

DATA_DIR = Path(__file__).resolve().parent / "dataset"
SEED = 123
TOP_K = 10
CHECK_EVERY = 25
STABLE_CHECKS = 5
MIN_TREES = 50


class RankingStable(Exception):
    """Raised from the training callback to stop once the ranking is stable."""


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


def print_ranking(counts, names):
    """Print the most used features with their share of all splits."""
    total = sum(counts.values())
    print(f"{'rank':>4}  {'feature':<10} {'splits':>7} {'share':>7}")
    for rank, (feature, count) in enumerate(counts.most_common(TOP_K), start=1):
        print(f"{rank:>4}  {names[feature]:<10} {count:>7} {count / total:>7.1%}")


def main():
    """Train the model until the ranking is stable and print it."""
    x_train, _, y_train, _, _, feature_names = load_dataset(DATA_DIR)
    print("Data loaded", flush=True)
    keep = np.array([name not in ID_COLUMNS for name in feature_names], dtype=bool)
    kept_names = [name for name, k in zip(feature_names, keep) if k]

    binner = QuantileBinner(max_bins=32, categorical_threshold=20).fit(x_train[:, keep])
    bins = binner.transform(x_train[:, keep])

    model = HistogramGradientBoosting(**BOOSTING_PARAMS, seed=SEED)
    history = []

    def check_ranking(iteration, _, __):
        if iteration % CHECK_EVERY != 0:
            return
        top = {f for f, _ in split_counts(model.trees).most_common(TOP_K)}
        unchanged = bool(history) and top == history[-1]
        history.append(top)
        status = "unchanged" if unchanged else "changed"
        print(f"  trees={iteration}: top-{TOP_K} {status}", flush=True)
        recent = history[-(STABLE_CHECKS + 1) :]
        stable = len(recent) == STABLE_CHECKS + 1 and all(
            t == recent[0] for t in recent
        )
        if iteration >= MIN_TREES and stable:
            raise RankingStable

    try:
        model.fit(
            bins,
            y_train,
            binner.n_bins,
            categorical_features=binner.categorical_features,
            callback=check_ranking,
        )
        print(f"\nReached {len(model.trees)} trees without a stable ranking.")
    except RankingStable:
        print(f"\nTop-{TOP_K} stable, stopped after {len(model.trees)} trees.")

    counts = split_counts(model.trees)
    print(f"{sum(counts.values())} splits in {len(model.trees)} trees")
    print(f"{len(counts)} of {len(kept_names)} features are used at least once\n")
    print_ranking(counts, kept_names)


if __name__ == "__main__":
    main()
