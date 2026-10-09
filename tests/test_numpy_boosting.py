"""Smoke tests for the NumPy-only boosting implementation."""

import numpy as np

from numpy_boosting import HistogramGradientBoosting, QuantileBinner
from project_utils import choose_threshold
from run import predict_top_fraction


def test_boosting_learns_small_nonlinear_problem():
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] * 100)
    y = np.where(x[:, 0] + x[:, 1] > 0, 1, -1)
    binner = QuantileBinner(max_bins=8).fit(x)
    bins = binner.transform(x)
    model = HistogramGradientBoosting(
        n_estimators=30,
        max_features=2,
        min_samples_leaf=10,
        seed=7,
    ).fit(bins, y, binner.n_bins)

    scores = model.decision_function(bins)
    threshold, metrics = choose_threshold(scores, y)
    predictions = np.where(scores >= threshold, 1, -1)

    assert metrics["f1"] == 1.0
    np.testing.assert_array_equal(predictions, y)


def test_categorical_split_can_group_non_adjacent_values():
    x = np.repeat(np.arange(4, dtype=float), 100)[:, None]
    y = np.where(np.isin(x[:, 0], [0.0, 3.0]), 1, -1)
    binner = QuantileBinner(max_bins=8, categorical_threshold=4).fit(x)
    bins = binner.transform(x)
    model = HistogramGradientBoosting(
        n_estimators=1,
        max_depth=1,
        max_features=1,
        min_samples_leaf=10,
        categorical_splits=True,
        seed=7,
    ).fit(
        bins,
        y,
        binner.n_bins,
        categorical_features=binner.categorical_features,
    )

    scores = model.decision_function(bins)
    _, metrics = choose_threshold(scores, y)

    assert metrics["f1"] == 1.0


def test_invalid_feature_sampling_is_rejected():
    with np.testing.assert_raises(ValueError):
        HistogramGradientBoosting(feature_sampling="branch")


def test_predict_top_fraction_returns_exact_count():
    scores = np.arange(100, dtype=float)
    predictions = predict_top_fraction(scores, 0.13)

    assert np.sum(predictions == 1) == 13
    assert np.all(predictions[-13:] == 1)
    assert np.all(predictions[:-13] == -1)
