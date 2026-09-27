"""Focused regression tests for the project pipeline."""

import numpy as np
import pytest

from project_utils import (
    Preprocessor,
    choose_threshold,
    classification_metrics,
    stratified_split,
    write_submission,
)


def test_stratified_split_is_reproducible_and_disjoint():
    y = np.array([-1] * 90 + [1] * 10)
    train_a, validation_a = stratified_split(y, seed=7)
    train_b, validation_b = stratified_split(y, seed=7)

    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(validation_a, validation_b)
    assert np.intersect1d(train_a, validation_a).size == 0
    assert np.mean(y[validation_a] == 1) == pytest.approx(0.1)


def test_preprocessor_fits_statistics_only_on_fit_data():
    x_fit = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, np.nan]], dtype=np.float32)
    x_other = np.array([[1000.0, 2.0]], dtype=np.float32)
    preprocessor = Preprocessor(categorical_threshold=2).fit(x_fit, ["a", "b"])

    transformed = preprocessor.transform(x_other)

    assert transformed.shape[0] == 1
    assert np.isfinite(transformed).all()
    assert preprocessor.numeric_high[0] < 1000.0


def test_threshold_selection_and_metrics_agree():
    scores = np.array([-2.0, -1.0, 0.2, 0.8, 1.5])
    y_true = np.array([-1, -1, 1, -1, 1])
    threshold, metrics = choose_threshold(scores, y_true, quantile_count=50)
    predictions = np.where(scores >= threshold, 1, -1)

    assert metrics == classification_metrics(y_true, predictions)
    assert metrics["f1"] > 0.0


def test_submission_rejects_invalid_input(tmp_path):
    with pytest.raises(ValueError):
        write_submission([1, 2], [1], tmp_path / "submission.csv")
    with pytest.raises(ValueError):
        write_submission([1, 2], [1, 0], tmp_path / "submission.csv")
