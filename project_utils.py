"""Utilities for the project 1 prediction pipeline.

The code intentionally uses only the Python standard library and NumPy.
"""

import csv
from pathlib import Path

import numpy as np


ID_COLUMNS = {
    "IDATE",
    "SEQNO",
    "_PSU",
    "_STSTR",
    "_STRWT",
    "_WT2RAKE",
    "_LLCPWT",
    "_CLLCPWT",
}

CONTINUOUS_SENTINELS = np.array(
    [555, 5555, 777, 7777, 77777, 999, 9999, 99900, 999000, 999999],
    dtype=np.float32,
)


def load_dataset(data_dir):
    """Load train/test arrays, labels, ids, and feature names."""
    data_dir = Path(data_dir)
    with (data_dir / "x_train.csv").open() as handle:
        feature_names = next(csv.reader(handle))[1:]

    x_train = np.genfromtxt(
        data_dir / "x_train.csv",
        delimiter=",",
        skip_header=1,
        dtype=np.float32,
    )
    x_test = np.genfromtxt(
        data_dir / "x_test.csv",
        delimiter=",",
        skip_header=1,
        dtype=np.float32,
    )
    y_train = np.genfromtxt(
        data_dir / "y_train.csv",
        delimiter=",",
        skip_header=1,
        dtype=np.int8,
        usecols=1,
    )

    train_ids = x_train[:, 0].astype(np.int64)
    test_ids = x_test[:, 0].astype(np.int64)
    return (
        x_train[:, 1:],
        x_test[:, 1:],
        y_train,
        train_ids,
        test_ids,
        feature_names,
    )


def write_submission(ids, predictions, output_path):
    """Write predictions in the AIcrowd submission format."""
    ids = np.asarray(ids)
    predictions = np.asarray(predictions, dtype=int)
    if ids.shape[0] != predictions.shape[0]:
        raise ValueError("Ids and predictions must have the same length.")
    if not np.all((predictions == -1) | (predictions == 1)):
        raise ValueError("Predictions must contain only -1 and 1.")

    with Path(output_path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Id", "Prediction"])
        writer.writeheader()
        for row_id, prediction in zip(ids, predictions):
            writer.writerow({"Id": int(row_id), "Prediction": int(prediction)})


def stratified_split(y, validation_fraction=0.2, seed=123):
    """Return train and validation indices with preserved label balance."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1.")

    y = np.asarray(y)
    if not np.all((y == -1) | (y == 1)):
        raise ValueError("Labels must contain only -1 and 1.")

    rng = np.random.default_rng(seed)
    train_parts = []
    validation_parts = []

    for label in (-1, 1):
        indices = np.where(y == label)[0]
        rng.shuffle(indices)
        cut = int(round(validation_fraction * indices.size))
        validation_parts.append(indices[:cut])
        train_parts.append(indices[cut:])

    train_idx = np.concatenate(train_parts)
    validation_idx = np.concatenate(validation_parts)
    rng.shuffle(train_idx)
    rng.shuffle(validation_idx)
    return train_idx, validation_idx


def classification_metrics(y_true, y_pred):
    """Compute accuracy, precision, recall, and F1 for positive label 1."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == -1) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == -1))

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1_score = (
        2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    accuracy = np.mean(y_true == y_pred)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1_score),
    }


def choose_threshold(scores, y_true, quantile_count=500):
    """Choose the score threshold that maximizes validation F1."""
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)
    if scores.shape != y_true.shape:
        raise ValueError("Scores and labels must have the same shape.")
    if quantile_count < 2:
        raise ValueError("quantile_count must be at least 2.")

    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size == 0:
        raise ValueError("At least one score must be finite.")
    candidates = np.unique(
        np.quantile(finite_scores, np.linspace(0.01, 0.99, quantile_count))
    )

    best_threshold = float(candidates[0])
    best_metrics = None
    best_f1 = -1.0

    for threshold in candidates:
        predictions = np.where(scores >= threshold, 1, -1)
        metrics = classification_metrics(y_true, predictions)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


class Preprocessor:
    """Fit and apply a compact BRFSS feature representation."""

    def __init__(self, categorical_threshold=20, max_missing_fraction=0.995):
        self.categorical_threshold = categorical_threshold
        self.max_missing_fraction = max_missing_fraction

    def fit(self, x, feature_names):
        """Learn column types, categorical values, and numeric statistics."""
        missing_fraction = np.isnan(x).mean(axis=0)
        self.categorical_columns = []
        self.categorical_values = []
        self.numeric_columns = []

        for column, name in enumerate(feature_names):
            if missing_fraction[column] >= self.max_missing_fraction:
                continue
            if name in ID_COLUMNS:
                continue

            values = x[:, column]
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                continue

            unique_values = np.unique(finite_values)
            if unique_values.size <= self.categorical_threshold:
                self.categorical_columns.append(column)
                self.categorical_values.append(unique_values.astype(np.float32))
            else:
                self.numeric_columns.append(column)

        numeric = x[:, self.numeric_columns].astype(np.float32, copy=True)
        numeric[np.isin(numeric, CONTINUOUS_SENTINELS)] = np.nan
        missing_numeric = np.isnan(numeric)
        self.numeric_missing_indices = np.where(missing_numeric.mean(axis=0) > 0.005)[0]

        self.numeric_median = np.nanmedian(numeric, axis=0).astype(np.float32)
        self.numeric_median = np.where(
            np.isfinite(self.numeric_median),
            self.numeric_median,
            0.0,
        ).astype(np.float32)

        row_idx, col_idx = np.where(np.isnan(numeric))
        if row_idx.size:
            numeric[row_idx, col_idx] = self.numeric_median[col_idx]

        self.numeric_low = np.nanpercentile(numeric, 1, axis=0).astype(np.float32)
        self.numeric_high = np.nanpercentile(numeric, 99, axis=0).astype(np.float32)
        invalid_range = ~(self.numeric_high > self.numeric_low)
        self.numeric_high[invalid_range] = self.numeric_low[invalid_range] + 1.0

        numeric = np.minimum(np.maximum(numeric, self.numeric_low), self.numeric_high)
        self.numeric_mean = numeric.mean(axis=0).astype(np.float32)
        self.numeric_std = numeric.std(axis=0).astype(np.float32)
        self.numeric_std = np.where(self.numeric_std > 1e-6, self.numeric_std, 1.0)
        self.numeric_std = self.numeric_std.astype(np.float32)

        self.n_features_out = (
            1
            + len(self.numeric_missing_indices)
            + len(self.numeric_columns)
            + sum(values.size + 1 for values in self.categorical_values)
        )
        return self

    def transform(self, x):
        """Transform raw features into the fitted model matrix."""
        transformed = np.empty((x.shape[0], self.n_features_out), dtype=np.float32)
        write_col = 0
        transformed[:, write_col] = 1.0
        write_col += 1

        numeric = x[:, self.numeric_columns].astype(np.float32, copy=True)
        numeric[np.isin(numeric, CONTINUOUS_SENTINELS)] = np.nan
        missing_numeric = np.isnan(numeric)

        if self.numeric_missing_indices.size:
            end_col = write_col + self.numeric_missing_indices.size
            transformed[:, write_col:end_col] = missing_numeric[
                :, self.numeric_missing_indices
            ].astype(np.float32)
            write_col = end_col

        row_idx, col_idx = np.where(missing_numeric)
        if row_idx.size:
            numeric[row_idx, col_idx] = self.numeric_median[col_idx]

        numeric = np.minimum(np.maximum(numeric, self.numeric_low), self.numeric_high)
        end_col = write_col + len(self.numeric_columns)
        transformed[:, write_col:end_col] = (
            numeric - self.numeric_mean
        ) / self.numeric_std
        write_col = end_col

        for column, values in zip(self.categorical_columns, self.categorical_values):
            raw_column = x[:, column]
            finite = np.isfinite(raw_column)
            for value in values:
                transformed[:, write_col] = (finite & (raw_column == value)).astype(
                    np.float32
                )
                write_col += 1

            transformed[:, write_col] = (~finite).astype(np.float32)
            write_col += 1

        return transformed


def fit_weighted_ridge(
    preprocessor,
    x,
    y,
    lambda_,
    positive_weight=8.0,
    batch_size=32768,
):
    """Fit a weighted ridge model from transformed chunks."""
    n_samples = y.shape[0]
    n_features = preprocessor.n_features_out
    gram = np.zeros((n_features, n_features), dtype=np.float64)
    rhs = np.zeros(n_features, dtype=np.float64)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        features = preprocessor.transform(x[start:end]).astype(np.float64)
        targets = y[start:end].astype(np.float64)
        weights = np.where(targets == 1.0, positive_weight, 1.0)
        weighted_features = features * weights[:, None]
        gram += features.T @ weighted_features
        rhs += features.T @ (weights * targets)

    penalty = 2.0 * n_samples * lambda_ * np.eye(n_features)
    penalty[0, 0] = 0.0
    system = gram + penalty
    try:
        weights = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(system) @ rhs
    return weights.astype(np.float32)


def predict_scores(preprocessor, weights, x, batch_size=32768):
    """Predict model scores from raw features in memory-bounded chunks."""
    scores = np.empty(x.shape[0], dtype=np.float32)
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        scores[start:end] = preprocessor.transform(x[start:end]) @ weights
    return scores
