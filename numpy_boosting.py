"""Small NumPy-only histogram gradient boosting classifier.

The implementation is intentionally compact and tailored to medium-cardinality
survey data. It uses second-order logistic-loss updates and shallow trees.
"""

import numpy as np


def _sigmoid(values):
    values = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-values))


class QuantileBinner:
    """Map numeric survey columns to compact ordered integer bins."""

    def __init__(self, max_bins=32, categorical_threshold=20):
        self.max_bins = max_bins
        self.categorical_threshold = categorical_threshold

    def fit(self, x):
        """Learn bin boundaries from training data."""
        self.boundaries = []
        self.categorical_features = np.zeros(x.shape[1], dtype=bool)
        for column in range(x.shape[1]):
            values = x[:, column]
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                boundaries = np.empty(0, dtype=np.float32)
            else:
                unique = np.unique(finite)
                self.categorical_features[column] = (
                    unique.size <= self.categorical_threshold
                )
                if unique.size < self.max_bins:
                    boundaries = ((unique[:-1] + unique[1:]) / 2.0).astype(np.float32)
                else:
                    quantiles = np.linspace(0.0, 1.0, self.max_bins + 1)[1:-1]
                    boundaries = np.unique(np.quantile(finite, quantiles)).astype(
                        np.float32
                    )
            self.boundaries.append(boundaries)
        return self

    def transform(self, x):
        """Apply fitted boundaries and reserve the last bin for missing data."""
        binned = np.empty(x.shape, dtype=np.uint8, order="F")
        self.n_bins = np.empty(x.shape[1], dtype=np.int16)
        for column, boundaries in enumerate(self.boundaries):
            values = x[:, column]
            finite = np.isfinite(values)
            bins = np.searchsorted(boundaries, values, side="right").astype(np.uint8)
            missing_bin = boundaries.size + 1
            bins[~finite] = missing_bin
            binned[:, column] = bins
            self.n_bins[column] = missing_bin + 1
        return np.asfortranarray(binned)


class HistogramGradientBoosting:
    """Binary logistic gradient boosting with fixed-depth histogram trees."""

    def __init__(
        self,
        n_estimators=120,
        learning_rate=0.06,
        max_depth=2,
        l2=20.0,
        min_samples_leaf=300,
        max_features=160,
        positive_weight=1.5,
        categorical_splits=False,
        learn_missing_direction=False,
        feature_sampling="tree",
        seed=123,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.l2 = l2
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.positive_weight = positive_weight
        self.categorical_splits = categorical_splits
        self.learn_missing_direction = learn_missing_direction
        if feature_sampling not in {"tree", "node"}:
            raise ValueError("feature_sampling must be 'tree' or 'node'.")
        self.feature_sampling = feature_sampling
        self.seed = seed

    def _best_split(
        self,
        x,
        gradient,
        hessian,
        indices,
        feature_indices,
        n_bins,
        categorical_features,
    ):
        """Find the highest-gain ordered-bin split for one tree node."""
        if indices.size < 2 * self.min_samples_leaf:
            return None
        total_gradient = gradient[indices].sum()
        total_hessian = hessian[indices].sum()
        parent_gain = total_gradient**2 / (total_hessian + self.l2)
        best = None
        best_gain = 0.0

        for feature in feature_indices:
            bins = x[indices, feature]
            feature_bin_count = int(n_bins[feature])
            bin_count = np.bincount(bins, minlength=feature_bin_count)
            bin_gradient = np.bincount(
                bins,
                weights=gradient[indices],
                minlength=feature_bin_count,
            )
            bin_hessian = np.bincount(
                bins,
                weights=hessian[indices],
                minlength=feature_bin_count,
            )

            candidate_orders = [np.arange(feature_bin_count)]
            if self.categorical_splits and categorical_features[feature]:
                active_bins = np.where(bin_count > 0)[0]
                if active_bins.size > 1:
                    newton_value = bin_gradient[active_bins] / (
                        bin_hessian[active_bins] + 1e-12
                    )
                    candidate_orders.append(active_bins[np.argsort(newton_value)])

            for order in candidate_orders:
                if order.size < 2:
                    continue
                ordered_count = bin_count[order]
                ordered_gradient = bin_gradient[order]
                ordered_hessian = bin_hessian[order]
                left_count = np.cumsum(ordered_count)[:-1]
                valid = (left_count >= self.min_samples_leaf) & (
                    indices.size - left_count >= self.min_samples_leaf
                )
                if not np.any(valid):
                    continue
                left_gradient = np.cumsum(ordered_gradient)[:-1]
                left_hessian = np.cumsum(ordered_hessian)[:-1]
                right_gradient = total_gradient - left_gradient
                right_hessian = total_hessian - left_hessian
                gains = (
                    left_gradient**2 / (left_hessian + self.l2)
                    + right_gradient**2 / (right_hessian + self.l2)
                    - parent_gain
                )
                gains[~valid] = -np.inf
                split = int(np.argmax(gains))
                gain = float(gains[split])
                if gain > best_gain:
                    left_bins = np.zeros(feature_bin_count, dtype=bool)
                    left_bins[order[: split + 1]] = True
                    best_gain = gain
                    best = (int(feature), left_bins)

            uses_categorical_split = (
                self.categorical_splits and categorical_features[feature]
            )
            if self.learn_missing_direction and not uses_categorical_split:
                missing_bin = feature_bin_count - 1
                missing_count = bin_count[missing_bin]
                if missing_count == 0:
                    continue
                finite_count = np.cumsum(bin_count[:-1])
                finite_gradient = np.cumsum(bin_gradient[:-1])
                finite_hessian = np.cumsum(bin_hessian[:-1])
                left_count = finite_count + missing_count
                valid = (left_count >= self.min_samples_leaf) & (
                    indices.size - left_count >= self.min_samples_leaf
                )
                if not np.any(valid):
                    continue
                left_gradient = finite_gradient + bin_gradient[missing_bin]
                left_hessian = finite_hessian + bin_hessian[missing_bin]
                right_gradient = total_gradient - left_gradient
                right_hessian = total_hessian - left_hessian
                gains = (
                    left_gradient**2 / (left_hessian + self.l2)
                    + right_gradient**2 / (right_hessian + self.l2)
                    - parent_gain
                )
                gains[~valid] = -np.inf
                split = int(np.argmax(gains))
                gain = float(gains[split])
                if gain > best_gain:
                    left_bins = np.zeros(feature_bin_count, dtype=bool)
                    left_bins[: split + 1] = True
                    left_bins[missing_bin] = True
                    best_gain = gain
                    best = (int(feature), left_bins)
        return best

    def _fit_tree(
        self,
        x,
        gradient,
        hessian,
        n_bins,
        categorical_features,
        rng,
    ):
        """Fit one shallow tree and return its leaf rules and values."""
        feature_count = min(self.max_features, x.shape[1])
        tree_features = rng.choice(x.shape[1], feature_count, replace=False)
        nodes = [(np.arange(x.shape[0]), [])]
        leaves = []

        for depth in range(self.max_depth + 1):
            next_nodes = []
            for indices, rules in nodes:
                if depth == self.max_depth:
                    leaves.append((indices, rules))
                    continue
                if self.feature_sampling == "node":
                    features = rng.choice(x.shape[1], feature_count, replace=False)
                else:
                    features = tree_features
                split = self._best_split(
                    x,
                    gradient,
                    hessian,
                    indices,
                    features,
                    n_bins,
                    categorical_features,
                )
                if split is None:
                    leaves.append((indices, rules))
                    continue
                feature, left_bins = split
                go_left = left_bins[x[indices, feature]]
                next_nodes.append(
                    (indices[go_left], rules + [(feature, left_bins, True)])
                )
                next_nodes.append(
                    (indices[~go_left], rules + [(feature, left_bins, False)])
                )
            nodes = next_nodes
        leaves.extend(nodes)

        tree = []
        for indices, rules in leaves:
            value = gradient[indices].sum() / (hessian[indices].sum() + self.l2)
            tree.append((rules, float(np.clip(value, -3.0, 3.0))))
        return tree

    @staticmethod
    def _tree_output(tree, x):
        output = np.zeros(x.shape[0], dtype=np.float32)
        for rules, value in tree:
            selected = np.ones(x.shape[0], dtype=bool)
            for feature, left_bins, is_left in rules:
                if is_left:
                    selected &= left_bins[x[:, feature]]
                else:
                    selected &= ~left_bins[x[:, feature]]
            output[selected] = value
        return output

    def fit(
        self,
        x,
        y,
        n_bins,
        categorical_features=None,
        validation_data=None,
        callback=None,
    ):
        """Fit boosted trees for labels encoded as -1 and 1."""
        if categorical_features is None:
            categorical_features = np.zeros(x.shape[1], dtype=bool)
        categorical_features = np.asarray(categorical_features, dtype=bool)
        if categorical_features.shape != (x.shape[1],):
            raise ValueError("categorical_features must match the feature count.")
        target = (y == 1).astype(np.float64)
        sample_weight = np.where(target == 1.0, self.positive_weight, 1.0)
        weighted_rate = np.sum(sample_weight * target) / np.sum(sample_weight)
        self.initial_score = float(np.log(weighted_rate / (1.0 - weighted_rate)))
        scores = np.full(y.shape[0], self.initial_score, dtype=np.float64)
        self.trees = []
        rng = np.random.default_rng(self.seed)

        if validation_data is not None:
            validation_x, validation_y = validation_data
            validation_scores = np.full(
                validation_y.shape[0], self.initial_score, dtype=np.float64
            )
        else:
            validation_scores = None

        for iteration in range(self.n_estimators):
            probability = _sigmoid(scores)
            gradient = sample_weight * (target - probability)
            hessian = sample_weight * probability * (1.0 - probability)
            tree = self._fit_tree(
                x,
                gradient,
                hessian,
                n_bins,
                categorical_features,
                rng,
            )
            self.trees.append(tree)
            scores += self.learning_rate * self._tree_output(tree, x)
            if validation_scores is not None:
                validation_scores += self.learning_rate * self._tree_output(
                    tree, validation_x
                )
            if callback is not None:
                callback(iteration + 1, scores, validation_scores)
        return self

    def decision_function(self, x, n_estimators=None):
        """Return raw scores, optionally at an earlier boosting checkpoint."""
        scores = np.full(x.shape[0], self.initial_score, dtype=np.float64)
        trees = self.trees if n_estimators is None else self.trees[:n_estimators]
        for tree in trees:
            scores += self.learning_rate * self._tree_output(tree, x)
        return scores
