# EPFL CS-433 Project 1

NumPy-only prediction of coronary heart disease (MICHD) from the BRFSS 2015
survey. The final model is a gradient boosting ensemble written from scratch
in NumPy (`numpy_boosting.py`). The six methods of the project statement are
used as linear baselines.

## Authors

- **Marie Berbichashvili**: design and implementation of the whole code base: data
  loading and preprocessing, the six required methods, the weighted ridge
  regression, the NumPy gradient boosting model written from scratch
  (quantile binning, categorical splits, missing-value handling), and the
  `run.py` pipeline.

- **Nafi Anna Dramé**: running all models under the same protocol (same
  validation split and metric) to compare them, the ablation study,
  interpreting the results, including the detection of the `HAREHAB1`
  leakage, the exploratory data analysis, and writing the report.

AIcrowd team: `WMN` (`marieberbipynb`, `Nafi_Drm`).

## Files

- `run.py`: deterministic end-to-end script that trains the final model on
  all training rows and writes `submission.csv`. The final model is a
  two-seed gradient boosting ensemble trained on the raw features, without
  identifiers, survey weights and the leaky `HAREHAB1` column.

- `implementations.py`: the six mandatory methods requested in the project
  statement.

- `numpy_boosting.py`: NumPy implementation of quantile binning and
  histogram gradient boosting.

- `project_utils.py`: data loading, excluded column lists (`ID_COLUMNS`,
  `LEAKAGE_COLUMNS`), metrics, stratified split, and the preprocessing used by
  the linear baselines.

- `validate.py`: validation counterpart of `run.py`: trains the same model on
  80% of the training rows and reports validation F1, precision, recall and
  the most used features.

- `baselines.py`: validation F1 of the six linear methods (Table I of the
  report).

- `ablation.py`: ablation study of the boosting model (Table II). It was run
  before removing `HAREHAB1`, so it evaluates the model with this column.

- `feature_importance.py`: split-count feature importance of the model with
  `HAREHAB1`, used to detect its label leakage.

- `exploratory_data_analysis.ipynb`: exploratory data analysis and report
  figures.

- `tests/`: unit tests of `numpy_boosting.py` and `project_utils.py`.

- `dataset/`: expected location for `x_train.csv`, `y_train.csv`, and
  `x_test.csv`.

  - `report/report.pdf`: the project report.

## Submission

`run.py` produces our final submission: **F1 0.448, accuracy 0.882** on
AIcrowd.

The key step of the project was replacing the linear models by a gradient
boosting model implemented from scratch in NumPy, which handles categorical
and missing values natively and gave our best results. Our best leaderboard
score (F1 0.454) comes from this same model with the `HAREHAB1` column, which
we then removed because it leaks the label (see the report).


## Method summary

**Evaluation.** The data is highly imbalanced (8.83% positives), so the metric
is the F1 score of the positive class, computed on the 20% validation split.

**Final model (gradient boosting).** Identifiers, survey weights and
`HAREHAB1` are removed, leaving 312 raw features. Each feature is mapped to at
most 32 ordered quantile bins, plus one bin for missing values; features with
at most 20 distinct values are treated as categorical. Trees of depth 3 are
grown with second-order (Newton) updates of the logistic loss, with a class
weight of 1.5 for positives. For categorical features, bins are sorted by their
gradient/hessian ratio before searching the best split; for the others, the
tree also learns in which branch missing values go. The scores of two models
(seeds 123 and 321) are standardized and averaged, and the 12.7% highest
scores are predicted positive.

Hyperparameters: 250 trees per model, depth 3, learning rate 0.06, L2
regularization 20, at least 300 samples per leaf, 160 features sampled per
tree, at most 32 bins, positive class weight 1.5, predicted positive rate
12.7%.

**Linear baselines.** For the six methods of `implementations.py`, survey
identifiers, sampling weights and features missing for at least 99.5% of
respondents are removed. Features with at most 20 distinct values are one-hot
encoded with a missing category. For continuous features, special codes are
set to missing, then values are median-imputed, clipped to the 1st-99th
percentiles and standardized, with a missing-value indicator when more than
0.5% are missing. Interactions between the age group and 11 risk factors are
added, giving 2,252 features. Ridge regression uses `lambda = 0.01` and the
regularized logistic regression `lambda = 1e-4`; gradient-based methods run
500 iterations (50,000 for SGD), and each model is evaluated with a decision
threshold tuned on the validation set.

**Results (validation F1).** Best linear baseline 0.434 (least squares and
ridge regression), final gradient boosting 0.440 (0.449 with the leaky
`HAREHAB1`).

No external machine-learning library is used: only NumPy and the Python
standard library (matplotlib for the figures of the notebook).