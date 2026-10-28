# EPFL CS-433 Project 1

This repository contains a complete NumPy-only solution for project 1 on
MICHD prediction from the BRFSS medical survey dataset.

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

- `dataset/`: expected location for `x_train.csv`, `y_train.csv`, and
  `x_test.csv`.

All validation scripts use the same stratified 80/20 split (seed 123).

## Reproducing the submission

Create the grading environment from the provided file, then run:

```bash
python run.py
```

The script loads the CSV files from `dataset/`, trains two categorical
histogram-boosting models on all training rows, bags their standardized scores,
and writes `submission.csv`. Its trees evaluate both ordered thresholds and
learned category groupings, choose missing-value directions explicitly, and
model three levels of interactions.

At 250 trees, this configuration reaches F1 0.4506 and 0.4517 on two fixed
validation splits. The nearby 240--270 tree plateau has mean F1 about 0.4512.
The final script uses seeds 123 and 321 and labels the top 12.7% of test scores
as positive.

The public AIcrowd results provide the more useful selection signal: the
previous 240-tree pure boosting ensemble scored F1 0.448 and accuracy 0.881,
whereas the previous ridge blend scored F1 0.447 and accuracy 0.885. These
leaderboard results motivated retaining pure boosting and replacing the ridge
blend with the categorical depth-3 model.

To reproduce the compact hyperparameter comparison reported in the project,
run:

```bash
python experiments.py
```

Run the local and public tests with:

```bash
pytest -q --github_link /path/to/repository
```

Ridge baseline hyperparameters:

- model: weighted ridge regression
- categorical one-hot threshold: 20 unique finite values
- ridge regularization `lambda`: 0.005
- positive class weight: 4.0
- validation seed: 123

Categorical boosting hyperparameters:

- trees: 250 per model, seeds 123 and 321
- depth: 3
- learning rate: 0.06
- maximum bins: 32
- sampled features per tree: 160
- minimum samples per leaf: 300
- positive class weight: 1.5

## Method summary

The preprocessing removes survey identifier and sampling-weight columns, drops
almost-empty columns, one-hot encodes low-cardinality variables, imputes and
standardizes continuous variables, and adds missing-value indicators for
continuous variables with non-negligible missingness. It also includes a small
set of age-by-risk-factor interactions that improved F1 on three validation
splits.

The local validation metric is F1 for the positive MICHD class, because the
dataset is highly imbalanced. The weighted ridge baseline reaches approximately:

- accuracy: 0.873
- precision: 0.359
- recall: 0.564
- F1: 0.439

No external machine-learning library is used.
