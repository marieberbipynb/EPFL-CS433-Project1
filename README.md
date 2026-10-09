# EPFL CS-433 Project 1

This repository contains a complete NumPy-only solution for project 1 on
MICHD prediction from the BRFSS medical survey dataset.

## Files

- `implementations.py`: the six mandatory methods requested in the project
  statement.
- `project_utils.py`: loading, preprocessing, metrics, threshold selection,
  and weighted ridge training utilities.
- `experiments.py`: reproducible comparison of ridge penalties and positive
  class weights on the validation split.
- `numpy_boosting.py`: NumPy implementation of histogram gradient boosting.
- `boosting_experiment.py`: reproducible validation of the selected model.
- `run.py`: deterministic end-to-end script that creates the final
  `submission.csv`.
- `dataset/`: expected location for `x_train.csv`, `y_train.csv`, and
  `x_test.csv`.

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
