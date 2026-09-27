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
- `run.py`: deterministic end-to-end script that creates `submission.csv`.
- `report.tex` and `report.pdf`: short project report.
- `dataset/`: expected location for `x_train.csv`, `y_train.csv`, and
  `x_test.csv`.

## Reproducing the submission

Create the grading environment from the provided file, then run:

```bash
python run.py
```

The script loads the CSV files from `dataset/`, performs a fixed stratified
80/20 train-validation split, selects the decision threshold on the validation
set, retrains the selected model on the full training set, and writes
`submission.csv`.

To reproduce the compact hyperparameter comparison reported in the project,
run:

```bash
python experiments.py
```

Run the local and public tests with:

```bash
pytest -q
```

Final hyperparameters:

- model: weighted ridge regression
- categorical one-hot threshold: 20 unique finite values
- ridge regularization `lambda`: 0.01
- positive class weight: 4.0
- validation seed: 123

## Method summary

The preprocessing removes survey identifier and sampling-weight columns, drops
almost-empty columns, one-hot encodes low-cardinality variables, imputes and
standardizes continuous variables, and adds missing-value indicators for
continuous variables with non-negligible missingness.

The local validation metric is F1 for the positive MICHD class, because the
dataset is highly imbalanced. On the fixed validation split used in `run.py`,
the selected configuration reaches approximately:

- accuracy: 0.876
- precision: 0.364
- recall: 0.544
- F1: 0.436

No external machine-learning library is used.
