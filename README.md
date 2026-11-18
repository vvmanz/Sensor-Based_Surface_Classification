# lb2_kurenko — Sensor-Based Surface Classification

**PyTorch binary classification of surface type from robot sensor measurements, with Optuna tuning and architecture/data-ablation studies.**

English | [Русский](README.ru.md)

This experimental machine-learning project investigates whether a neural network can distinguish a selected surface category from sensor measurements. The target is built from the dataset's `Type` column: **class 1 = `Type == 2`**, **class 0 = other types**. The repository includes training/testing scripts, saved model and scaler files, original Excel tables and plots comparing design choices.

![Activation function study](activation_analysis.png)

## Features

- **Binary PyTorch classifier** using a multilayer perceptron (`nn.Linear`, activations and dropout).
- **15 sensor features:** current (`I1`–`I3`), gyroscope (`gx`–`gz`), accelerometer (`ax`–`az`), voltages (`V1real`–`V3real`) and `N1`–`N3`.
- **Optuna optimization** of layer size/count, learning rate, batch size, dropout and activation.
- **Class imbalance handling** with class-weighted cross-entropy; separate imbalance/balancing experiments.
- **Model inspection:** F1, accuracy, confusion matrix, classification report and ROC/AUC in `testing.py`.
- **Ablation studies** on activations, epochs, optimizers, number of layers/neurons, random state, balancing and input normalization.

## Requirements

Python 3.10+ recommended, with `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `optuna`, `joblib`, `openpyxl` and `imbalanced-learn` (imported by the training experiments).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch numpy pandas scikit-learn matplotlib optuna joblib openpyxl imbalanced-learn
```

## Data

| Input / artifact | Description |
|---|---|
| `raw/train.xlsx`, `raw/test.xlsx` | Original tabular train/test sets |
| `train_augmented.xlsx` | Prepared dataset used by `main.py` |
| `train_imbalanced.xlsx`, `train_unnormalized.xlsx`, `train_unsorted.xlsx` | Experiment-specific variants |
| `scaler_v1_features.pkl` | Saved feature scaler loaded by `testing.py` |
| `model.pt` | Saved neural-network checkpoint |

`main.py` directly loads `train_augmented.xlsx`. The preprocessing work is also recorded in `preparation_dataset.ipynb`; the precise relation between all dataset variants should be reviewed before treating their results as strictly comparable.

## Train and evaluate

From the repository root:

```bash
python main.py        # Optuna search, final training, saves model.pt
python testing.py     # loads model.pt and scaler_v1_features.pkl, evaluates raw/test.xlsx
```

**Note:** `main.py` reads the preprocessed augmented dataset directly, whereas `testing.py` applies the saved scaler to the raw test set. Verify that the training and inference feature transformations are consistent before using metrics or deploying the checkpoint.

## Experiments and illustrations

| Script | Comparison | Saved plot |
|---|---|---|
| `activation_analysis.py` | Activation functions | [`activation_analysis.png`](activation_analysis.png) |
| `epochs_analysis.py` | Training epochs | [`epochs_analysis.png`](epochs_analysis.png) |
| `layers_analysis.py` | Network depth | [`layers_analysis.png`](layers_analysis.png) |
| `neurons_analysis.py` | Neurons per layer | [`neurons_analysis.png`](neurons_analysis.png) |
| `optimizer_analysis.py` | Optimizers | [`optimizer_analysis.png`](optimizer_analysis.png) |
| `random_state_analysis.py` | Random initialization/split | [`random_state_analysis.png`](random_state_analysis.png) |

Each analysis script is a separate experiment and can be invoked from the repository root after installing the dependencies, for example `python activation_analysis.py`.

## Repository layout

```text
main.py                         training and Optuna optimization
testing.py                      checkpoint evaluation
preparation_dataset.ipynb       data-preparation notebook
train.ipynb, test.ipynb         exploratory training / testing notebooks
*_analysis.py                   architecture / training sensitivity studies
*_experiment.py                 data imbalance / preprocessing studies
raw/                            original Excel inputs
train_*.xlsx                    prepared and experimental datasets
model.pt, scaler_v1_features.pkl saved artifacts
*.png                           published experiment figures
```

## Scope and limitations

- The label is constructed from `Type`, not an externally verified semantic description of all physical surfaces.
- Hyperparameter tuning and final evaluation reuse a split of `train_augmented.xlsx` in `main.py`; report independent-test results separately.
- A saved model is present, but the code does not define a general inference package or a deployed ROS node.
- No top-level licence file was included in the supplied archive.
