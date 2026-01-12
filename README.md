# Multi-Task Wasserstein–Regularized VAR(1) Estimation

Multi-task estimation of **VAR(1)** dynamics across many related time series (tasks/subjects) using **Wasserstein (optimal-transport) regularization** to encourage *structured similarity* between tasks while still allowing task-specific coefficients.

This repo is designed for ecological momentary assessment (EMA) / intensive longitudinal data, where each participant is one “task” and you want a separate VAR(1) coefficient matrix per participant, with coupling across participants via a geometry-aware penalty.

---

## Table of contents
- [What this implements](#what-this-implements)
- [Method overview](#method-overview)
- [Dataset used in examples](#dataset-used-in-examples)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Input / data format](#input--data-format)
- [API](#api)
- [Reproducibility](#reproducibility)
- [Project structure](#project-structure)
- [How to cite](#how-to-cite)
- [References](#references)

---

## What this implements

- **Per-task VAR(1)** estimation: for each task \(k\), estimate a matrix \(\Phi^{(k)} \in \mathbb{R}^{d \times d}\).
- **Multi-task coupling via Wasserstein regularization** (MTW): encourages coefficients to be “close” across tasks **according to a ground metric** you define over variables/features.
- **Column-wise and row-wise variants** (common in VAR contexts):
  - **Column-wise**: encourage similarity across tasks for each **predictor** column of \(\Phi\).
  - **Row-wise**: encourage similarity across tasks for each **response** row of \(\Phi\).
- **Cross-validation** over the Wasserstein penalty strength `alpha`.
- **Missing-data handling** via dropping invalid (x, y) pairs when forming the VAR regression design.

---

## Method overview

For each task \(k = 1,\dots,K\) with multivariate time series \(x^{(k)}_t \in \mathbb{R}^d\), a VAR(1) model is:

\[
x^{(k)}_t = \Phi^{(k)} x^{(k)}_{t-1} + \epsilon^{(k)}_t
\]

A standard regression formulation stacks responses and uses a Kronecker-style design; then we solve a multi-task optimization problem that adds a **Wasserstein / unbalanced optimal transport** penalty across tasks (Janati et al., 2019). Intuitively: if variable \(i\) is “near” variable \(j\) in your ground metric, the regularizer treats coefficients on \(i\) and \(j\) as more interchangeable across tasks than far-apart variables.

---

## Dataset used in examples

Examples in this repo are set up around the EMA study:

**Fried, Papanikolaou, & Epskamp — “Mental health and social contact during the COVID-19 pandemic: An ecological momentary assessment study.”**  
- 80 undergraduate students
- 14 days
- 4 prompts/day (fixed schedule)

The paper states that **data and materials are available on OSF**, and the openESM project also provides a curated access point (including a Zenodo DOI).

> ⚠️ Please follow the dataset’s license and citation requirements (see [How to cite](#how-to-cite)).

---

## Installation

### Option A: pip (recommended)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option B: editable install (if you package this repo)
```bash
pip install -e .
```

---

## Quickstart

### 1) Load and prepare a task’s time series
Assume each task has a matrix `Xk` of shape `(T, d)` (time × variables), with possible NaNs.

```python
import numpy as np
from fit_blockWiseFunctions import fit_WassColumnWise, cv_fitWassColumnWise
from groundmetric import create_ground_metric
from preprocessing import convert_to_regression_noNA  # or wherever you keep this

# Xk: (T, d) for one participant/task
Y_clean, Z_clean, M, valid_mask = convert_to_regression_noNA(Xk)
```

### 2) Build a ground metric over variables
You define the geometry. Common choices:
- **Chain / circular** adjacency (for ordered constructs)
- **Learned / empirical** distances (correlation-based, domain knowledge, etc.)
- **Graph distances** (if variables live on a network)

```python
d = Xk.shape[1]
ground_M = create_ground_metric(d, kind="euclidean")  # example placeholder
```

### 3) Fit MTW (column-wise)
```python
Phi_hat = fit_WassColumnWise(
    alpha=0.1,
    ground_M=ground_M,
    Xs=Xs,   # list/array of task designs
    Ys=Ys,   # list/array of task outcomes
    N=N,     # per-task time points or effective lengths (depending on your wrapper)
    n_tasks=len(Xs),
    d=d,
    Phi_init=None,
    verbose=True,
)
```

### 4) Cross-validate alpha
```python
best_alpha, cv_table = cv_fitWassColumnWise(
    wassPen_vals=[0.01, 0.05, 0.1, 0.25, 0.5],
    ground_M=ground_M,
    Xs=Xs,
    Ys=Ys,
    # ... other args ...
)
print("Best alpha:", best_alpha)
```

---

## Input / data format

You’ll typically organize data as:

- `Xs`: list of per-task predictor matrices (e.g., VAR design)
- `Ys`: list of per-task response vectors/matrices
- `ground_M`: `(d, d)` nonnegative matrix defining distances/costs between variables

If you start from raw time series \(X^{(k)}\) of shape `(T_k, d)`, your preprocessing step should:
1. Create `(x_{t-1}, x_t)` pairs (VAR lag-1)
2. Drop any pair where either side has NaNs (or handle missingness in a principled way)
3. Build the regression design (often Kronecker / block structure)

---

## API

Core entry points (names may vary depending on your code layout):

- `fit_WassColumnWise(...)` — Fit MTW with **column-wise** coupling
- `cv_fitWassColumnWise(...)` — CV over `alpha` for the column-wise variant
- `fit_WassRowWise(...)` — (if available) Fit MTW with **row-wise** coupling
- `create_ground_metric(d, ...)` — Build the ground metric over variables
- `convert_to_regression_noNA(X)` — Convert a `(T, d)` time series into regression form with NA handling

---

## Reproducibility

- Fix random seeds in Python/NumPy for CV splits and any randomized initialization.
- Record package versions (e.g., `pip freeze > requirements-lock.txt`).
- Save fitted matrices and CV tables with a timestamped run ID.

---

## Project structure

A typical layout is:

```
.
├── README.md
├── requirements.txt
├── data/                  # (optional) local copies; keep raw data out of git if large/restricted
├── notebooks/             # exploration / demos
├── src/                   # your package code
│   ├── preprocessing.py
│   ├── groundmetric.py
│   ├── fit_blockWiseFunctions.py
│   └── ...
└── scripts/
    ├── run_fit.py
    └── run_cv.py
```

---

## How to cite

### If you use MTW / Wasserstein regularization
Please cite the MTW method paper:

**Janati, Cuturi, & Gramfort (2019). Wasserstein regularization for sparse multi-task regression. AISTATS 2019.**

### If you use the EMA dataset from Fried et al.
Cite the original article:

**Fried, E. I., Papanikolaou, F., & Epskamp, S. (2022). Mental health and social contact during the COVID-19 pandemic: An ecological momentary assessment study. _Clinical Psychological Science, 10_(2), 340–354. https://doi.org/10.1177/21677026211017839**

Data availability pointers (as described in the paper / curated mirrors):
- OSF project (as reported in the paper): `https://osf.io/erp7v`
- Curated access via openESM (includes Zenodo DOI): `https://openesmdata.org/datasets/0001_fried/`

---

## References

- Janati, H., Cuturi, M., & Gramfort, A. (2019). *Wasserstein regularization for sparse multi-task regression.* Proceedings of AISTATS 2019. (PMLR 89)  
  Preprint: https://arxiv.org/abs/1805.07833

- Fried, E. I., Papanikolaou, F., & Epskamp, S. (2022). *Mental health and social contact during the COVID-19 pandemic: An ecological momentary assessment study.* *Clinical Psychological Science, 10*(2), 340–354. https://doi.org/10.1177/21677026211017839

- openESM dataset page for Fried (2021/2022): https://openesmdata.org/datasets/0001_fried/

---

## BibTeX

```bibtex
@inproceedings{janati2019wasserstein,
  title     = {Wasserstein Regularization for Sparse Multi-Task Regression},
  author    = {Janati, Hicham and Cuturi, Marco and Gramfort, Alexandre},
  booktitle = {Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year      = {2019}
}

@article{fried2022ema_covid,
  title   = {Mental Health and Social Contact During the COVID-19 Pandemic: An Ecological Momentary Assessment Study},
  author  = {Fried, Eiko I. and Papanikolaou, Faidra and Epskamp, Sacha},
  journal = {Clinical Psychological Science},
  year    = {2022},
  doi     = {10.1177/21677026211017839}
}
```
