# Multi-Task Wasserstein–Regularized VAR(1) Estimation

Multi-task estimation of **VAR(1)** dynamics across related time series using **Wasserstein regularization** to encourage similarity between tasks while still allowing task-specific coefficients.

This repo is designed for ecological momentary assessment (EMA) / intensive longitudinal data, where each participant is one “task” and you want a separate VAR(1) coefficient matrix per participant, with coupling across participants.

---

## Dataset

The main example (ema_fried.py) is set up around the EMA study:

Fried, Papanikolaou, & Epskamp — “Mental health and social contact during the COVID-19 pandemic: An ecological momentary assessment study.” 
- 80 undergraduate students
- 14 days
- 4 prompts/day (fixed schedule)

---

## Reproducibility

1) Create the environment: from the project root, run 
    ```python
    conda env create -f environment.yml
    ```
2) Activate the environment: 
    ```python
    conda activate mtw-env
    ```
---

## Executing the program
If needed, modify the configurations at the top of ema_fried.py. You can change the folds, test set, total time points, parallel jobs, minimum participants, and horizon for choosing the best alpha in CV. 

From the project root, run 
  ```python
    python blockmtw/ema_fried.py
  ```

It will save a new plot for each of the test samples, with each plot containing info for all ground metrics (ground metrics generated from word embeddings from various encoder-only models, see get_ground_metrics.R and (Kjell, 2023) below.) Additional plots include task specific errors for each testing point. 

## Citations

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

@article{kjell2023textpackage,
  title   = {The Text-Package: An R-Package for Analyzing and Visualizing Human Language Using Natural Language Processing and Transformers},
  author  = {Kjell, Oscar and Giorgi, Salvatore and Schwartz, H. Andrew},
  journal = {Psychological Methods},
  year    = {2023},
  month   = {May},
  day     = {1},
  note    = {Advance online publication},
  doi     = {10.1037/met0000542}
}

```
