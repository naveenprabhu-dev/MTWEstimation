# Multi-Task Wasserstein–Regularized VAR(1) Estimation

Multi-task estimation of **VAR(1)** dynamics across many related time series (tasks/subjects) using **Wasserstein regularization** to encourage similarity between tasks while still allowing task-specific coefficients.

This repo is designed for ecological momentary assessment (EMA) / intensive longitudinal data, where each participant is one “task” and you want a separate VAR(1) coefficient matrix per participant, with coupling across participants.

---

## Dataset

The main example (ema_script.py) is set up around the EMA study:

Fried, Papanikolaou, & Epskamp — “Mental health and social contact during the COVID-19 pandemic: An ecological momentary assessment study.” 
- 80 undergraduate students
- 14 days
- 4 prompts/day (fixed schedule)

---
## Executing the program
Modify the configurations at the top of ema_script.py. You can change the folds, test set, total time points, parallel jobs, minimum participants, and horizon for choosing the best alpha in CV. 

Run the script! It will save plots for each of the test samples, showing the performance for each of the ground metrics which are generated from chosen word embedding models, see (Kjell, 2023) below. 

## Reproducibility

- Fix random seeds in Python/NumPy for CV splits and any randomized initialization.
- Record package versions (e.g., `pip freeze > requirements-lock.txt`).
- Save fitted matrices and CV tables with a timestamped run ID.

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
