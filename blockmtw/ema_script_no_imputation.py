from mtw import MTW
import pandas as pd
import os
import numpy as np
from fit_blockWiseFunctions import fit_WassColumnWise, cv_fitWassColumnWise
from groundmetric import create_ground_metric
import matplotlib.pyplot as plt
import random
from statsmodels.tsa.stattools import adfuller, kpss

random.seed(395)  # reproducible selection
N_FOLDS = 5
TOTAL_POINTS = 56  # time points per participant
MIN_PEOPLE = 30  # minimum number of participants with enough data

alphas = [1e-10, 1e-5, 1e-2, 5e-2, 1e-1, 1.0, 2.0, 5.0, 25.0]
lags = 1
# Load data
base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, "..", "ofsstorage", "data", "clean_ema.csv")
data_path = os.path.abspath(data_path)

ema = pd.read_csv(data_path)

# Print number of rows for each unique ID
rows_per_id = ema.groupby("ID").size()

print("\nRows per participant (after cleaning):")
for pid, count in rows_per_id.items():
    print(f"ID {pid}: {count} rows")
rows_per_id.to_csv("rows_per_participant.csv", index=False)

# IDs that have exactly 56 rows
ids_with_56 = rows_per_id[rows_per_id == 56].index.tolist()
print("IDs with exactly 56 rows:", ids_with_56)

# Filter the dataset to only those participants
ema = ema[ema["ID"].isin(ids_with_56)].copy()
# Save filtered dataset
output_path = "ema_filtered_56rows.csv"
ema.to_csv(output_path, index=False)

print(f"Saved filtered dataset to {output_path}")
# print(ema["time"].value_counts().sort_index().to_string()) # Printing the counts of each timestamp

# Printing sorted timestamps per participant
# for pid, df in ema.groupby("ID"):
#     print(f"\n=== ID {pid} ===")

#     # sort times and keep the original index
#     df_sorted = df.sort_values("time")

#     # print index + time together
#     print(df_sorted[["time"]].to_string())

# Print summary after filtering
print("Remaining rows:", len(ema))
print("Remaining unique IDs:", ema['ID'].nunique())
    
qcols_all = [f"Q{i}" for i in range(1, 19)]
for q in qcols_all:
    ema[q] = pd.to_numeric(ema[q], errors="coerce")

# Sort chronologically within participant, used for regression
ema["time"] = pd.to_datetime(ema["time"])
ema = ema.sort_values(["ID", "time"]).reset_index(drop=True)

# --- 1) Drop the questions we don’t include ---
drop_vars = ["Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "Q13", "Q14", "Q16", "Q17"]
kept_vars = [q for q in qcols_all if q not in drop_vars]   # 18 - 10 = 8 vars
d = len(kept_vars)                                         # should be 8


# --- 2) Build per-participant (T, d) time-series matrices for all people who have 56 entries---
ts_data_all = []
per_pid_mats = {}  # also keep a dict if you want to track participant IDs

for pid, dfp in ema.groupby("ID"):
    mat = dfp[kept_vars].to_numpy(dtype=float)   # shape (T, d)
    ts_data_all.append(mat)
    per_pid_mats[pid] = mat

people_remaining = len(ts_data_all)

print(f"Participants kept who have exactly 56 data points: {people_remaining}, T={TOTAL_POINTS}, d={d}")


def convert_to_regression_noNA(data_matrix):
    """
    VAR(1) regression setup with NA handling.

    data_matrix: array of shape (T, d) with possible NaNs.

    Returns:
        Y_clean: (M, 1) – outcomes after dropping any invalid (x, y) pairs
        Z_clean: (M, d*d) – corresponding rows of the Kronecker design
        M: int – number of valid (x, y) pairs (effective sample size)
        valid_mask: (d*N,) bool – which rows of the original (Y,Z) were kept
    """
    p = 1
    T, d = data_matrix.shape
    N = T - p

    # Original VAR(1) setup
    Y_mat = data_matrix[p:, :]    # shape (N, d), y_t
    X_mat = data_matrix[:-p, :]   # shape (N, d), x_t

    # 1) validity of y: not NaN
    y_ok = ~np.isnan(Y_mat)                 # (N, d) bool

    # 2) validity of x row: all predictors present for that time t
    x_ok_row = ~np.isnan(X_mat).any(axis=1, keepdims=True)  # (N, 1) broadcast

    # 3) valid pair (t, j) iff y_ok AND x_ok_row
    valid = y_ok & x_ok_row                 # (N, d) bool

    # Flatten in the same order as your original code:
    # Y: (N, d) -> (d*N, 1) stacked by variable
    Y_vec = Y_mat.T.reshape(-1, 1)          # (d*N, 1)

    # Kronecker design: (d*N, d*d)
    Z_full = np.kron(np.eye(d), X_mat)

    # Flatten the validity mask in the same order
    # (N, d) -> (d*N,) where we stack by variable
    valid_vec = valid.T.reshape(-1)         # (d*N,)

    # Keep only valid (x, y) pairs
    Y_clean = Y_vec[valid_vec, :]           # (M, 1)
    Z_clean = Z_full[valid_vec, :]          # (M, d*d)

    M = Y_clean.shape[0]                    # effective sample size

    return Y_clean, Z_clean, M, valid_vec

# MIN_Y_SAMPLES = 440  # minimum number of (x, y) pairs after NA removal (440 is max, decrements in steps of 8) If not 440, make sure to modify the below functions accordingly.
# # --- 4) Convert to regression format for your functions ---
# Ys, Xs, Ms = [], [], []
# for data_k in ts_data_all:
#     Y_k, Z_k, M_i, valid_mask_i = convert_to_regression_noNA(data_k)  # VAR(1)
#     if M_i >= MIN_Y_SAMPLES:
#         Ys.append(Y_k)
#         Xs.append(Z_k)
#         Ms.append(M_i)
    

    

# Xs_array = np.stack(Xs, axis=0)                # (n_tasks, N*d, d*d)
# Ys_array = np.stack(Ys, axis=0).squeeze(-1)    # (n_tasks, N*d)

# print("Xs_array:", Xs_array.shape, "Ys_array:", Ys_array.shape)


horizon = 5
T = TOTAL_POINTS
N_full = T - lags
T_train = T - horizon
N_train = T_train - lags
max_M = d * N_full # maximum possible M (no missing data)

# 1) Select only tasks with full M_i = 440 and build training regression data
full_ts = []           # original time-series matrices (T, d) for the 9 tasks
Xs_train_list = []     # list of (M_train, d*d)
Ys_train_list = []     # list of (M_train, 1)
ts_data_train_list = [] # list of (T_train, d) training data matrices

for mat in ts_data_all: 
    # skip anything with missing values or wrong length
    if np.isnan(mat).any():
        continue
    if mat.shape[0] != T:
        continue

    # Check full M_i using the whole series
    Y_full, Z_full, M_full, _ = convert_to_regression_noNA(mat)
    if M_full != max_M:
        continue  # not a "perfect" participant

    # Training data: first T_train points
    data_train = mat[:T_train, :]  # shape (T_train, d) = (51, 8)
    Y_tr, Z_tr, M_tr, _ = convert_to_regression_noNA(data_train)

    if M_tr != d * N_train:
        raise ValueError(f"Unexpected M_tr={M_tr}, expected {d * N_train}")
    
    
    full_ts.append(mat)
    Xs_train_list.append(Z_tr)      # (M_tr, d*d)
    Ys_train_list.append(Y_tr)      # (M_tr, 1)
    ts_data_train_list.append(data_train)

n_tasks = len(full_ts)

if n_tasks == 0:
    raise ValueError("No tasks with full M_i = 440 found.")
print(f"Using {n_tasks} tasks with full data (M_i = {max_M}).")

# Stack into arrays for MTW
Xs_train = np.stack(Xs_train_list, axis=0)            # (n_tasks, M_train, d*d)
Ys_train = np.stack(Ys_train_list, axis=0).squeeze(-1)  # (n_tasks, M_train)
M_train = Ys_train.shape[1]                           # should be d * N_train = 8 * 50 = 400

print("Xs_train:", Xs_train.shape, "Ys_train:", Ys_train.shape)
print(f"N_train (per task) = {N_train}, M_train = {M_train}")

M_bert = np.array([
    [0.000000, 9.026052, 9.039160, 9.102301,10.789334,10.224275,10.472787,11.015110],
    [9.026052, 0.000000, 9.775446, 7.802097,11.427602,11.433619,11.179109,11.796992],
    [9.039160, 9.775446, 0.000000, 9.763803,10.216198, 9.933905,10.177053,10.420399],
    [9.102301, 7.802097, 9.763803, 0.000000,10.644347,10.862135,10.746799,11.183156],
    [10.789334,11.427602,10.216198,10.644347, 0.000000, 7.289855, 8.315838, 7.400572],
    [10.224275,11.433619, 9.933905,10.862135, 7.289855, 0.000000, 8.207881, 7.872855],
    [10.472787,11.179109,10.177053,10.746799, 8.315838, 8.207881, 0.000000, 6.269398],
    [11.015110,11.796992,10.420399,11.183156, 7.400572, 7.872855, 6.269398, 0.000000]
], dtype=float)

M_roberta = np.array([
    [0.000000, 4.782017, 5.466909, 5.934401, 6.417702, 6.158980, 6.551129, 6.570465],
    [4.782017, 0.000000, 4.991821, 5.131624, 6.837944, 6.916638, 6.215274, 6.403657],
    [5.466909, 4.991821, 0.000000, 5.759943, 7.015269, 6.674944, 6.494271, 6.509900],
    [5.934401, 5.131624, 5.759943, 0.000000, 6.710364, 6.847467, 6.755209, 6.427419],
    [6.417702, 6.837944, 7.015269, 6.710364, 0.000000, 5.511885, 6.180311, 6.329397],
    [6.158980, 6.916638, 6.674944, 6.847467, 5.511885, 0.000000, 6.003810, 5.922115],
    [6.551129, 6.215274, 6.494271, 6.755209, 6.180311, 6.003810, 0.000000, 5.053934],
    [6.570465, 6.403657, 6.509900, 6.427419, 6.329397, 5.922115, 5.053934, 0.000000]
], dtype=float)

M_xl = np.array([
    [0.00000, 17.53548, 12.92191, 19.70460, 23.47416, 15.86623, 17.78021, 21.51745],
    [17.53548, 0.00000, 16.49812, 15.11495, 20.24020, 20.45692, 17.25238, 21.01010],
    [12.92191, 16.49812, 0.00000, 18.86201, 23.24259, 18.51105, 18.47904, 22.26704],
    [19.70460, 15.11495, 18.86201, 0.00000, 16.02256, 19.99772, 15.68838, 19.97836],
    [23.47416, 20.24020, 23.24259, 16.02256, 0.00000, 20.79540, 16.98549, 18.59291],
    [15.86623, 20.45692, 18.51105, 19.99772, 20.79540, 0.00000, 15.50676, 18.95745],
    [17.78021, 17.25238, 18.47904, 15.68838, 16.98549, 15.50676, 0.00000, 15.48136],
    [21.51745, 21.01010, 22.26704, 19.97836, 18.59291, 18.95745, 15.48136, 0.00000]
])

metric_mats = {
    "bert": M_bert,
    "roberta": M_roberta,
    "xl": M_xl,
    
}

# Helper: convert MTW coefs_ to list of d x d matrices, one per task
def coefs_to_phi_list(coefs, d, n_tasks):
    if coefs.shape == (d * d, n_tasks):
        return [coefs[:, k].reshape(d, d) for k in range(n_tasks)]
    elif coefs.shape == (n_tasks, d * d):
        return [coefs[k, :].reshape(d, d) for k in range(n_tasks)]
    else:
        raise ValueError(f"Unexpected coefs_ shape: {coefs.shape}")
    
    
# 3) Storage for forecast errors:
# errors[h][metric_name]['alpha0' or 'alpha1e-2'] -> list of SSEs over tasks
errors = {
    h: {m_name: {"alpha0": [], f"bestalpha": []}
        for m_name in metric_mats.keys()}
    for h in range(1, horizon + 1)
}

# 4) Loop over all three M's
for m_name, M_mat in metric_mats.items():
    print(f"\n=== Metric: {m_name} ===")

    # Ground metric for column-wise Wasserstein
    ground_M, coefnames = create_ground_metric(M_mat)

    # 4a) Alpha = 0 run (MTW)
    mtw_model = MTW(alpha=0.0, beta=0.0, maxiter=50000, M=M_mat)
    mtw_model.fit(Xs_train, Ys_train)
    coefs_warm_start = mtw_model.coefs_

    Phi_alpha0_list = coefs_to_phi_list(coefs_warm_start, d, n_tasks)
    cv_result = cv_fitWassColumnWise(ts_data_list=ts_data_train_list, Xs_array=Xs_train, Ys_array=Ys_train, ground_M=M_mat, wassPen_vals=alphas, n_folds=N_FOLDS, horizon=horizon, Phi_init=coefs_warm_start)
    best_alpha = cv_result['best_alpha']
    print("**Best alpha:**", best_alpha)
    
    # 4b) Wasserstein-regularized run with alpha = 1e-2
    Phi_best_alpha_list = fit_WassColumnWise(
        alpha=best_alpha,
        ground_M=ground_M,
        Xs=Xs_train,
        Ys=Ys_train,
        N=N_train,
        n_tasks=n_tasks,
        d=d,
        Phi_init=coefs_warm_start,  # warm start from alpha=0
    )


    if len(Phi_best_alpha_list) != n_tasks:
        raise ValueError("fit_WassColumnWise did not return one Phi per task.")

    # 4c) Forecast on the 5-step test horizon for each task
    for task_idx, mat in enumerate(full_ts):
        # Starting point: last training observation y_50 (index T_train - 1)
        y0 = mat[T_train - 1, :].astype(float)   # shape (d,)

        Phi0 = Phi_alpha0_list[task_idx]
        Phi1 = Phi_best_alpha_list[task_idx]

        # True future values y_{51}, ..., y_{55}
        # mat indices: 51..55 == range(T_train, T_train + H)
        y_true_seq = mat[T_train:T_train + horizon, :]  # shape (H, d)

        for h in range(1, horizon + 1):
            # Forecast y_{50 + h} using each Phi via h-step recursion
            # y_hat_h = (Phi^h) * y_50
            # Alpha = 0
            Phi0_h = np.linalg.matrix_power(Phi0, h)
            y_hat0 = Phi0_h @ y0
            # Alpha = 1e-2
            Phi1_h = np.linalg.matrix_power(Phi1, h)
            y_hat1 = Phi1_h @ y0

            y_true_h = y_true_seq[h - 1, :]  # true y_{50+h}, shape (d,)

            # Sum of squared errors over variables (NOT sum variables first)
            err0 = np.sum((y_hat0 - y_true_h) ** 2)
            err1 = np.sum((y_hat1 - y_true_h) ** 2)

            errors[h][m_name]["alpha0"].append(err0)
            errors[h][m_name][f"alpha{best_alpha}"].append(err1)

# 5) Print numeric results to terminal
for h in range(1, horizon + 1):
    print(f"\n===== Horizon {h} step(s) =====")
    for m_name in metric_mats.keys():
        for alpha_label in ["alpha0", f"alpha{best_alpha}"]:
            arr = np.array(errors[h][m_name][alpha_label])
            if arr.size == 0:
                print(f"{m_name} | {alpha_label}: no data")
                continue
            print(
                f"{m_name} | {alpha_label}: "
                f"mean={arr.mean():.4f}, median={np.median(arr):.4f}, "
                f"std={arr.std():.4f}, n={arr.size}"
            )
            print(f"  values: {arr.tolist()}")

# 6) Make 5 figures, one per horizon, each with 6 boxplots (3 Ms × 2 alphas)
metric_order = ["bert", "roberta", "xl"]
alpha_order = ["alpha0", f"alpha{best_alpha}"]

for h in range(1, horizon + 1):
    fig, ax = plt.subplots(figsize=(10, 6))

    box_data = []
    labels = []

    for m_name in metric_order:
        for alpha_label in alpha_order:
            box_data.append(errors[h][m_name][alpha_label])
            labels.append(f"{m_name}\n{alpha_label}")

    ax.boxplot(box_data, showfliers=True)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Sum of squared errors (per task)")
    ax.set_title(f"Forecast error distribution – horizon {h} step(s)")
    plt.tight_layout()
    fig_name = f"forecast_boxplots_h{h}.png"
    plt.savefig(fig_name, dpi=300)
    plt.close(fig)
    print(f"Saved {fig_name}")