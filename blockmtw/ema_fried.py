from mtw import MTW
import pandas as pd
import os
import numpy as np
from fit_blockWiseFunctions import cv_fitWassColumnWise
from groundmetric import create_ground_metric
import matplotlib.pyplot as plt
import random

# =============================================================================
# Config
# =============================================================================
random.seed(395)  # reproducible selection
N_FOLDS = 5
TOTAL_POINTS = 56  # time points per participant
MIN_PEOPLE = 30  # minimum number of participants with enough data
N_JOBS = 6 # multiprocessing for CV
LAGS = 1 # We only support VAR(1) for now.
TEST_SAMPLES = 5 # This is the set for the final forecasting after choosing the best alpha. 
HORIZON = 1 # CV horizon (how many points are predicted ahead during CV)


alphas = [0.0, 1e-10, 1e-5, 1e-4, 5e-4, 1e-2, 5e-2, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
# alphas = [1.0]

# =============================================================================
# DATA PREPROCESSING
# =============================================================================
# Load data
base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, "..", "ofsstorage", "data", "clean_ema.csv")
data_path = os.path.abspath(data_path)
ema = pd.read_csv(data_path)

# Summarize number of rows per participant
rows_per_id = ema.groupby("ID").size()
print("\nRows per participant (after cleaning):")
for pid, count in rows_per_id.items():
    print(f"ID {pid}: {count} rows")
rows_per_id.to_csv("rows_per_participant.csv", index=False)

# # Keep only participants with exactly 56 observations (expected for this dataset)
ids_with_56 = rows_per_id[rows_per_id == TOTAL_POINTS].index.tolist()
print("People who have exactly 56 rows:", ids_with_56)
ema = ema[ema["ID"].isin(ids_with_56)].copy()

# Save filtered dataset
output_path = "ema_filtered_56rows.csv"
ema.to_csv(output_path, index=False)
print(f"Saved filtered dataset to {output_path}")

# Print summary after filtering
print("Remaining rows:", len(ema))
print("Remaining unique IDs:", ema['ID'].nunique())

# Convert Likert scores to numeric, note any parsing errors as NAN. 
qcols_all = [f"Q{i}" for i in range(1, 19)]
for q in qcols_all:
    ema[q] = pd.to_numeric(ema[q], errors="coerce")

# Sort chronologically within participant
ema["time"] = pd.to_datetime(ema["time"])
ema = ema.sort_values(["ID", "time"]).reset_index(drop=True)

# Drop the questions we don’t include
drop_vars = ["Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "Q13", "Q14", "Q16", "Q17"]
kept_vars = [q for q in qcols_all if q not in drop_vars]   # 18 - 10 = 8 vars
d = len(kept_vars)                                         # should be 8


def convert_to_regression_noNA(data_matrix):
    """
    VAR(1) regression setup with NA handling (ignores all (x,y) pairs with any NaN).

    data_matrix: array of shape (T, d) with possible NaNs.

    Returns:
        Y_clean: (n_eff, 1) - outcomes after dropping any invalid (x, y) pairs
        Z_clean: (n_eff, d*d) - corresponding rows of the Kronecker design
        n_eff: int - number of valid (x, y) pairs (effective sample size)
        valid_mask: (d*N,) bool - which rows of the original (Y,Z) were kept
    """
    p = 1
    T, d = data_matrix.shape
    N = T - p

    # Build VAR(1) predictor/outcome matrices
    Y_mat = data_matrix[p:, :]    # shape (N, d), y_t
    X_mat = data_matrix[:-p, :]   # shape (N, d), x_t

    # Mark outcomes that exist
    y_ok = ~np.isnan(Y_mat)                 # (N, d) bool

    # Mark timepoints where *all* predictors exist (row-wise)
    x_ok_row = ~np.isnan(X_mat).any(axis=1, keepdims=True)  # (N, 1) broadcast

    # Valid pairs are those with valid y and a fully-observed predictor row
    valid = y_ok & x_ok_row                 # (N, d) bool

    # Vectorize Y by stacking variables (same ordering as original)
    Y_vec = Y_mat.T.reshape(-1, 1)          # (d*N, 1)

    # Kronecker design: block-diagonal over variables
    Z_full = np.kron(np.eye(d), X_mat)

    # Vectorize mask in matching order
    valid_vec = valid.T.reshape(-1)         # (d*N,)

    # Keep only valid (x, y) pairs
    Y_clean = Y_vec[valid_vec, :]           # (n_eff, 1)
    Z_clean = Z_full[valid_vec, :]          # (n_eff, d*d)

    n_eff = Y_clean.shape[0]                    # effective sample size

    return Y_clean, Z_clean, n_eff, valid_vec



# =============================================================================
# DATA INTERPOLATION
# =============================================================================
ts_data_all = []
per_pid_mats = {}
min_n_eff_ratio = 0.875   # require at least 87.5% well-formed samples BEFORE interpolation, otherwise remove this participant

for pid, dfp in ema.groupby("ID"):
    # Extract only relevant variables
    df_vars = dfp[kept_vars].copy()

    # Remove participants that don't have enough valid samples
    mat_raw = df_vars.to_numpy(dtype=float)
    Y_raw, Z_raw, n_eff_raw, _ = convert_to_regression_noNA(mat_raw)

    if n_eff_raw < TOTAL_POINTS * d * min_n_eff_ratio:
        print(f"Skipping ID {pid}: only {n_eff_raw} valid samples before interpolation (<{TOTAL_POINTS * d * min_n_eff_ratio})")
        continue

    # Interpolate and forward/back fill any edge gaps
    df_interp = (
        df_vars
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )

    mat = df_interp.to_numpy(dtype=float)

    # Sanity check: no missing values remain
    if np.isnan(mat).any():
        raise ValueError(f"Interpolation failed for ID {pid}")

    # Store for MTW training
    ts_data_all.append(mat)
    per_pid_mats[pid] = mat

print(f"Participants kept after filtering + interpolation: {len(ts_data_all)}")
people_remaining = len(ts_data_all)

# Collect raw (pre-interpolation) matrices for everyone
raw_ts_data = {}
for pid, dfp in ema.groupby("ID"):
    raw_ts_data[pid] = dfp[kept_vars].to_numpy(dtype=float)

# Print full set of IDs vs. interpolated IDs
all_ids = list(raw_ts_data.keys())
interp_ids = list(per_pid_mats.keys())  # from your code above

print("\n=== ALL PARTICIPANTS (raw) ===")
for i, pid in enumerate(all_ids):
    print(f"Index {i}: ID {pid}")

print("\n=== PARTICIPANTS AFTER INTERPOLATION (kept) ===")
for i, pid in enumerate(interp_ids):
    print(f"Index {i}: ID {pid}")

print(f"\nTotal raw: {len(all_ids)}, After interpolation: {len(interp_ids)}")


# =============================================================================
# OPTIONAL: Uncomment to plot raw vs interpolated for each kept participant
# =============================================================================
# for pid in interp_ids:
#     raw = raw_ts_data[pid]
#     interp = per_pid_mats[pid]

#     fig, axes = plt.subplots(d, 2, figsize=(12, 2.5*d), sharex=True)
#     fig.suptitle(f"Participant {pid}: Raw vs Interpolated", fontsize=14)

#     for j in range(d):
#         axes[j, 0].plot(raw[:, j], marker="o")
#         axes[j, 0].set_ylabel(kept_vars[j])
#         axes[j, 0].set_title("Raw")

#         axes[j, 1].plot(interp[:, j], marker="o")
#         axes[j, 1].set_title("Interpolated")

#     plt.tight_layout()
#     plt.show()

# -----------------------------------------------------------------------------
# OPTIONAL: Uncomment to plot excluded participants (raw only)
# -----------------------------------------------------------------------------
# excluded_ids = [pid for pid in all_ids if pid not in interp_ids]

# print("\n=== PARTICIPANTS EXCLUDED BEFORE INTERPOLATION ===")
# for i, pid in enumerate(excluded_ids):
#     print(f"Excluded Index {i}: ID {pid}")

# for pid in excluded_ids:
#     raw = raw_ts_data[pid]

#     fig, axes = plt.subplots(d, 1, figsize=(10, 2.5*d), sharex=True)
#     fig.suptitle(f"Excluded Participant {pid} (Raw Data Only)", fontsize=14)

#     for j in range(d):
#         axes[j].plot(raw[:, j], marker="o")
#         axes[j].set_ylabel(kept_vars[j])

#     plt.tight_layout()
#     plt.show()


# =============================================================================
# MTW FITTING + FORECAST EVALUATION
# =============================================================================

T = TOTAL_POINTS
N_full = T - LAGS
T_train = T - TEST_SAMPLES
N_train = T_train - LAGS

# Build per-task regression arrays for MTW
full_ts = []
Xs_train_list = []
Ys_train_list = []
ts_data_train_list = []

for mat in ts_data_all:
    # Sanity check for number of time points
    if mat.shape[0] != T:
        raise ValueError("Participant has wrong number of rows")

    # Full regression size: d * (T - 1)
    Y_full, Z_full, M_full, _ = convert_to_regression_noNA(mat)
    expected_full = d * (T - 1)      # 8 * 55 = 440
    if M_full != expected_full:
        raise ValueError(f"Unexpected M_full={M_full}, expected {expected_full}")

    # Training regression size: d * (T_train - 1)
    data_train = mat[:T_train, :]
    Y_tr, Z_tr, eff_tr, _ = convert_to_regression_noNA(data_train)
    expected_train = d * N_train     # 8 * 50 = 400
    if eff_tr != expected_train:
        raise ValueError(f"Unexpected eff_tr={eff_tr}, expected {expected_train}")

    full_ts.append(mat)
    Xs_train_list.append(Z_tr)
    Ys_train_list.append(Y_tr)
    ts_data_train_list.append(data_train)

n_tasks = len(full_ts)
if n_tasks == 0:
    raise ValueError(f"No tasks with {min_n_eff_ratio} well formed points found.")
print(f"[INFO] Number of tasks selected for MTW: {n_tasks}")

# Stack into numpy arrays for MTW
Xs_train = np.stack(Xs_train_list, axis=0)            # (n_tasks, M_train, d*d)
Ys_train = np.stack(Ys_train_list, axis=0).squeeze(-1)  # (n_tasks, M_train)
M_train = Ys_train.shape[1]                           # should be d * N_train = 8 * 50 = 400

print("Xs_train:", Xs_train.shape, "Ys_train:", Ys_train.shape)
print(f"N_train (per task) = {N_train}, M_train = {M_train}")

# Ground metrics from get_ground_metrics.R
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
    
# Track per-horizon forecast deltas: (best_alpha MSE) - (alpha=0 MSE)
errors = {
    h: {m_name: {"alpha0": [], f"bestalpha": []}
        for m_name in metric_mats.keys()}
    for h in range(1, TEST_SAMPLES + 1)
}

best_alphas_per_metric = {}
# Loop over all three M's
for m_name, M_mat in metric_mats.items():
    print("\n" + "=" * 70)
    print(f"\033[1m=== STARTING METRIC: {m_name.upper()} ===\033[0m")
    print("=" * 70)
    
    
    print(f"\033[94m[{m_name}] Building ground metric...\033[0m")

    # Build ground metric
    ground_M, coefnames = create_ground_metric(M_mat)

    # Baseline alpha = 0
    print(f"\033[94m[{m_name}] Fitting alpha=0 (baseline MTW)...\033[0m")
    mtw_model = MTW(alpha=0.0, beta=0.0, maxiter=50000, M=ground_M)
    mtw_model.fit(Xs_train, Ys_train)
    coefs_warm_start = mtw_model.coefs_
    Phi_alpha0_list = coefs_to_phi_list(coefs_warm_start, d, n_tasks)

    # Reshape warm start for fit_WassColumnWise format
    if coefs_warm_start.shape == (d*d, n_tasks):
        coefs_warm_start = coefs_warm_start.reshape(n_tasks, d, d) 
    elif coefs_warm_start.shape == (n_tasks, d*d):
        coefs_warm_start = coefs_warm_start.T.reshape(n_tasks, d, d) 
    else:
        raise ValueError(f"Unexpected MTW coefs_ shape: {coefs_warm_start.shape}; expected d*d, n_tasks) or (n_tasks, d*d).")
    print(f"\033[92m[{m_name}] Alpha=0 fit complete.\033[0m")

    # Run cross validation, find best alpha
    print(f"\033[94m[{m_name}] Starting cross-validation to find best alpha...\033[0m")
    cv_result = cv_fitWassColumnWise(ts_data_list=ts_data_train_list, Xs_array=Xs_train, Ys_array=Ys_train, ground_M=ground_M, wassPen_vals=alphas, n_folds=N_FOLDS, horizon=HORIZON, Phi_init=coefs_warm_start, n_jobs=N_JOBS)
    best_alpha = cv_result['best_alpha']
    best_alphas_per_metric[m_name] = best_alpha
    print(f"\033[92m[{m_name}] Best alpha selected: {best_alpha}\033[0m")
    Phi_best_alpha_list = cv_result['best_alpha_fit']
    if len(Phi_best_alpha_list) != n_tasks:
        raise ValueError("fit_WassColumnWise did not return one Phi per task.")

    # Forecast evaluation: compare baseline vs best-alpha across horizons
    print(f"\033[94m[{m_name}] Starting forecast evaluation for horizons 1-{TEST_SAMPLES}...\033[0m")
    for task_idx, mat in enumerate(full_ts):
        # y0: last training observation (index T_train - 1)
        y0 = mat[T_train - 1, :].astype(float)

        Phi0 = Phi_alpha0_list[task_idx]
        Phi_best = Phi_best_alpha_list[task_idx]


        # mat indices: 51..55
        y_true_seq = mat[T_train:T_train + TEST_SAMPLES, :]  # shape (H, d)

        for h in range(1, TEST_SAMPLES + 1):
            # h-step VAR recursion uses Phi^h @ y0
            Phi0_h = np.linalg.matrix_power(Phi0, h)
            y_hat0 = Phi0_h @ y0
            Phi_best_h = np.linalg.matrix_power(Phi_best, h)
            y_hat1 = Phi_best_h @ y0

            y_true_h = y_true_seq[h - 1, :]  

            # Mean squared errors over variables at horizon h
            err0 = np.mean((y_hat0 - y_true_h) ** 2)
            err1 = np.mean((y_hat1 - y_true_h) ** 2)

            # Store difference: improvement (<0 means better)
            if "diff" not in errors[h][m_name]:
                errors[h][m_name]["diff"] = []
            errors[h][m_name]["diff"].append(err1 - err0)
    print(f"\033[92m[{m_name}] Forecast evaluation complete.\033[0m")
    print(f"\033[1m=== FINISHED METRIC: {m_name.upper()} ===\033[0m")
    print("=" * 70)


# Print numeric results for differences (best_alpha - alpha0)
for h in range(1, TEST_SAMPLES + 1):
    print(f"\n===== Horizon {h} step(s) =====")
    for m_name in metric_mats.keys():
        diff_arr = np.array(errors[h][m_name]["diff"])
        if diff_arr.size == 0:
            print(f"{m_name} | diff: no data")
            continue
        print(
            f"{m_name} | diff (best={best_alphas_per_metric[m_name]} - alpha0): "
            f"mean={diff_arr.mean():.4f}, median={np.median(diff_arr):.4f}, "
            f"std={diff_arr.std():.4f}, n={diff_arr.size}"
        )
        print(f"  values: {diff_arr.tolist()}")


metric_order = ["bert", "roberta", "xl"]

# =============================================================================
# PLOTTING
# =============================================================================
# Boxplots per horizon
for h in range(1, TEST_SAMPLES + 1):
    fig, ax = plt.subplots(figsize=(10, 6))

    box_data = []
    labels = []

    for m_name in metric_order:
        box_data.append(errors[h][m_name]["diff"])
        labels.append(f"{m_name}\n(best={best_alphas_per_metric[m_name]} vs 0)")

    ax.boxplot(box_data, showfliers=True)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Difference in MSE (best α − α=0)")
    ax.axhline(0, color="black", linewidth=1)  # reference line at no improvement

    ax.set_title(f"Forecast error difference distribution - horizon {h}")
    plt.tight_layout()

    fig_name = f"forecast_diff_boxplots_h{h}.png"
    plt.savefig(fig_name, dpi=300)
    plt.close(fig)
    print(f"Saved {fig_name}")
    

# Per-task bar plots: for each horizon, show diffs by task across metrics

for h in range(1, TEST_SAMPLES + 1):
    fig, ax = plt.subplots(figsize=(14, 6))

    # x positions for tasks: 0, 1, ..., n_tasks-1
    task_indices = np.arange(n_tasks)
    bar_width = 0.25  # width for each metric bar

    # For each metric, plot a bar slightly shifted on the x-axis
    for m_idx, m_name in enumerate(metric_order):
        diffs = np.array(errors[h][m_name]["diff"])

        if diffs.size != n_tasks:
            raise ValueError(
                f"Horizon {h}, metric {m_name}: expected {n_tasks} diffs, got {diffs.size}"
            )

        # Shift bars so they sit side-by-side for each task
        offset = (m_idx - 1) * bar_width  # -0.25, 0, +0.25 for 3 metrics
        ax.bar(task_indices + offset,
               diffs,
               width=bar_width,
               label=f"{m_name} (best={best_alphas_per_metric[m_name]})")

    # Horizontal reference line at 0 (improvement if diff < 0)
    ax.axhline(0.0, color="black", linewidth=1)

    ax.set_xlabel("Task")
    ax.set_ylabel("Difference in MSE (best α − α=0)")
    ax.set_title(f"Per-task MSE difference by ground metric – horizon {h}")
    ax.set_xticks(task_indices)
    ax.set_xticklabels(task_indices) 
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig_name = f"per_task_mse_diff_h{h}.png"
    plt.savefig(fig_name, dpi=300)
    plt.close(fig)
    print(f"Saved {fig_name}")
