import numpy as np
import pandas
from numpy.random import default_rng

from mtw.mtw import MTW
from sklearn.metrics import mean_squared_error
import pandas as pd
import time

import matplotlib.pyplot as plt

def make_matrix_stable(Phi, max_radius=0.95): # Make matrix stable (abs(eigenvalues) < 1) for modeling time series data
    eigvals = np.linalg.eigvals(Phi)
    spectral_radius = max(abs(eigvals))
    if spectral_radius >= max_radius:
        stable_phi = (max_radius / spectral_radius) * Phi
        return stable_phi
    return Phi

def convert_to_regression(data_matrix, p=1): # Convert to regression as used in Fisher paper
    T, d = data_matrix.shape
    N = T - p

    Y = data_matrix[p:]

    rawY = Y
    print(f"rawY: {rawY}")

    Y = Y.T.reshape(-1, 1)

    X = np.hstack([data_matrix[p - i - 1:T - i - 1] for i in range(p)]) # shape is (N, d * p)

    Z = np.kron(np.eye(d), X)

    return rawY, Y, Z

def row_of(name):         # outcome variable (row)
    return name[2]

def col_of(name):         # predictor variable (column)
    return name[3]

def var_distance(a, b):
    # Distance between two of three variables: sad, happy, depressed
    if a == b:
        return 0
    # 0.5 if sad and depressed, otherwise 3 (depressed/happy or sad/happy)
    close_pairs = {("s","d"),("d","s")}
    return 0.5 if (a, b) in close_pairs else 3

def coeff_distance(a, b):
    # Distance between two pairs of predicted/predictors, uses var_distance
    y1, x1 = row_of(a), col_of(a)      # θ_y1x1 : y1 ← x1
    y2, x2 = row_of(b), col_of(b)

    same_outcome = (y1 == y2)
    auto1 = (y1 == x1)           # Checks if auto‑regressive
    auto2 = (y2 == x2)

    # Case 1: same outcome column
    if same_outcome:
        return var_distance(x1, x2)

    # Case 2: different outcomes, at least one cross‑variable
    if not auto1 or not auto2:
        return abs(var_distance(x1, y1) - var_distance(x2, y2)) # difference of differences

    # Case 3: different outcomes, both auto
    return var_distance(x1, x2)  # use rule 1 on the outcomes


def nearest_pd(A, eps=1e-8): # Finds nearest positive definite matrix to A (as covariance matrix requires PD)
    B = (A + A.T) / 2
    eigval, eigvec = np.linalg.eigh(B)
    eigval[eigval < eps] = eps          # lift negatives
    return (eigvec * eigval) @ eigvec.T

def simulate_var1(Phi, Cov, T=100): # Simulate the VAR(1) process to create synthetic data
    k = Phi.shape[0]
    print(k)
    y = np.zeros((T, k)) # Initial matrix
    y[0] = np.random.normal(0, 1, size=k) # Start with random values

    errors = np.random.multivariate_normal(mean=np.zeros(k), cov=Cov, size=T)

    for t in range(1, T):
        y[t] = Phi @ y[t - 1] + errors[t] # VAR(1) Update

    return y

def forecast_blocked_cv(Phi, y_start, horizon): # Predict horizon amount of points into the future, based on VAR(1) update
    predictions = []
    y_prev = y_start

    for _ in range(horizon):
        y_hat = Phi @ y_prev
        predictions.append(y_hat)
        y_prev = y_hat

    return np.stack(predictions, axis=0)




start_time = time.time()
rows, columns = 3, 3
rng = default_rng(42)
vars_ = ["s", "d", "h"]   # Sad, Depressed, Happy
coeffs = [f"θ_{y}{x}" for y in vars_ for x in vars_]   # ['θ_ss','θ_ds','θ_hs', 'θ_sd','θ_dd','θ_hd', 'θ_sh','θ_dh','θ_hh']
idx = {name: i for i, name in enumerate(coeffs)}




M = D = np.zeros((9, 9), dtype=float)
for i, a in enumerate(coeffs): # Creates matrix M
    for j, b in enumerate(coeffs):
        if i < j:  # fill upper‑triangular then copy
            D[i, j] = D[j, i] = coeff_distance(a, b)


means = np.array([       # base structure matrix (s, d, h)
    0.50, 0.50, -0.60,
   0.50,  0.50,  -0.60,
   -0.60,  -0.60,  0.50
])

beta = 0.7                      # decay rate for forming covariance matrix
Sigma = np.exp(-beta * M)        # element‑wise exp(−M)

Sigma += 1e-10 * np.eye(9)

Sigma = nearest_pd(Sigma, eps=1e-8) # Find nearest positive definite matrix

np.linalg.cholesky(Sigma) # Make sure the matrix is positive definite

# Create 5 VAR matrices with small random variation around a template
num_tasks = 5
true_matrices = []

for _ in range(num_tasks):
    vec = rng.multivariate_normal(mean=means, cov=Sigma)
    Phi = vec.reshape(3, 3)
    Phi = make_matrix_stable(Phi)
    true_matrices.append(Phi) # 'True' Coefficient matrix



# Generate time series data

time_series_data = []
Xs = []
Ys = []
rawYs = []
covariance_for_simulation = np.array([[1, 0.5, -0.6], [0.5, 1, -0.6], [-0.6, -0.6, 1]]) # Cov matrix w variable relationships for time series data simulation
for matrix in true_matrices:
    time_series_data.append(simulate_var1(matrix, covariance_for_simulation))

for data in time_series_data:
    rawY, Y, Z = convert_to_regression(data)
    rawYs.append(rawY)
    Ys.append(Y)
    Xs.append(Z)



print(f"Xs shape: {np.array(Xs).shape}")

d = 3 # 3 dimensions, sad, depressed, happy
rawYs = np.stack(rawYs, axis=0) # Contains Y values not refactored for kronecker regression
Xs_array = np.stack(Xs, axis=0)  # shape: (n_tasks, n_samples, n_features)
Ys_array = np.stack(Ys, axis=0).squeeze(-1)  # shape: (n_tasks, n_samples, 1)

print(f"rawYs_array: {rawYs}")
print(f"rawYs shape: {rawYs.shape}")

print(f"Xs_array: {Xs_array}")
print(f"Xs_array shape: {Xs_array.shape}")
print(f"Ys_array: {Ys_array}")
print(f"Ys_array shape: {Ys_array.shape}")

# Cross Validation and finding the best alpha/beta


alpha_list = [0.0, 1e-10, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 0.75, 1, 1.5, 2, 5, 10] # All alphas to test with CV
beta_list  = [0.0, 1e-10, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 0.75, 1, 1.5, 2] # All betas to test with CV



best_avg_mse = float('inf')
best_alpha = None
best_beta = None

fold_size = 20
n_folds = 5

results = []


for alpha in alpha_list:
    for beta in beta_list:

        all_fold_errors = []

        for k in range(n_folds):

            i0, i1 = k * fold_size, min((k + 1) * fold_size, rawYs.shape[1]) # Start and end point of fold

            # Blocked Split

            # X_val = Xs_array[:, i0:i1, :]
            # Y_val = Ys_array[:, i0:i1]

            X_train = np.concatenate([Xs_array[:, :i0, :], Xs_array[:, i1:, :]], axis=1) # Train on values not left out on this fold
            Y_train = np.concatenate([Ys_array[:, :i0], Ys_array[:, i1:]], axis=1) # Train for values not left out on this fold

            # print(f"X_train: {X_train}")
            # print(f"Y_train: {Y_train}")



            mtw_model = MTW(alpha=alpha, beta=beta, M=M)

            mtw_model.fit(X_train, Y_train)
            # print(f"True matrix: {var_matrices[0]}")
            estimated_matrices = [beta.reshape(d, d) for beta in mtw_model.coefs_.T]
            # print(f"Estimated matrix: {estimated_matrices[0]}")

            fold_errs = [] # Errors for each fold
            for task in range(num_tasks):
                true_matrix = true_matrices[task]
                estimated_matrix = estimated_matrices[task]

                # print(f"True Matrix: {true_matrix}")
                # print(f"Estimated Matrix: {estimated_matrix}")

                y0 = rawYs[task, i0-1]

                y_hat_pred = forecast_blocked_cv(estimated_matrix, y0, 1)

                Y_true = rawYs[task, i0:i0 + 1]

                mse = np.mean((Y_true - y_hat_pred)**2) # Compute difference between simulated and true for this task
                fold_errs.append(mse) # Append this task's (person's)
            avg_fold_mse = np.mean(fold_errs) # Average for this fold (over all tasks)
            all_fold_errors.append(avg_fold_mse)


        pair_errors = np.mean(all_fold_errors) # Error for overall (alpha, beta) pair - 5 folds, 5 tasks, 25 values averaged
        results.append({"alpha":alpha, "beta": beta, "mse": pair_errors})
        print(f"Alpha: {alpha}")
        print(f"Beta: {beta}")
        print(f"Average MSE: {pair_errors}")
        if pair_errors < best_avg_mse: # If the (alpha, beta) pair has a better mse than the current best pair
            best_avg_mse = pair_errors
            best_alpha = alpha
            best_beta = beta
print(f"Best Mean Squared Error: {best_avg_mse}")
print(f"Best Alpha: {best_alpha}")
print(f"Best Beta: {best_beta}")


# Plotting result
def nice_label(x):
    if abs(x) < 0.01 or abs(x) > 100:
        return f"{x:.0e}"
    return f"{x:.2f}".rstrip("0").rstrip(".")


df = pd.DataFrame(results)
mse_mat = (
    df.pivot(index="beta", columns="alpha", values="mse")
      .sort_index(ascending=True)          # β from small -> large
      .sort_index(axis=1, ascending=True)  # α from small -> large
)

beta_ticks  = mse_mat.index.to_list()      # y-values
alpha_ticks = mse_mat.columns.to_list()    # x-values


# plot
fig, ax = plt.subplots(figsize=(6, 5))

# origin="lower" flips the y-axis so SMALL β at bottom, LARGE β at top
im = ax.imshow(mse_mat.values, cmap="coolwarm",
               aspect="auto", origin="lower")

# x-axis (α)
ax.set_xticks(range(len(alpha_ticks)))
ax.set_xticklabels([nice_label(a) for a in alpha_ticks],
                   rotation=45, ha="right")

# y-axis (β)
ax.set_yticks(range(len(beta_ticks)))
ax.set_yticklabels([nice_label(b) for b in beta_ticks])

ax.set_xlabel(r"$\alpha$ value")   # swapped
ax.set_ylabel(r"$\beta$ value")
ax.set_title("Cross-validated MSE for each ($\\alpha$, $\\beta$) pair")

plt.colorbar(im, ax=ax, label="Mean MSE")
plt.tight_layout()
end_time = time.time()
print(f"Total time: {end_time - start_time}")
plt.show()
