import numpy as np
from numpy.random import default_rng

from mtw.mtw import MTW
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt


# Added for simpler usage 
import argparse 

# Parse arguments 
parser = argparse.ArgumentParser()
# parser.add_argument('--wasspen', type = float, required = True)
# parser.add_argument('--l1pen', type = float, required = True)
parser.add_argument('--numtasks', type = int, required = True)
parser.add_argument('--timelen', type = int, required = True)
args = parser.parse_args()



def simulate_var1(Phi, T=100, Sigma=np.array([       # base structure matrix (s, d, h)
   [1, 0.50, -0.60],
   [0.50,  1,  -0.60],
   [-0.60,  -0.60,  1]
])):
    k = Phi.shape[0]
    y = np.zeros((T, k)) # Initial matrix
    y[0] = np.random.normal(0, 1, size=k) # Start with random values

    errors = np.random.multivariate_normal(mean=np.zeros(k), cov=Sigma, size=T)

    for t in range(1, T):
        y[t] = Phi @ y[t - 1] + errors[t] # VAR(1) Update

    return y

def make_matrix_stable(Phi, max_radius=0.95):
    eigvals = np.linalg.eigvals(Phi)
    spectral_radius = max(abs(eigvals))
    if spectral_radius >= max_radius:
        Phi = (max_radius / spectral_radius) * Phi
    return Phi

def convert_to_regression(data_matrix, p=1):
    T, d = data_matrix.shape
    N = T - p

    # print(f"Y: {data_matrix[0:5]}")

    Y = data_matrix[p:]

    Y = Y.T.reshape(-1, 1)

    # print(f"Collapsed Y: {Y[0:15]}")

    X = np.hstack([data_matrix[p - i - 1:T - i - 1] for i in range(p)]) # shape is (N, d * p)
    # print(f"X: {X[0:5]}")
    Z = np.kron(np.eye(d), X)

    return Y, Z

rows, columns = 3, 3
rng = default_rng(42)
# Next time sample from normal distribution and covariance matrix, with distances similar for each one from
vars_ = ["s", "d", "h"]   # Sad, Depressed, Happy
coeffs = [f"θ_{y}{x}" for y in vars_ for x in vars_]   # row‑major
# ['θ_ss','θ_ds','θ_hs', 'θ_sd','θ_dd','θ_hd', 'θ_sh','θ_dh','θ_hh']
idx = {name: i for i, name in enumerate(coeffs)}




def row_of(name):         # outcome variable (row)
    return name[2]

def col_of(name):         # predictor variable (column)
    return name[3]

def var_distance(a, b):
    """Base distance between variables S,D,H."""
    if a == b:
        return 0
    # 0.5 if sad and depressed, otherwise 3 (depressed/happy or sad/happy)
    close_pairs = {("s","d"),("d","s")}
    return 0.5 if (a, b) in close_pairs else 3
    # return 5 if (a, b) in close_pairs else 15

def coeff_distance(a, b):
    """
    Distance between two coefficients θ_yx and θ_vu
    """
    # print(f"a: {a}")
    # print(f"b: {b}")
    y1, x1 = row_of(a), col_of(a)      # θ_y1x1 : y1 ← x1
    y2, x2 = row_of(b), col_of(b)

    same_outcome = (y1 == y2)
    auto1 = (y1 == x1)           # auto‑regressive?
    auto2 = (y2 == x2)

    # Case 1: same outcome column
    if same_outcome:
        return var_distance(x1, x2)

    # Case 2: different outcomes, at least one cross‑variable
    if not auto1 or not auto2:
        return abs(var_distance(x1, y1) - var_distance(x2, y2)) # difference of differences

    # Case 3: different outcomes, both auto
    return var_distance(x1, x2)  # use rule 1 on the outcomes


def nearest_pd(A, eps=1e-8): # Finds nearest positive definite matrix to A
    B = (A + A.T) / 2
    eigval, eigvec = np.linalg.eigh(B)
    eigval[eigval < eps] = eps          # lift negatives
    return (eigvec * eigval) @ eigvec.T

M = D = np.zeros((9, 9), dtype=float)

# print(coeffs)
for i, a in enumerate(coeffs):
    for j, b in enumerate(coeffs):
        if i < j:  # fill upper‑triangular then copy
            D[i, j] = D[j, i] = coeff_distance(a, b)


means = np.array([       # base structure matrix (s, d, h)
    0.50, 0.50, -0.60,
   0.50,  0.50,  -0.60,
   -0.60,  -0.60,  0.50
])



beta = 0.9                      # decay rate for forming covariance matrix; tweak if Σ too flat/peaked
Sigma = np.exp(-beta * M)        # element‑wise exp(−M)

Sigma += 1e-10 * np.eye(9)

Sigma = nearest_pd(Sigma, eps=1e-8) # Find nearest positive definite matrix

np.linalg.cholesky(Sigma)

# print("Debugging")
# print(np.round(Sigma, 2))

# Create 5 VAR matrices with small random variation around a template
num_tasks = args.numtasks
var_matrices = []

for _ in range(num_tasks):
    vec = rng.multivariate_normal(mean=means, cov=Sigma)
    Phi = vec.reshape(3, 3)
    Phi = make_matrix_stable(Phi)
    var_matrices.append(Phi)



# Generate time series data

time_series_data = []
Xs = []
Ys = []
for matrix in var_matrices:
    time_series_data.append(simulate_var1(matrix,
                                          T = args.timelen))

for data in time_series_data:
    Y, Z = convert_to_regression(data)
    Ys.append(Y)
    Xs.append(Z)

# M = np.eye(9)

# print("DEBUG M")
# print(M)

d = 3
# mtw_model = MTW(alpha=args.wasspen,
#                 beta=args.l1pen,
#                 max_iter=4000, M=M)

Xs_array = np.stack(Xs, axis=0)  # shape: (n_tasks, n_samples, n_features)
Ys_array = np.stack(Ys, axis=0).squeeze(-1)  # shape: (n_tasks, n_samples, 1)




# Try fit with different values of alpha 
alpha_list = [0.0, 1e-10, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 0.75, 1, 1.5, 2, 5, 10] # All alphas to test with CV
mse_list = [None] * len(alpha_list)
for alpha in alpha_list:

    # Define model 
    mtw_model = MTW(alpha=alpha,
                    beta=0,
                    max_iter=5000, M=M)

    # Fit MTW model 
    mtw_model.fit(Xs_array, Ys_array)

    # Extract matrices 
    estimated_matrices = [beta.reshape(d, d) for beta in mtw_model.coefs_.T]


    # print("\n=== Matrix Comparison: True vs Estimated (MSE) ===")

    mean_squared_errors = []
    for i in range(num_tasks):
        true_matrix = var_matrices[i]
        estimated_matrix = estimated_matrices[i]

        # Flatten both to compare element-wise
        mse = mean_squared_error(true_matrix.flatten(), estimated_matrix.flatten())
        mean_squared_errors.append(mse)

        # print(f"\nTask {i + 1}:")
        # print("True Matrix:\n", true_matrix)
        # print("Estimated Matrix:\n", estimated_matrix)
        # print(f"MSE for task: {i} {mse:.4f}")

    average_mse = sum(mean_squared_errors) / len(mean_squared_errors)
    mse_list[alpha_list.index(alpha)] = average_mse
    # print(f"Average Mean Squared Error: {average_mse}")
    # print("Parameter Settings: WassPen = {}, l1Pen = {}, T = {}".format(args.wasspen,
    #                                                                     args.l1pen,
    #                                                                     args.timelen))



# Plot the sequence
plt.plot(mse_list, marker='o')
plt.title("MSE Plot")
plt.xlabel("Alpha Index")
plt.ylabel("MSE")
plt.grid(True)

# Save the plot
plt.savefig("mse_wasspen_plot.png", dpi=300)

# Optional: Show the plot
plt.show()

print("Parameter Settings: numtasks = {}, T = {}".format(args.numtasks,
                                                         args.timelen))
