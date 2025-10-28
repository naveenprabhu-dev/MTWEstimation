 ### Code to fit Wasserstein-Penalized VAR, using updated Wasserstein distance

# Import functions 
from VAR_functions import gen_VARmat, simulateVAR1
from groundmetric import create_ground_metric
from fit_blockWiseFunctions import *
import numpy as np 
import time
import pandas as pd

# Seed
start_time = time.time()

seed_val = 12345

# Parameters for simulation 
# Consider 4 cases:
    # a) num_tasks = 30, T = 30
    # b) num_tasks = 30, T = 100
    # c) num_tasks = 100, T = 30
    # d) num_tasks = 100, T = 100
print("[DEBUGGING] trying small values for now ")
# Decreased these as compared to fit_WassVAR.py, to make sure it works for one iteration
num_tasks_vec = [30]
T_vec = [30] # One total point is lost in regression (one from predictors, one from outcomes). N = (T - lags)
n_folds = 5
cov_beta = 0.95
wassP_list = [1e-2]
horizon = 1
assert len(wassP_list) == len(set(wassP_list))

# Specify ground metric 
# Mimic the following: ['sad', 'depressed', 'happy', 'relaxed', 'stressed']
# ['sad', 'depressed'] are in the lower left quadrant
# remaining 3 each inhabit one of the remaining quadrants
# Use radians 
dist_in_degree = np.array([[0, 20, 180, 140, 30],
                           [0, 0, 160, 120, 50],
                           [0, 0, 0, 40, 150],
                           [0, 0, 0, 0, 180],
                           [0, 0, 0, 0, 0]], dtype = float)
num_vars, _ = dist_in_degree.shape
dist_in_degree[np.tril_indices(num_vars)] = np.nan
dist_in_rad = np.deg2rad(dist_in_degree)
M, coefnames = create_ground_metric(bDist_mat = dist_in_rad)

# assuming M, coefnames already exist
import numpy as np
import pandas as pd

mask_upper = np.triu(np.ones_like(M, dtype=bool))
M_upper = np.where(mask_upper, M, np.nan)

dfM_upper = pd.DataFrame(M_upper, index=coefnames, columns=coefnames)

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:7.3f}")


print(dfM_upper.to_string())


# Generate max(num_tasks) model 
task_params = gen_VARmat(num_tasks = max(num_tasks_vec),
                         num_vars = num_vars,
                         coef_cov_mat = np.exp(cov_beta * M),   # Convert to covariance 
                         use_seed = seed_val)


# Generate 1 TS from each model 
ts_data_all = []
for params_k in task_params:
    ts_data_k = simulateVAR1(Phi = params_k["Phi"],
                             Sigma = params_k["Psi"],
                             T = max(T_vec))
    ts_data_all.append(ts_data_k)


# For each case, compute MSE (assuming we know the true values)
# across different wassPens
mse_results = []
for ntask_val in num_tasks_vec:
    for T_val in T_vec:
        # Print information 
        print("[INFO] ntasks = {}; T = {}".format(ntask_val, T_val))

        # Use only the desired time points for the specific amount of tasks
        data_subset_j = [ x[:T_val] for x in ts_data_all[:ntask_val] ]
        time_before_cv = time.time()
        print(f"Time before CV: {time_before_cv - start_time}")
        # Find best alpha value using CV approach
        result_j = cv_fitWassColumnWise(ts_data_list = data_subset_j,
                                     task_params=task_params,
                                     wassPen_vals = wassP_list,
                                     ground_M = M, n_folds=n_folds, horizon=horizon)  # Supply true M matrix later
        time_after_cv = time.time()
        print(f"Time after cv: {time_after_cv - start_time}")
        compare_and_plot_alphas(result_j)

# OPTIONAL FOR NOW
# Compute maximal improvement for each case 
# (Actually, this might noe be necessary)
# --- this is just min(MSE) - MSE[0]



