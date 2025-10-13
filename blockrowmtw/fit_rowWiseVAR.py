### Code to fit Wasserstein-Penalized VAR, using updated Wasserstein distance

# Import functions 
from VAR_functions import gen_VARmat, simulateVAR1
from groundmetric import create_ground_metric
from fit_rowWiseFunctions import fit_WassRowWise, cv_fitWassRowWise
import matplotlib.pyplot as plt
import numpy as np 
import time

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
num_tasks_vec = [31, 101]
T_vec = [31, 101]
n_folds = 3
cov_beta = 0.95
wassP_list = [1e-5]
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
        fold_size = T_val // n_folds
        print("Fold size:", fold_size)

        # Print information 
        print("[INFO] ntasks = {}; T = {}".format(ntask_val, T_val))

        # Use only the desired time points for the specific amount of tasks
        data_subset_j = [ x[:T_val] for x in ts_data_all[:ntask_val] ]
        time_before_cv = time.time()
        print(f"Time before CV: {time_before_cv - start_time}")
        # Fit multitask VAR (j-th combination)
        result_j = cv_fitWassRowWise(ts_data_list = data_subset_j,
                                wassPen_vals = wassP_list,
                                ground_M = M, n_folds=n_folds, fold_size=fold_size, horizon=2)  # Supply true M matrix 

        # For each value of wassPen, compute average (k-th value of wassPen)
        mse_wassPen_list = []
        for wassPen_jk in wassP_list:

            # Loop through tasks 
            mse_tasks_list = []
            for params_jkl_index in range(ntask_val):   # l-th task 

                # Get true parameters 
                truePhi_jkl = task_params[params_jkl_index]["Phi"]

                # Get estimated parameters 
                estPhi_jk = [ x['est_Phi'] for x in result_j if x["wassPen"] == wassPen_jk ]
                [estPhi_jk_singleton] = estPhi_jk
                estPhi_jkl = estPhi_jk_singleton[params_jkl_index]

                # Compute MSE 
                mse_jkl = np.mean((truePhi_jkl - estPhi_jkl) ** 2)
                mse_tasks_list.append(mse_jkl)

            # Compute average across tasks 
            mse_wassPen_avg_tasks = np.mean(mse_tasks_list)

            # Append to wassPen list 
            mse_wassPen_list.append({'wassPen': wassPen_jk,
                                     'avg_mse_Phi': mse_wassPen_avg_tasks})


        # Collect for each combination 
        # Output should be:
        # - vector of MSE for each case
        mse_results.append({'ntasks': ntask_val,
                            'T': T_val,
                            'mse_wassPen': mse_wassPen_list})

print("[INFO] Finished simulation.")

# Plotting
plt.figure(figsize=(10, 6))

for entry in mse_results:
    ntasks = entry['ntasks']
    T = entry['T']
    wP_list = [d['wassPen'] for d in entry['mse_wassPen']]
    mse = [d['avg_mse_Phi'] for d in entry['mse_wassPen']]
    
    label = f"ntasks={ntasks}, T={T}"
    plt.plot(range(len(wP_list)), mse, marker='o', label=label)

    xtick_labels = [f"{w:.0e}" if w != 0 else "0" for w in wP_list]
    plt.xticks(ticks=range(len(wP_list)), labels=xtick_labels, rotation=45)

plt.xlabel('Wasserstein Penalty')
plt.ylabel('Average MSE of Phi')
plt.title('Avg MSE vs Wasserstein Penalty')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("NEWavg_mse_vs_wasspen.png", dpi=300, bbox_inches='tight')
plt.show()


# OPTIONAL FOR NOW
# Compute maximal improvement for each case 
# (Actually, this might noe be necessary)
# --- this is just min(MSE) - MSE[0]



