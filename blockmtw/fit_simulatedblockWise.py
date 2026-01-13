# Import functions 
from VAR_functions import gen_VARmat, simulateVAR1
from groundmetric import create_ground_metric
from fit_blockWiseFunctions import *
import numpy as np 
import time
ALPHA = 0.01
N_PASSES = 10
LAGS = 1

# Simulated data experiment. We compare the impact of alpha = 0.01 vs alpha = 0.0 for various tasks and time points, given that the synthetic data has some predefined relationships.
# Explores the effectiveness of columnWise Wasserstein regularization. 
def main(seed_val = 395):
    start_time = time.time()
    T_values_record = []
    mean_mse_wass_record = []   
    # Parameters for simulation 
    # Consider 4 cases:
        # a) num_tasks = 30, T = 30
        # b) num_tasks = 30, T = 100
        # c) num_tasks = 100, T = 30
        # d) num_tasks = 100, T = 100
    num_tasks_vec = [30, 100]
    T_vec = [30, 100] 
    cov_beta = 0.95

    # Specify ground metric (predefined relationships between variables)
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


    # Create a DataFrame to print the upper triangular M
    # mask_upper = np.triu(np.ones_like(M, dtype=bool))
    # M_upper = np.where(mask_upper, M, np.nan)

    # dfM_upper = pd.DataFrame(M_upper, index=coefnames, columns=coefnames)

    # pd.set_option("display.width", 200)
    # pd.set_option("display.max_rows", None)
    # pd.set_option("display.max_columns", None)
    # pd.set_option("display.float_format", lambda x: f"{x:7.3f}")
    # print(dfM_upper.to_string())


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
                                T = max(T_vec), 
                                use_seed = seed_val)
        ts_data_all.append(ts_data_k)


    # For each case, compute MSE (assuming we know the true values)
    for ntask_val in num_tasks_vec:
        for T_val in T_vec:
            # Print information 
            print("[INFO] ntasks = {}; T = {}".format(ntask_val, T_val))

            # Use only the desired time points for the specific amount of tasks
            data_subset_j = [ x[:T_val] for x in ts_data_all[:ntask_val] ]
            
            T, d = data_subset_j[0].shape # Shape of one task
            N = T - LAGS    # Number of regression points per task (one lost at beginning and end from predictors/outcomes)
            Ys = []
            Xs = []
            # Convert to regression format using previously defined function
            for data_k in data_subset_j: 
                Y_k, Z_k, N = convert_to_regression(data_k)
                Ys.append(Y_k)   # (d*N, 1)
                Xs.append(Z_k)   # (d*N, d*d)
        
            Xs_array = np.stack(Xs, axis=0)
            Ys_array = np.stack(Ys, axis=0).squeeze(-1)
            
            # Assert that the shapes are correct after converting to regression
            assert Xs_array.shape == (ntask_val, N*d, d*d), f"Xs_array {Xs_array.shape} != {(ntask_val, N*d, d*d)}"
            assert Ys_array.shape == (ntask_val, N*d),      f"Ys_array {Ys_array.shape} != {(ntask_val, N*d)}"
            
            # Create and fit MTW model with alpha = 0 for warm start
            mtw_model = MTW(alpha=0, beta=0.0, maxiter=50000, M=M)
            mtw_model.fit(Xs_array, Ys_array)
            coefs_warm_start = mtw_model.coefs_ # Warm start 
            if coefs_warm_start.shape == (d*d, ntask_val):
                B = coefs_warm_start.reshape(ntask_val, d, d) # Reshape into format that fits our fit_WassColumnWise
            elif coefs_warm_start.shape == (ntask_val, d*d):
                B = coefs_warm_start.T.reshape(ntask_val, d, d) # Reshape into format that fits our fit_WassColumnWise
            else:
                raise ValueError(f"Unexpected MTW coefs_ shape: {coefs_warm_start.shape}; expected d*d, n_tasks) or (n_tasks, d*d).")
        
            time_before_cv = time.time()
            print(f"Time before CV: {time_before_cv - start_time}") # Track time
            
            # Fit the columnWise approach with our chosen alpha (0.01)
            result_j = fit_WassColumnWise(alpha=ALPHA,
                                            ground_M=M,
                                            Xs=Xs_array, 
                                            Ys=Ys_array, N=N,
                                            n_tasks=ntask_val, 
                                            d=d,
                                        Phi_init = B, 
                                        n_passes=N_PASSES)  
            time_after_cv = time.time()
            print(f"Time after cv: {time_after_cv - start_time}")
            mse_wass = [] # MSE for alpha = 0.01 for all tasks
            mse_regression = []# MSE for alpha = 0 for all tasks
            print("B shape:", B.shape)
            for t in range(ntask_val): # Take the errors for each task and append them to new lists for computign MSE
                truePhi = task_params[t]["Phi"]
                estPhi_wass = result_j[t]
                estPhi_wass_0 = B.reshape(ntask_val, d*d)[t, :].reshape(d,d) # Reshape warm start in format we want
                mse_wass.append(np.mean((truePhi - estPhi_wass)**2))
                mse_regression.append(np.mean((truePhi - estPhi_wass_0)**2))
            
            # Convert to numpy arrays
            mse_regression = np.array(mse_regression)
            mse_wass = np.array(mse_wass)
            # Record Wasserstein performance only for plotting (seeing if there is benefit with increased time points)
            T_values_record.append(T_val)
            mean_mse_wass_record.append(mse_wass.mean())        
                
            print(f"Mean MSE (alpha = 0, no column wise fit): {mse_regression.mean()}")
            print(f"Median MSE (alpha = 0, no column wise fit): {np.median(mse_regression)}")
            print(f"Mean MSE (alpha = 0.01, column wise fit): {mse_wass.mean()}" )
            print(f"Median MSE (alpha = 0.01, column wise fit): {np.median(mse_wass)}")


    # This plot is only comparing the change in T on alpha = 0.01 columnWise fit result. 
    # Just as a sanity check that the method actually works (we expect MSE to decrease as T increases)
    plt.figure(figsize=(7,5))
    plt.plot(T_values_record, mean_mse_wass_record, marker='o', linewidth=2)

    plt.xlabel("Number of Time Points (T)")
    plt.ylabel("Mean Wasserstein MSE Across Tasks")
    plt.title("Mean Wasserstein-Regularized VAR Error vs Sample Size")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__": 
    main()