# Functions to fit MTW regression 

# Import libraries 
from mtw.mtw import MTW
import numpy as np
import warnings 
import matplotlib.pyplot as plt

N_PASSES = 1

def convert_to_regression(data_matrix):
        # VAR(1)
        p = 1
        T, d = data_matrix.shape
        N = T - p

        # Outcomes: Y in shape (d*N, 1), stacked by variable (var0's N first, then var1's N second, etc)
        Y = data_matrix[p:]            # (N, d)
        Y = Y.T.reshape(-1, 1)         # (d*N, 1)

        # Predictors X (N, d)
        X = data_matrix[0:T-1]

        # Kronecker design Z = kron(I_d, X) -> (d*N, d*d)
        Z = np.kron(np.eye(d), X)
        
        print("N: ", N)

        return Y, Z, N
    
### Things to add ###
# Set a number of time points to predict into the future as an argument #
# Make this function callable such that we can like put it into some kind of package and they can just enter the information#


def flat_to_time_view(y_flat, d):
    """(N*d, ) to (N, d) view (ordered by time). Used to predict one time step ahead in the future."""
    y = np.asarray(y_flat).reshape(-1) # Array should already be in this form 
    N = y_flat.size // d
    return y.reshape(d, N).T

def cv_fitWassRowWise(ts_data_list, task_params, ground_M, wassPen_vals, n_folds, horizon, lags=1): 
    # Sanity checks
    n_tasks = len(ts_data_list)
    print("Tasks:", n_tasks)
    T, d = ts_data_list[0].shape
    print("Shape of one task (time points, variables):", ts_data_list[0].shape)
    fold_size = int(np.ceil((T - lags) / n_folds)) # N // n_folds, rounded up to get the exact number of folds specified
    print("[FOLD SIZE]:", fold_size)

    # Trial/limits (copied from your style)
    # warnings.warn("TRIAL: We do not use L1 penalty for now.")
    # warnings.warn("LIMITATION: the number of samples cannot differ per task for now.")

    # --- Convert each task to regression format (Kronecker design) ---
    
    # Build per-task (Y, Z)
    Ys = []
    Xs = []
    for data_k in ts_data_list:
        Y_k, Z_k, N = convert_to_regression(data_k)
        Ys.append(Y_k)   # (d*N, 1)
        Xs.append(Z_k)   # (d*N, d*d)
        
    Xs_array = np.stack(Xs, axis=0)
    Ys_array = np.stack(Ys, axis=0).squeeze(-1)
    
    assert Xs_array.shape == (n_tasks, N*d, d*d), f"Xs_array {Xs_array.shape} != {(n_tasks, N*d, d*d)}"
    assert Ys_array.shape == (n_tasks, N*d),      f"Ys_array {Ys_array.shape} != {(n_tasks, N*d)}"
    if n_folds * fold_size > N: 
        warnings.warn(f"n_folds*fold_size = {n_folds*fold_size} exceeds N={N}; last fold may be smaller.")
    
    def build_validation_idx(i0, i1, N, d): 
        """Indices to get the proper validation time window [i0, i1] for all variables"""
        return np.concatenate([np.arange(v*N + i0, v*N + (i1 + 1)) for v in range(d)], axis=0)
    
    def build_kill_idx(i1, N, d, lags=1): 
        """Kill indices to drop from training to avoid data leakage in the validation set.
           Default is VAR(1): if VAR(p), drop p lags after i1 (i1+1, ..., i1+lags) per variable block. 
        """
        start = i1 + 1
        stop = min(i1 + 1 + lags, (N - 1)) # N - 1 to index the last element
        if start >= (N - 1): 
            return np.array([], dtype=int) # Validation fold contains the final time points, no risk of leakage
        drop_times = np.arange(start, stop, dtype=int)
        return np.concatenate([v * N + drop_times for v in range(d)]).astype(int)

    # print("Example shapes before slicing:")
    # print("Z[0] shape:", Xs[0].shape, " (should be (d*N, d*d))")
    # print("Y[0] shape:", Ys[0].shape, " (should be (d*N, 1))")
    cv_table = []
    best_avg_mse = float('inf')
    best_alpha = None
    for alpha in wassPen_vals: 
        fold_mses = []
        for k in range(n_folds):
            i0, i1 = k * fold_size, min((k+1) * fold_size - 1, N - 1)
            validation_idx = build_validation_idx(i0, i1, N, d) # Indices to keep out
            kill_idx = build_kill_idx(i1, N=N, d=d, lags=1) # Points to kill (one after end of validation) to ensure no data leakage
            # print("Validation idx:", validation_idx)
            # print("Kill idx:", kill_idx)
            n_flat = N * d 
            keep_mask = np.ones(n_flat, dtype=bool)
            keep_mask[validation_idx] = False
            keep_mask[kill_idx] = False
            keep_idx = np.where(keep_mask)[0] # Get first array of the tuple
            # print("I0: ", i0)
            # print("I1:", i1)
            # print("Keep idx:", keep_idx)
            # print("X_train before removing indexes:", Xs_array.shape)
            # print("Y_train before removing indexes:", Ys_array.shape)
            X_train = Xs_array[:, keep_idx, :] # Train on values not left out on this fold
            Y_train = Ys_array[:, keep_idx] # Train for values not left out on this fold
            N_train = keep_idx.size // d # Reduced time length due to leaving out the fold
            # print("X_train after removing indexes shape:", X_train.shape)
            # print("Y_train after removing indexes shape:", Y_train.shape)
            # print("N_train:", N_train)
        
            out = fit_WassRowWise(alpha=alpha, ground_M=ground_M, Xs=X_train, Ys=Y_train, N=N_train, n_tasks=n_tasks, d=d)

            est_Phis = out[0]['est_Phi'] # Extract all estimated coefficient matrices
            
            def forecast_mse(horizon, Ys_array, est_Phis, ts_data_list):
                # VAR (1) forecasting
                # Outputs validation fold error for defined time length in future (horizon)
                if horizon < 1: 
                    raise ValueError(f"horizon must be >=1, got {horizon}")
                
                task_mses = []
                per_task_details = []
                for t_idx in range(n_tasks): 
                    Phi_hat = est_Phis[t_idx]
                    Y_time_full = flat_to_time_view(Ys_array[t_idx], d)
                    if i0 == 0: 
                        curr = ts_data_list[t_idx][0]
                    else: 
                        curr = Y_time_full[i0 - 1]
                    end_idx = i0 + horizon
                    if end_idx > i1: 
                        raise ValueError(
                            f"Requested horizon {horizon} from start_idx {i0}"
                            f"exceeds endpoint of validation fold {i1}"
                        )
                        
                    y_preds = np.empty((horizon, d), dtype=float)
                    for s in range(horizon): 
                        curr = (Phi_hat @ curr)
                        y_preds[s, :] = curr
                    y_true = Y_time_full[i0:end_idx, :]
                    mse = float(np.mean((y_true - y_preds) **2))
                    task_mses.append(mse)
                    
                    per_task_details.append({"y_true": y_true, "y_pred": y_preds})
                    
                return task_mses, per_task_details
        
            task_mses, _ = forecast_mse(horizon=horizon, Ys_array=Ys_array, est_Phis=est_Phis, ts_data_list=ts_data_list)
            fold_mses.append(float(np.mean(task_mses)))
            
        avg_mse = float(np.mean(fold_mses)) if fold_mses else float('inf')
        cv_table.append({"alpha": alpha, "fold_mses": fold_mses, "avg_mse": avg_mse})

        if avg_mse < best_avg_mse:
            best_avg_mse = avg_mse
            best_alpha = alpha  
              
    # ---- Fit given the best alpha from CV ----
    best_alpha_fit = fit_WassRowWise(
        alpha=best_alpha,
        ground_M=ground_M,
        Xs=Xs_array,
        Ys=Ys_array,
        N=N,                 # full length
        n_tasks=n_tasks,
        d=d,
        max_iter=50000
    )
    
    zero_alpha_fit = fit_WassRowWise(
        alpha=0,
        ground_M=ground_M,
        Xs=Xs_array,
        Ys=Ys_array,
        N=N,                 # full length
        n_tasks=n_tasks,
        d=d,
        max_iter=50000
    )
    # best_alpha_fit is a list like:
    #   [{'wassPen': a, 'est_Phi': [Phi_task0, Phi_task1, ...]}, ...]
    
    result = {
        "cv_table": cv_table,
        "best_alpha": best_alpha,
        "best_avg_mse": best_avg_mse,
        "best_alpha_fit": best_alpha_fit, 
        "zero_alpha_fit": zero_alpha_fit, 
        "task_params": task_params, 
        "n_tasks": n_tasks
    }
    
    # print("Result:", result)

    return result
                
def cv_fitWassColumnWise(ts_data_list, task_params, ground_M, wassPen_vals, n_folds, horizon, lags=1): 
    # Sanity checks
    n_tasks = len(ts_data_list)
    print("Tasks:", n_tasks)
    T, d = ts_data_list[0].shape
    print("Shape of one task (time points, variables):", ts_data_list[0].shape)
    fold_size = int(np.ceil((T - lags) / n_folds)) # N // n_folds, rounded up to get the exact number of folds specified
    print("[FOLD SIZE]:", fold_size)

    # Trial/limits (copied from your style)
    # warnings.warn("TRIAL: We do not use L1 penalty for now.")
    # warnings.warn("LIMITATION: the number of samples cannot differ per task for now.")

    # --- Convert each task to regression format (Kronecker design) ---
    
    # Build per-task (Y, Z)
    Ys = []
    Xs = []
    for data_k in ts_data_list:
        Y_k, Z_k, N = convert_to_regression(data_k)
        Ys.append(Y_k)   # (d*N, 1)
        Xs.append(Z_k)   # (d*N, d*d)
        
    Xs_array = np.stack(Xs, axis=0)
    Ys_array = np.stack(Ys, axis=0).squeeze(-1)
    
    assert Xs_array.shape == (n_tasks, N*d, d*d), f"Xs_array {Xs_array.shape} != {(n_tasks, N*d, d*d)}"
    assert Ys_array.shape == (n_tasks, N*d),      f"Ys_array {Ys_array.shape} != {(n_tasks, N*d)}"
    if n_folds * fold_size > N: 
        warnings.warn(f"n_folds*fold_size = {n_folds*fold_size} exceeds N={N}; last fold may be smaller.")
    
    def build_validation_idx(i0, i1, N, d): 
        """Indices to get the proper validation time window [i0, i1] for all variables"""
        return np.concatenate([np.arange(v*N + i0, v*N + (i1 + 1)) for v in range(d)], axis=0)
    
    def build_kill_idx(i1, N, d, lags=1): 
        """Kill indices to drop from training to avoid data leakage in the validation set.
           Default is VAR(1): if VAR(p), drop p lags after i1 (i1+1, ..., i1+lags) per variable block. 
        """
        start = i1 + 1
        stop = min(i1 + 1 + lags, (N - 1)) # N - 1 to index the last element
        if start >= (N - 1): 
            return np.array([], dtype=int) # Validation fold contains the final time points, no risk of leakage
        drop_times = np.arange(start, stop, dtype=int)
        return np.concatenate([v * N + drop_times for v in range(d)]).astype(int)

    # print("Example shapes before slicing:")
    # print("Z[0] shape:", Xs[0].shape, " (should be (d*N, d*d))")
    # print("Y[0] shape:", Ys[0].shape, " (should be (d*N, 1))")
    cv_table = []
    best_avg_mse = float('inf')
    best_alpha = None
    for alpha in wassPen_vals: 
        fold_mses = []
        for k in range(n_folds):
            print(f"Fold {k + 1} out of {n_folds}")
            i0, i1 = k * fold_size, min((k+1) * fold_size - 1, N - 1)
            validation_idx = build_validation_idx(i0, i1, N, d) # Indices to keep out
            kill_idx = build_kill_idx(i1, N=N, d=d, lags=1) # Points to kill (one after end of validation) to ensure no data leakage
            # print("Validation idx:", validation_idx)
            # print("Kill idx:", kill_idx)
            n_flat = N * d 
            keep_mask = np.ones(n_flat, dtype=bool)
            keep_mask[validation_idx] = False
            keep_mask[kill_idx] = False
            keep_idx = np.where(keep_mask)[0] # Get first array of the tuple
            # print("I0: ", i0)
            # print("I1:", i1)
            # print("Keep idx:", keep_idx)
            # print("X_train before removing indexes:", Xs_array.shape)
            # print("Y_train before removing indexes:", Ys_array.shape)
            X_train = Xs_array[:, keep_idx, :] # Train on values not left out on this fold
            Y_train = Ys_array[:, keep_idx] # Train for values not left out on this fold
            N_train = keep_idx.size // d # Reduced time length due to leaving out the fold
            # print("X_train after removing indexes shape:", X_train.shape)
            # print("Y_train after removing indexes shape:", Y_train.shape)
            # print("N_train:", N_train)
        
            out = fit_WassColumnWise(alpha=alpha, ground_M=ground_M, Xs=X_train, Ys=Y_train, N=N_train, n_tasks=n_tasks, d=d)

            est_Phis = out[0]['est_Phi'] # Extract all estimated coefficient matrices
            
            def forecast_mse(horizon, Ys_array, est_Phis, ts_data_list):
                # VAR (1) forecasting
                # Outputs validation fold error for defined time length in future (horizon)
                if horizon < 1: 
                    raise ValueError(f"horizon must be >=1, got {horizon}")
                
                task_mses = []
                per_task_details = []
                for t_idx in range(n_tasks): 
                    Phi_hat = est_Phis[t_idx]
                    Y_time_full = flat_to_time_view(Ys_array[t_idx], d)
                    if i0 == 0: 
                        curr = ts_data_list[t_idx][0]
                    else: 
                        curr = Y_time_full[i0 - 1]
                    end_idx = i0 + horizon
                    if end_idx > i1: 
                        raise ValueError(
                            f"Requested horizon {horizon} from start_idx {i0}"
                            f"exceeds endpoint of validation fold {i1}"
                        )
                        
                    y_preds = np.empty((horizon, d), dtype=float)
                    for s in range(horizon): 
                        curr = (Phi_hat @ curr)
                        y_preds[s, :] = curr
                    y_true = Y_time_full[i0:end_idx, :]
                    mse = float(np.mean((y_true - y_preds) **2))
                    task_mses.append(mse)
                    
                    per_task_details.append({"y_true": y_true, "y_pred": y_preds})
                    
                return task_mses, per_task_details
        
            task_mses, _ = forecast_mse(horizon=horizon, Ys_array=Ys_array, est_Phis=est_Phis, ts_data_list=ts_data_list)
            fold_mses.append(float(np.mean(task_mses)))
            
        avg_mse = float(np.mean(fold_mses))
        cv_table.append({"alpha": alpha, "fold_mses": fold_mses, "avg_mse": avg_mse})

        if avg_mse < best_avg_mse:
            best_avg_mse = avg_mse
            best_alpha = alpha  
            
    print(f"Performing alpha fit for best alpha: {best_alpha}")
              
    # ---- Fit given the best alpha from CV ----
    best_alpha_fit = fit_WassColumnWise(
        alpha=best_alpha,
        ground_M=ground_M,
        Xs=Xs_array,
        Ys=Ys_array,
        N=N,                 # full length
        n_tasks=n_tasks,
        d=d,
        max_iter=50000
    )
    
    print("Performing fit for alpha = 0")
    
    zero_alpha_fit = fit_WassColumnWise(
        alpha=0,
        ground_M=ground_M,
        Xs=Xs_array,
        Ys=Ys_array,
        N=N,                 # full length
        n_tasks=n_tasks,
        d=d,
        max_iter=50000
    )
    # best_alpha_fit is a list like:
    #   [{'wassPen': a, 'est_Phi': [Phi_task0, Phi_task1, ...]}, ...]
    
    result = {
        "cv_table": cv_table,
        "best_alpha": best_alpha,
        "best_avg_mse": best_avg_mse,
        "best_alpha_fit": best_alpha_fit, 
        "zero_alpha_fit": zero_alpha_fit, 
        "task_params": task_params, 
        "n_tasks": n_tasks
    }
    
    # print("Result:", result)

    return result


# Row-wise MTW for VAR(1), predicting each variable separately
def fit_WassRowWise(alpha,
                    ground_M, Xs, Ys, N, n_tasks, d, max_iter=50000): # Build one for columns as well separately
    """
    Fits multitask VAR(1) row-by-row with MTW.
    Keeps your Kronecker Z = kron(I_d, X) construction, but for each row j:
      - selects the Y and Z rows corresponding to target j,
      - selects the Z columns corresponding to row-j coefficients (length d),
      - stacks across tasks, fits MTW, and assembles Phi per task.

    Returns
    -------
    out_results : dictionary for current alpha being tested:
      {
        'wassPen': alpha,
        'est_Phi': [Phi_task0 (d,d), Phi_task1 (d,d), ...]
      }
    """
    print(f"Length of Ys", len(Ys))
    print(f"Ys shape:", Ys[0].shape)
    print(f"Length of Xs", len(Xs))
    print(f"Xs shape:", Xs[0].shape)
    # Prepare output container (same structure as fit_WassVAR1)
    results_map = {alpha: [np.zeros((d, d), dtype=float) for _ in range(n_tasks)]}
    
    print("N_train:", N)
    

    # --- Row-wise loop ---
    # For target row j, select:
    #   Y_j (N,)   := Ys[k][j*N:(j+1)*N, 0]
    #   X_j (N,d)  := Xs[k][j*N:(j+1)*N, j*d:(j+1)*d]
    for j in range(d):
        # Stack across tasks into MTW format: (n_tasks, N, d), (n_tasks, N)
        Xs_row = []
        Ys_row = []
        row_slice = slice(j * N, (j + 1) * N)
        col_slice = slice(j * d, (j + 1) * d)

        for k in range(n_tasks):
            X_k = Xs[k]
            Y_k = Ys[k]

            X_row_k = X_k[row_slice, col_slice]            # (N, d)
            y_row_k = Y_k[row_slice]                       # (N,)

            # Defensive checks
            print("X_row_k.shape:", X_row_k.shape)
            print("Y_row_k.shape:", y_row_k.shape)
            assert X_row_k.shape == (N, d)
            assert y_row_k.shape == (N,)

            Xs_row.append(X_row_k)
            Ys_row.append(y_row_k)

        Xs_array = np.stack(Xs_row, axis=0)  # Prev shape: (n_tasks, N, d), New shape: (n_tasks, N)
        Ys_array = np.stack(Ys_row, axis=0)  # Prev shape: (n_tasks, N)
        print("X_s array shape:", Xs_array.shape)
        print("Y_s array shape:", Ys_array.shape)

        # Fit MTW for given alpha and fill row j of Phi across tasks
            # Creates aligned dxd block from ground metric M (diagonal blocks)
        M_block = ground_M[j * d:(j + 1) * d, j * d:(j + 1) * d]
        print("Block M shape:", M_block.shape)
        mtw_model = MTW(alpha=alpha, beta=0.0, max_iter=max_iter, M=M_block)
        mtw_model.fit(Xs_array, Ys_array)

        coefs = mtw_model.coefs_
        # Expect either (d, n_tasks) or (n_tasks, d)
        if coefs.shape == (d, n_tasks):
            B = coefs  # predictors x tasks
        elif coefs.shape == (n_tasks, d):
            B = coefs.T
        else:
            raise ValueError(f"Unexpected MTW coefs_ shape: {coefs.shape}; "
                                f"expected (d, n_tasks) or (n_tasks, d).")

        # Assign learned coefficients into row j for each task
        for k in range(n_tasks):
            results_map[alpha][k][j, :] = B[:, k]

    # Convert to list of results in the same format as fit_WassVAR1
    out_results = []
    out_results.append({
        'wassPen': alpha,
        'est_Phi': results_map[alpha]  # list length n_tasks, each (d,d)
    })

    return out_results

def fit_WassColumnWise(alpha, ground_M, Xs, Ys, N, n_tasks, d, Phi_init=None, n_passes=N_PASSES, max_iter=50000, verbose=True):
    """ 
    Column wise MTW fitting for multitask VAR(1)
    Encourages columns of Phi to be similar via Wasserstein penalty. 
    n_passes chooses the amount of passes before assuming convergence (as the columns are not independent like the rows are)
    Phi_init: list of (d,d) arrays or None 
        Optional warm start for Phi per task. If none, Phi initialized to zeros.
    """
    Phi_est = [np.zeros((d,d), dtype=float) for _ in range(n_tasks)]
    
    def phi_to_vec(Phi):
        return Phi.reshape(-1, order='F')  # (d*d,)

    if verbose:
        print("=== Column-wise MTW ===")
        print(f"Xs[0] shape: {Xs[0].shape}, Ys[0] shape: {Ys[0].shape}, d={d}, N={N}, tasks={n_tasks}")
        
    for iter in range(n_passes):
        if verbose:
            print(f"\n--- Pass {iter+1}/{n_passes} ---")
        
        for c in range(d): # For each column c of Phi
            col_idx = np.array([c + j*d for j in range(d)]) # Indices in the global coefficient 25-vector corresponding to column c: Positions [c, c+d, c+2d, ... c+ (d-1)*d]
  

            print("[COL IDX]:", col_idx)
            # print("Ground m indices:", np.ix_(col_idx, col_idx))
            
            
            M_sub = ground_M[c::d, c::d] # Grabs the corresponding d x d submatrix in M for this column (not contiguous)
            # print("Ground M shape:", ground_M.shape)
            # print("COL 1:", M_sub)
            # print("COL 2:", ground_M[c+1::d, c+1::d])
            # print("COL 3:", ground_M[c+2::d, c+2::d])
            # print("Col 4:", ground_M[c+3::d, c+3::d])
            # print("Col 5:", ground_M[c+4::d, c+4::d])
            
            if verbose: 
                print(f"Column {c}: selecting {len(col_idx)} coefficients; M_sub shape {M_sub.shape}")
            
            Xs_col, Ys_res = [], []
            
            for k in range(n_tasks): # Loop over all tasks and get the correct column
                X_k = Xs[k]
                y_k = Ys[k]
                # print("X_s shape:", Xs.shape)
                # print("Y_s shape:", Ys.shape)
                # print("X_k shape:", X_k.shape)
                # print("y_k shape:", y_k.shape)
                
                # Full prediction with all columns
                # phi_vec_k = phi_to_vec(Phi_est[k]) # Creates 25-vector of Phi coefficients in row order
                # print("[THETA K]", theta_k)
                # full_pred = X_k @ phi_vec_k # Basically our estimated y
                # # print("X_k shape", X_k.shape)
                # print("Phi vec shape", phi_vec_k.shape)
                # print("[FULL PRED]", full_pred)
                
                # Contribution of current column c
                # phi_col_k = Phi_est[k][:, c]
                # # print("THETA_C_K", theta_c_k)
                # contrib_c = X_k[:, c*d:(c+1)*d] @ phi_col_k
                # print(f"[CONTRIBUTION OF COLUMN {c}]", contrib_c)
                
                # Residual that removes all columns except c
                # rk = y_k - (full_pred - contrib_c)
                # print("[RESIDUAL]", rk)
                Xc_k = X_k[c*N:(c+1)*N, c*d:(c+1)*d]
                yc_k = y_k[c*N:(c+1)*N]
                # print(f"[COLUMN {c} OF X]", Xc_k)
                
                # if Xc_k.shape != (N*d, d) or rk.shape != (N*d,):
                #     raise ValueError("Shape mismatch in column-wise preparation.")
                
                Xs_col.append(Xc_k)
                Ys_res.append(yc_k)
                
            Xs_array = np.stack(Xs_col, axis=0)
            Ys_array = np.stack(Ys_res, axis=0)
            
            # print("Xs_array length", len(Xs_array))
            # print("Shape of Xs:", Xs_array[0].shape)
            # print("Ys_array length", len(Ys_array))
            # print("Shape of Ys:", Ys_array[0].shape)
            
            # if verbose: 
            #     print("MTW fit shapes -> X:", Xs_array.shape, "Y:", Ys_array.shape)
                
            mtw_model = MTW(alpha=alpha, beta=0.0, max_iter=max_iter, M=M_sub)
            mtw_model.fit(Xs_array, Ys_array)

            coefs = mtw_model.coefs_
            # Expect either (d, n_tasks) or (n_tasks, d)
            if coefs.shape == (d, n_tasks):
                B = coefs  # predictors x tasks
            elif coefs.shape == (n_tasks, d):
                B = coefs.T
            else:
                raise ValueError(f"Unexpected MTW coefs_ shape: {coefs.shape}; expected (d, n_tasks) or (n_tasks, d).")

            
            # Update the column c across tasks
            for k in range(n_tasks):
                Phi_est[k][:, c] = B[:, k]
            #     print(f"[PHI ESTIMATION for column {c}]", Phi_est[k][:, c])
            # print("[PHI FOR TASK K]", Phi_est[k])
                
    out_results = [{
        'wassPen': alpha,
        'est_Phi': Phi_est  # list of (d,d) per task
    }]
    
    return out_results

# Compares best alpha and alpha = 0 with the true Phis
# Takes in the result of the cv_fitWassRowWise function
def compare_and_plot_alphas(result):
    # print(f"Result: {result}")
    best_alpha = float(result["best_alpha"])
    best_alpha_str = f"{best_alpha:.3g}"
    best_alpha_est_Phi = result["best_alpha_fit"][0]["est_Phi"]
    zero_alpha_est_Phi = result["zero_alpha_fit"][0]["est_Phi"]
    task_params = result["task_params"]
    n_tasks = result["n_tasks"]
    
    mse_best = []
    mse_zero = []
    # Compare best alpha with true Phis
    # Compare zero alpha with true Phis
    for t in range(n_tasks):
        truePhi = task_params[t]["Phi"]
        est_best = best_alpha_est_Phi[t]
        est_zero = zero_alpha_est_Phi[t]
        print(f"truePhi: {truePhi}")
        print(f"Estimated: {est_best}")
        # Frobenius MSE across matrix entries
        mse_best.append(np.mean((truePhi - est_best)**2))
        mse_zero.append(np.mean((truePhi - est_zero)**2))
        
    mse_best = np.array(mse_best)
    mse_zero = np.array(mse_zero)
    # --- summary stats ---
    print(f"Mean MSE  (best α: {best_alpha}): {mse_best.mean():.6g}")
    print(f"Mean MSE  (α=0)   : {mse_zero.mean():.6g}")
    print(f"Median MSE(best α: {best_alpha}): {np.median(mse_best):.6g}")
    print(f"Median MSE(α=0)   : {np.median(mse_zero):.6g}")

    # --- Figure 1: grouped bars per task ---
    task_idx = np.arange(n_tasks)
    width = 0.4

    fig1, ax1 = plt.subplots(figsize=(10, 4 + 0.05*n_tasks))
    ax1.bar(task_idx - width/2, mse_best, width, label="best α")
    ax1.bar(task_idx + width/2, mse_zero, width, label="α = 0")
    ax1.set_xlabel("Task")
    ax1.set_ylabel("MSE vs true Φ")
    ax1.set_title("Per-task MSE comparison: best α vs α=0")
    ax1.set_xticks(task_idx)
    ax1.set_xticklabels([str(i) for i in task_idx])
    ax1.legend()
    ax1.grid(axis="y", linestyle=":", alpha=0.4)

    # --- annotate best alpha on fig 1 ---
    ax1.text(
        0.99, 0.98, f"best α = {best_alpha_str}",
        transform=ax1.transAxes, ha="right", va="top",
        fontsize=10, bbox=dict(facecolor="white", edgecolor="black", alpha=0.7, pad=6)
    )

    fig1.tight_layout()

    # --- Figure 2: paired scatter vs y=x ---
    fig2, ax2 = plt.subplots(figsize=(5.5, 5.5))
    ax2.scatter(mse_zero, mse_best, s=30)
    lims = [min(mse_zero.min(), mse_best.min()), max(mse_zero.max(), mse_best.max())]
    span = lims[1] - lims[0]
    lims = [lims[0] - 0.05*span, lims[1] + 0.05*span]
    ax2.plot(lims, lims, linestyle="--", linewidth=1)  # y = x
    ax2.set_xlim(lims); ax2.set_ylim(lims)
    ax2.set_xlabel("MSE (α = 0)")
    ax2.set_ylabel("MSE (best α)")
    ax2.set_title("Task-wise comparison (points below line favor best α)")
    ax2.grid(linestyle=":", alpha=0.4)

    # --- annotate best alpha on fig 2 ---
    ax2.text(
        0.99, 0.02, f"best α = {best_alpha_str}",
        transform=ax2.transAxes, ha="right", va="bottom",
        fontsize=10, bbox=dict(facecolor="white", edgecolor="black", alpha=0.7, pad=6)
    )

    fig2.tight_layout()
    plt.show()
    return mse_best, mse_zero
    
    

    
    
    
    
        
    
    
    
    