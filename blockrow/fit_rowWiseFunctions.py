# Functions to fit MTW regression 

# Import libraries 
from mtw.mtw import MTW
import numpy as np
import warnings 

def convert_to_regression(data_matrix):
        # VAR(1)
        p = 1
        T, d = data_matrix.shape
        N = T - p

        # Outcomes: Y in shape (d*N, 1), stacked by variable (var0's N first, etc.)
        Y = data_matrix[p:]            # (N, d)
        Y = Y.T.reshape(-1, 1)         # (d*N, 1)

        # Predictors X (N, d)
        X = data_matrix[0:T-1]

        # Kronecker design Z = kron(I_d, X) -> (d*N, d*d)
        Z = np.kron(np.eye(d), X)

        return Y, Z, N
    
### Things to add ###
# Set a number of time points to predict into the future as an argument #
# Make this function callable such that we can like put it into some kind of package and they can just enter the information#


def flat_to_time_view(y_flat, d):
    """(N*d, ) to (N, d) view (ordered by time). Used to predict one time step ahead in the future."""
    y = np.asarray(y_flat).reshape(-1) # Array should already be in this form 
    N = y_flat.size // d
    return y.reshape(d, N).T

def cv_fitWassRowWise(ts_data_list, ground_M, wassPen_vals, n_folds, fold_size, horizon): 
    # Sanity checks
    n_tasks = len(ts_data_list)
    print("Tasks:", n_tasks)
    T, d = ts_data_list[0].shape
    print("Shape of one task (time points, variables):", ts_data_list[0].shape)

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
    
    def flat_time_block_indices(i0, i1, N, d): 
        """Indices to get the proper time window [i0, i1]"""
        return np.concatenate([np.arange(v*N + i0, v*N + i1) for v in range(d)], axis=0)

    # print("Example shapes before slicing:")
    # print("Z[0] shape:", Xs[0].shape, " (should be (d*N, d*d))")
    # print("Y[0] shape:", Ys[0].shape, " (should be (d*N, 1))")
    cv_table = []
    best_avg_mse = float('inf')
    best_alpha = None
    for alpha in wassPen_vals: 
        fold_mses = []
        for k in range(n_folds):
            i0, i1 = k * fold_size, min((k+1) * fold_size, N)
            validation_idx = flat_time_block_indices(i0, i1, N, d) # Indices to keep out
            n_flat = N * d 
            keep_mask = np.ones(n_flat, dtype=bool)
            keep_mask[validation_idx] = False
            keep_idx = np.where(keep_mask)[0] # Get first array of the tuple
            
            # print("Keep idx:", keep_idx)
            # print("X_train before removing indexes:", Xs_array.shape)
            # print("Y_train before removing indexes:", Ys_array.shape)
            X_train = Xs_array[:, keep_idx, :] # Train on values not left out on this fold
            Y_train = Ys_array[:, keep_idx] # Train for values not left out on this fold
            N_train = N - (i1 - i0) # Reduced time length due to leaving out the fold
            # print("X_train after removing indexes shape:", X_train.shape)
            # print("Y_train after removing indexes shape:", Y_train.shape)
            # print("N_train:", N_train)
        
            out = fit_WassRowWise(wassPen_vals=[alpha], ground_M=ground_M, Xs=X_train, Ys=Y_train, N=N_train, n_tasks=n_tasks, d=d)

            est_Phis = out[0]['est_Phi'] # Extract all estimated coefficient matrices
            
            
            def forecast(horizon, Ys_array, est_Phis, ts_data_list):
                # Predicts the validation fold for defined time length in future (horizon)
                # TODO MODIFY THIS FUNCTION SO THAT IT WORKS TO TAKE ANY FORECAST LENGTH
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
        
            task_mses, _ = forecast(horizon=horizon, Ys_array=Ys_array, est_Phis=est_Phis, ts_data_list=ts_data_list)
            fold_mses.append(float(np.mean(task_mses)))
            
        avg_mse = float(np.mean(fold_mses)) if fold_mses else float('inf')
        cv_table.append({"alpha": alpha, "fold_mses": fold_mses, "avg_mse": avg_mse})

        if avg_mse < best_avg_mse:
            best_avg_mse = avg_mse
            best_alpha = alpha  
              
    # ---- Full-data fits for ALL alphas (so fit_rowWiseVAR.py can compute MSE vs truth) ----
    per_alpha_fit = fit_WassRowWise(
        wassPen_vals=wassPen_vals,
        ground_M=ground_M,
        Xs=Xs_array,
        Ys=Ys_array,
        N=N,                 # full length
        n_tasks=n_tasks,
        d=d,
        max_iter=50000
    )
    # per_alpha_fit is a list like:
    #   [{'wassPen': a, 'est_Phi': [Phi_task0, Phi_task1, ...]}, ...]
    
    result = {
        "cv_table": cv_table,
        "best_alpha": best_alpha,
        "best_avg_mse": best_avg_mse,
        "per_alpha_fit": per_alpha_fit
    }
    
    # print("Result:", result)

    return result
                

# Row-wise MTW for VAR(1), predicting each variable separately
def fit_WassRowWise(wassPen_vals,
                    ground_M, Xs, Ys, N, n_tasks, d,
                    max_iter=50000):
    """
    Fits multitask VAR(1) row-by-row with MTW.
    Keeps your Kronecker Z = kron(I_d, X) construction, but for each row j:
      - selects the Y and Z rows corresponding to target j,
      - selects the Z columns corresponding to row-j coefficients (length d),
      - stacks across tasks, fits MTW, and assembles Phi per task.

    Returns
    -------
    out_results : list of dicts, one per alpha:
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
    results_map = {alpha: [np.zeros((d, d), dtype=float) for _ in range(n_tasks)]
                   for alpha in wassPen_vals}

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
            y_row_k = Y_k[row_slice]                    # (N,)

            # Defensive checks
            assert X_row_k.shape == (N, d)
            assert y_row_k.shape == (N,)

            Xs_row.append(X_row_k)
            Ys_row.append(y_row_k)

        Xs_array = np.stack(Xs_row, axis=0)  # Prev shape: (n_tasks, N, d)
        Ys_array = np.stack(Ys_row, axis=0)  # Prev shape: (n_tasks, N)

        # Fit MTW for each alpha and fill row j of Phi across tasks
        for alpha in wassPen_vals:
            # Creates aligned dxd block from ground metric M (diagonal blocks)
            M_block = ground_M[j * d:(j + 1) * d, j * d:(j + 1) * d]
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
    for alpha in wassPen_vals:
        out_results.append({
            'wassPen': alpha,
            'est_Phi': results_map[alpha]  # list length n_tasks, each (d,d)
        })

    return out_results