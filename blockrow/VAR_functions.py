# Import libraries
import numpy as np
import warnings

# Simulate VAR(1)
def simulateVAR1(Phi, Sigma, T = 100, burn_in = 1000):

    """
    Function to simulate VAR(1)
    """

    # Check inputs
    assert type(Phi) is np.ndarray
    assert type(Sigma) is np.ndarray
    assert Phi.shape[0] == Phi.shape[1]
    assert Sigma.shape[0] == Sigma.shape[1]
    assert Phi.shape[0] == Sigma.shape[0]
    assert len(Phi.shape) == 2
    assert len(Sigma.shape) == 2
    assert np.allclose(Sigma.T, Sigma)
    assert type(T) is int

    # Get dimension 
    k = Phi.shape[0]

    # Total number of time points
    TTotal = T + burn_in

    # Initialize container 
    y = np.zeros((TTotal, k))
    y[0,] = np.random.normal(0,1,size=k) 

    # Generate errors 
    errors = np.random.multivariate_normal(mean = np.zeros(k),
                                           cov = Sigma,
                                           size = TTotal - 1)

    # Generate TS 
    for t in range(1, TTotal):
        y[t] = Phi @ y[t-1] + errors[t-1]

    # Drop burn-in samples 
    return(y[burn_in:,])


# Stability condition
def make_matrix_stable(Phi, max_radius=0.95):

    # Get maximal eigenvalue 
    eigvals = np.linalg.eigvals(Phi)
    spectral_radius = max(abs(eigvals))

    # Set to maximum radius  
    if spectral_radius >= max_radius:
        Phi = (max_radius / spectral_radius) * Phi

    return Phi


# Finds nearest positive definite matrix to A
def nearest_pd(A, eps = 1e-8): 

    # Get eigenvalues 
    eigval, eigvec = np.linalg.eigh(A)

    # Lift negative eigenvalues if applicable 
    if np.any(eigval < eps):
        eigval[eigval < eps] = eps          
        warn_msg = """Matrix is not PD; enforcing PD
         through eigenvalue decomposition"""
        warnings.warn(warn_msg)
    
    return (eigvec * eigval) @ eigvec.T


# Generate VAR matrices 
def gen_VARmat(num_tasks,
               num_vars,
               coef_mean_struct = None,
               coef_cov_mat = None,
               use_seed = None):

    # If both inputs mean and cov are set to None,
    # generate from mean zero isotropic gaussian with unit variance 
    if coef_mean_struct is None:
        coef_mean_struct = np.zeros(num_vars ** 2)
    
    if coef_cov_mat is None:
        coef_cov_mat = np.eye(num_vars ** 2)

    # Force PD if necessary 
    coef_cov_mat = nearest_pd(coef_cov_mat)

    # Seed the generator 
    rng = np.random.default_rng(seed = use_seed)

    # Create VAR matrices 
    VAR_matrices = []
    for _ in range(num_tasks):

        # Generate Phi
        phivec = rng.multivariate_normal(mean = coef_mean_struct,
                                         cov = coef_cov_mat)
        Phi = phivec.reshape(num_vars, num_vars, order = 'C')
        Phi = make_matrix_stable(Phi)

        # Generate Psi 
        psivec = rng.multivariate_normal(mean = np.zeros(num_vars ** 2),
                                         cov = np.eye(num_vars ** 2))
        Psi = psivec.reshape(num_vars, num_vars, order = 'C')
        Psi = nearest_pd(Psi)

        # Append as dictionary
        VAR_matrices.append({'Phi': Phi,
                             'Psi': Psi})
    
    return VAR_matrices
    
