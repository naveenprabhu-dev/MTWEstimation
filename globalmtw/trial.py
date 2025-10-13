### This is for testing 

# import numpy as np 
# from simulateVAR1 import simulateVAR1
# from numpy.random import default_rng


# # Error covariance 
# Sigma = np.array([      
#    [1, 0.50, -0.60],
#    [0.50,  1,  -0.60],
#    [-0.60,  -0.60,  1]
# ])


# def make_matrix_stable(Phi, max_radius=0.95):
#     eigvals = np.linalg.eigvals(Phi)
#     spectral_radius = max(abs(eigvals))
#     if spectral_radius >= max_radius:
#         Phi = (max_radius / spectral_radius) * Phi
#     return Phi

# means = np.array([       # base structure matrix (s, d, h)
#     0.50, 0.50, -0.60,
#    0.50,  0.50,  -0.60,
#    -0.60,  -0.60,  0.50
# ])

# rng = default_rng(42)
# num_tasks = 1
# for _ in range(num_tasks):
#     vec = rng.multivariate_normal(mean=means, cov=np.eye(9))
#     Phi = vec.reshape(3, 3)
#     Phi = make_matrix_stable(Phi)

# out_ts = simulateVAR1(Phi = Phi,
#                       Sigma = Sigma,
#                       T = 7)

# print(out_ts)


from groundmetric import *
from VAR_functions import *
import numpy as np 

D = np.array([[1,3,0.5],
              [4,5,3],
              [7,8,9]])
# D = D.astype(float)
# D[np.tril_indices(3)] = np.nan
# print(D)
# print(np.isnan(D[np.tril_indices(3)]))
# print(np.all(D[np.tril_indices(3)] == np.nan))
# bfunc = create_base_distance_func(D, ["x", "y", "z"])
# print(bfunc("x", "z"))
# print(bfunc("z", "x"))
# print(bfunc("x", "y"))
# print(bfunc("y", "z"))
# print(bfunc("y", "z"))


M, cnames = create_ground_metric(bDist_mat = D, varnames = ["x", "y", "z"])
print(M)
print(cnames)

from pprint import pprint


coef_cov = np.exp(-0.9 * M)
print(coef_cov)

test =  gen_VARmat(num_tasks = 5,
                   num_vars = 3,
                   coef_mean_struct = None,
                   coef_cov_mat = coef_cov,
                   use_seed = 12345)


# # pprint(test)
# print(nearest_pd(coef_cov))

# # Check correlation (between 1st and 3rd element of Phi)
# vals = [ (x["Phi"][0,1], x["Phi"][1,0]) for x in test ]
# # print(vals)

# x_vals, y_vals = zip(*vals)
# corr = np.cov(x_vals, y_vals)[0,1]
# print(corr)