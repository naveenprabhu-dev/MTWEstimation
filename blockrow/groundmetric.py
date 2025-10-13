# Functions to specify how the ground metric in a VAR should be 

# Libraries
import warnings
import numpy as np 
from itertools import permutations, product


def create_ground_metric(bDist_mat, varnames = None):
    """
    Creates the ground metric M
    """

    # Get number of variables 
    k = bDist_mat.shape[0]

    # Create coefficient names 
    if varnames is None:
        varnames = ["V" + str(i) for i in range(1, k+1)]
    else:
        # Check if dimensions match 
        assert k == len(varnames)

    # Create function 
    base_distance = create_base_distance_func(bDist_mat = bDist_mat,
                                              varnames = varnames)

    # Create coefficient names (using ~ notation)
    varpairs = list(product(varnames, repeat = 2))
    coefnames = [ x + '~' + y for (x, y) in varpairs ]

    # Initialize ground metric 
    M = D = np.zeros((k ** 2,k ** 2), dtype = float)

    # Fill in upper triangular, then copy
    for i, a in enumerate(coefnames):
        for j, b in enumerate(coefnames):
            if i < j:  
                D[i, j] = D[j, i] = coeff_distance(a, b, 
                                                   dist_func = base_distance)

    return M, coefnames

def create_base_distance_func(bDist_mat, varnames):
    """
    Creates a function 'base_distance(a, b)' that 
    takes in two variables in a VAR and outputs the distance

    Inputs:
        bDist_mat: upper-triangular matrix with NAs elsewhere specifying
            the distance between variables. 
    """

    # Check input 
    assert bDist_mat.shape[0] == bDist_mat.shape[1]
    assert len(bDist_mat.shape) == 2
    assert len(varnames) == len(set(varnames))

    # Get number of variables 
    k = bDist_mat.shape[0]
    nuniq_pairs = k * (k-1) / 2

    # Set to float
    bDist_mat = bDist_mat.astype(float)

    # Check if input is upper-triangular 
    if not np.all(np.isnan(bDist_mat[np.tril_indices(k)])):
        warntext = """
        Lower triangular elements (including diagonal) are not np.nan. These
        elements are ignored. Using upper triangular elements only.
        """
        warnings.warn(warntext)
        bDist_mat[np.tril_indices(k)] = np.nan

    # Create function 
    def base_distance(a, b):
        pairs = list(permutations(varnames, 2))
        if (a, b) in pairs:
            ab_indices = [i for i, x in enumerate(varnames) if x in [a, b]]
            i1 = ab_indices[0]
            i2 = ab_indices[1]
            if (i1 > i2): 
                out_value = bDist_mat[i2, i1]
            elif (i1 < i2):
                out_value = bDist_mat[i1, i2]
            else:
                raise ValueError("Indices cannot be the same.") 
        elif a == b and a in varnames:
            out_value = 0
        else:
            raise ValueError("Variable pair not found based on provided names.")
        return out_value 

    return base_distance


def coeff_distance(a, b, dist_func):
    """
    Distance between two coefficients Vx~Vy and Vz~Vk
    """

    # Get varnames 
    y1, x1 = [ part.strip() for part in a.split("~") ]
    y2, x2 = [ part.strip() for part in b.split("~") ]

    # Check type 
    same_outcome = (y1 == y2)
    auto1 = (y1 == x1)              # Check if autoregressive
    auto2 = (y2 == x2)              # Check if autoregressive

    # Check cases 
    if same_outcome:

        # Case 1: same outcome column
        out_val = dist_func(x1, x2)

    elif not auto1 or not auto2:

        # Case 2: different outcomes, at least one cross‑variable
        out_val = abs(dist_func(x1, y1) - dist_func(x2, y2)) # difference of differences

    else: 

        # Case 3: different outcomes, both auto
        out_val = dist_func(x1, x2)  # use rule 1 on the outcomes

    return out_val


