from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def rbf_kernel_pca(X, gamma, n_components):
    
    # squared euclidean distances between observations
    sq_dists = pdist(X, metric='sqeuclidean')

    # distances vector to matrix
    mat_sq_dists = squareform(sq_dists)

    # symmetrical kernel matrix
    K = exp(-gamma*mat_sq_dists)

    # centered kernel matrix
    N = K.shape[0]
    one_n = np.ones((N, N))/N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # get eigenpairs from centered kernel matrix
    eigvals,  eigvecs = eigh(K) # no need to sort, eigh does the stuff

    # get desired number of the "biggest" eigenvectors
    alphas = np.column_stack((eigvecs[:, -i] for i in range(1, n_components+1)))

    # get desired number of eigenvalues
    lambdas = [eigvals[-i] for i in range(1, n_components+1)]

    return alphas, lambdas

def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array(
        [np.sum((x_new-row)**2) for row in X]
        )
    k = np.exp(-gamma*pair_dist)
    return k.dot(alphas/lambdas)
