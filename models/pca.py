import numpy as np
from typing import Tuple

def pca(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies principal component analysis (PCA) to the given data matrix X and returns the transformed
    data matrix and the corresponding eigenvectors.
    """
    # center the data
    X_centered = X - np.mean(X, axis=0)
    
    # compute the covariance matrix
    cov_mat = np.cov(X_centered.T)
    
    # compute the eigenvalues and eigenvectors of the covariance matrix
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    
    # sort the eigenvalues and eigenvectors in descending order
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    eig_pairs.sort(reverse=True, key=lambda x: x[0])
    
    # select the top n_components eigenvectors
    top_eig_vecs = np.array([eig_pairs[i][1] for i in range(n_components)])
    
    # transform the data using the selected eigenvectors
    X_transformed = np.dot(X_centered, top_eig_vecs.T)
    
    return X_transformed, top_eig_vecs
