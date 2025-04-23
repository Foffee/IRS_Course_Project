from scipy.sparse import csr_matrix
import numpy as np

def build_guest_user_vector(indices: list[int], size: int, rating: float = 3.0) -> csr_matrix:
    """
    Build a sparse CSR matrix representing a guest user with interactions 
    set to a fixed rating value (default 3.0) at specified item indices.

    Parameters:
    - indices: list of ALS item indices the guest interacted with
    - size: total number of items (columns in the matrix)
    - rating: fixed confidence rating for each interaction (default = 3.0)

    Returns:
    - csr_matrix of shape (1, size)
    """
    data = np.full(len(indices), rating, dtype=np.float32)
    indptr = np.array([0, len(indices)], dtype=np.int32)
    return csr_matrix((data, indices, indptr), shape=(1, size))
