import numpy as np

def align_features(X, target_dim=30):
    """
    Aligns the feature dimension of X to the target dimension.
    If X has more features, it truncates.
    If X has fewer features, it pads with zeros.
    """
    current_dim = X.shape[1]
    if current_dim == target_dim:
        return X
    elif current_dim > target_dim:
        print(f"Truncating features from {current_dim} to {target_dim}")
        return X[:, :target_dim]
    else:
        print(f"Padding features from {current_dim} to {target_dim} with zeros")
        padding = np.zeros((X.shape[0], target_dim - current_dim), dtype=X.dtype)
        return np.hstack((X, padding))
