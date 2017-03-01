import numpy as np

def _assign_labels_array(np.ndarray[floating, ndim=2] X,
                                  np.ndarray[floating, ndim=1] x_squared_norms,
                                  np.ndarray[floating, ndim=2] centers,
                                  np.ndarray[INT, ndim=1] labels,
                                  np.ndarray[floating, ndim=1] distances):
    """Compute label assignment and inertia for a dense array

    Return the inertia (sum of squared distances to the centers).
    
    cdef:
        unsigned int n_clusters = centers.shape[0]
        unsigned int n_features = centers.shape[1]
        unsigned int n_samples = X.shape[0]
        unsigned int x_stride
        unsigned int center_stride
        unsigned int sample_idx, center_idx, feature_idx
        unsigned int store_distances = 0
        unsigned int k
        np.ndarray[floating, ndim=1] center_squared_norms
        # the following variables are always double cause make them floating
        # does not save any memory, but makes the code much bigger
        DOUBLE inertia = 0.0
        DOUBLE min_dist
        DOUBLE dist
        DOT dot
    """
    
    center_squared_norms = np.zeros(n_clusters, dtype=np.float64)
    x_stride = X.strides[1] / sizeof(np.float64)
    center_stride = centers.strides[1] / sizeof(np.float64)
    dot = ddot

    if n_samples == distances.shape[0]:
        store_distances = 1

    for center_idx in range(n_clusters):
        center_squared_norms[center_idx] = dot(
            n_features, &centers[center_idx, 0], center_stride,
            &centers[center_idx, 0], center_stride)

    for sample_idx in range(n_samples):
        min_dist = -1
        for center_idx in range(n_clusters):
            dist = 0.0
            # hardcoded: minimize euclidean distance to cluster center:
            # ||a - b||^2 = ||a||^2 + ||b||^2 -2 <a, b>
            dist += dot(n_features, &X[sample_idx, 0], x_stride,
                        &centers[center_idx, 0], center_stride)
            dist *= -2
            dist += center_squared_norms[center_idx]
            dist += x_squared_norms[sample_idx]
            if min_dist == -1 or dist < min_dist:
                min_dist = dist
                labels[sample_idx] = center_idx

        if store_distances:
            distances[sample_idx] = min_dist
        inertia += min_dist

    return inertia

