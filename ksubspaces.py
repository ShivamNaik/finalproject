import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from numpy import bincount

eta = 0.2

class KSubspaces:

    def __init__(self, n_clusters=8,
                 max_iter=300, tol=1e-4, precompute_distances='auto',
                 n_jobs=1, algorithm='auto'):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.precompute_distances = precompute_distances
        self.n_jobs = n_jobs
        self.algorithm = algorithm

    def fit(self, X):
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = \
            k_subspaces(X, n_clusters=self.n_clusters, return_n_iter=True)
        return self

    def predict(self, X):
        x_squared_norms = xsquarednorms(X)
        return _labels_inertia(X, self.n_clusters, x_squared_norms, self.cluster_centers_)[0]


def LoadData():
    f = open("poker-hand-testing.txt")
    pokerhands = []
    count = 0
    while count < 1000:
        for line in f:
            splitline = line.split(",")
            pokerhands.append(map(int, map(str.strip, splitline)))
            count += 1
    f.close()
    return pokerhands

def lineDistance(x, ck, ak):
    alpha = np.dot((x - ck).T,ak)
    dist = x-ck-np.dot(alpha, ak)
    return np.dot(dist, dist)

def planeDistance(x, ck, ak, bk):
    alpha = np.dot((x - ck).T,ak)
    beta = np.dot((x - ck).T,bk)
    dist = x-ck-(np.dot(alpha, ak))-(np.dot(beta, bk))
    return np.dot(dist, dist)

def getCenters(X, n, n_clusters, labels):
    # Xk is datapoints in Ck (equation 7)
    centers = [[0.0] * n] * n_clusters
    sizes = [0] * n_clusters
    for i in range(n):
        centers[label[i]] += X[i]
        sizes[label[i]] += 1

    for j in range(n_clusters):
        centers[j]/sizes[j]

    return centers

def firstDirection(Xk):
    # do PCA and return first direction
    return

def secondDirection(Xk):
    # do PCA and return second direction
    return

def sphereDistance(x, ck, eta, variance):
    # eta should be between 0.2 and 0.5
    v = np.subtract(x, ck)
    dist = np.dot(v, v)
    final = dist-(eta*variance)
    return max(0, final)

def findVariance(X, n_samples, centers, n_clusters, labels):
    variances = [0] * n_clusters

    for idx in range(n_samples):
        point = X[idx]
        label = labels[idx]
        center = centers[label]
        dist = np.subtract(point, center)
        variances[label] += (np.dot(dist, dist))
    return variances

def determinedistfuncs(X, n_samples, centers, n_clusters, labels):
    # for each cluster, find which model has lowest dispersion
    ak = firstDirection(Xk)
    bk = secondDirection(Xk)
    variances = findVariances(X, n_samples, centers, n_clusters, labels)
    
    lineDispersions = [0] * n_clusters
    planeDispersions = [0] * n_clusters
    sphereDispersions = [0] * n_clusters
    
    for idx in range(n_samples):
        point = X[idx]
        label = labels[idx]
        center = centers[label]
        lineDist = lineDistance(point, center, ak)
        lineDispersions[label] += lineDist
        planeDist = planeDistance(point, center, ak, bk)
        planeDispersions[label] += planeDist
        sphereDist = sphereDistance(point, center, eta, variances[label])
        sphereDispersions[label] += sphereDist

    distfunclist = [0] * n_clusters
    for center_idx in range(n_clusters):
        minimum = np.argmin([lineDispersions[center_idx], planeDispersions[center_idx], sphereDispersions[center_idx]])
        if minimum is 1:
            distfunclist[center_idx] = lineDistance
        elif minimum is 2:
            distfunclist[center_idx] = planeDistance
        else:
            distfunclist[center_idx] = sphereDistance
        
    return distfunclist
    

def xsquarednorms(X):
    return np.array([np.dot(x,x) for x in X])

def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum

    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    axis : int, optional
        Axis along which the cumulative sum is computed.
        The default (None) is to compute the cumsum over the flattened array.
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, axis=axis, dtype=np.float64)
    expected = np.sum(arr, axis=axis, dtype=np.float64)
    if not np.all(np.isclose(out.take(-1, axis=axis), expected, rtol=rtol,
                             atol=atol, equal_nan=True)):
        warnings.warn('cumsum was found to be unstable: '
                      'its last element does not correspond to sum',
                      RuntimeWarning)
    return out

def _k_init(X, n_clusters, x_squared_norms, n_local_trials=None):
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    assert x_squared_norms is not None, 'x_squared_norms None in _k_init'

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = np.random.randint(n_samples)
    centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
        squared=True)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = np.random.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                        rand_vals)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        # Decide which candidate is the best
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq,
                                     distance_to_candidates[trial])
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        # Permanently add best center candidate found in local tries
        centers[c] = X[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return centers

def clusterAssignment(X):
    # X is float64 array-like
    # E-step of EM
    # estimating the posterior probability for all data points belonging to different clusters
    # should return labels and inertia
        
    # set centers to -1 
    n_samples = X.shape[0]
    clusterLabels = -np.ones(n_samples, np.int32)
    x_squared_norms = xsquarednorms(X)
    
    if distances is None:
        distances = np.zeros(shape=(0,), dtype=X.dtype)
        
    inertia = assign_labels(X, x_squared_norms, centers, clusterLabels, distances=distances)
    

def modelEstimation():
    # M-step of EM
    return

def _centers_dense(X, labels, n_clusters, distances):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    centers = np.zeros((n_clusters, n_features), dtype=np.float32)

    n_samples_in_cluster = bincount(labels, minlength=n_clusters)
    empty_clusters = np.where(n_samples_in_cluster == 0)[0]

    if len(empty_clusters):
        # find points to reassign empty clusters to
        far_from_centers = distances.argsort()[::-1]

        for i, cluster_id in enumerate(empty_clusters):
            # XXX two relocated clusters could be close to each other
            new_center = X[far_from_centers[i]]
            centers[cluster_id] = new_center
            n_samples_in_cluster[cluster_id] = 1

    for i in range(n_samples):
        for j in range(n_features):
            centers[labels[i], j] += X[i, j]

    centers /= n_samples_in_cluster[:, np.newaxis]

    return centers

def squared_norm(x):
    """Squared Euclidean or Frobenius norm of x.

    Returns the Euclidean norm when x is a vector, the Frobenius norm when x
    is a matrix (2-d array). Faster than norm(x) ** 2.
    """
    x = np.ravel(x)
    return np.dot(x, x)

def ksubspaces_single(X, n_clusters, max_iter, precompute_distances, tol, x_squared_norms):
    centers = _k_init(X, n_clusters, x_squared_norms)
    best_labels, best_inertia, best_centers = None, None, None
    distances = np.zeros(shape=(X.shape[0],), dtype=X.dtype)

    # iterations
    for i in range(max_iter):
        print i
        centers_old = centers.copy()
        # labels assignment is also called the E-step of EM
        labels, inertia = \
            _labels_inertia(X, n_clusters, x_squared_norms, centers,
                            precompute_distances=precompute_distances,
                            distances=distances)
        centers = _centers_dense(X, labels, n_clusters, distances)
        
        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

        center_shift = centers_old - centers
        center_shift_total = squared_norm(center_shift)
        if center_shift_total <= tol:
            print("Converged at iteration %d: "
                      "center shift %e within tolerance %e"
                      % (i, center_shift_total, tol))
            break

    if center_shift_total > 0:
        # rerun E-step in case of non-convergence so that predicted labels
        # match cluster centers
        best_labels, best_inertia = \
            _labels_inertia(X, n_clusters=n_clusters, x_squared_norms=x_squared_norms,
                            centers=best_centers,
                            precompute_distances=precompute_distances,
                            distances=distances)

    return best_labels, best_inertia, best_centers, i + 1

def _labels_inertia(X, n_clusters, x_squared_norms, centers,
                    precompute_distances=True, distances=None):
    n_samples = X.shape[0]
    labels = -np.ones(n_samples, np.int32)
    if distances is None:
        distances = np.zeros(shape=(0,), dtype=X.dtype)
    # distances will be changed in-place
    inertia = _assign_labels_array(X, n_clusters, x_squared_norms, centers, labels, distances=distances)
    return labels, inertia

def eucdist(X, x_squared_norms, centers, center_squared_norms, sample_idx, center_idx):
    dist = 0.0
    # hardcoded: minimize euclidean distance to cluster center:
    # ||a - b||^2 = ||a||^2 + ||b||^2 -2 <a, b>
    dist += np.dot(X[sample_idx], centers[center_idx])
    dist *= -2
    dist += center_squared_norms[center_idx]
    dist += x_squared_norms[sample_idx]
    return dist
      
def _assign_labels_array(X, n_clusters, x_squared_norms, centers, labels, distances):
    n_samples = X.shape[0]
    if n_samples == distances.shape[0]:
        store_distances = 1
    else:
        store_distances = 0

    center_squared_norms = [np.dot(x,x) for x in centers]
    distfunclist = determinedistfuncs(X, n_samples, centers, n_clusters, labels)

    inertia = 0.0
    # for each cluster, choose model assignment and use that for distance
    for sample_idx in range(n_samples):
        min_dist = -1
        for center_idx in range(n_clusters):
            distfunc = distfunclist[center_idx]
            dist = distfunc(X, x_squared_norms, centers, center_squared_norms, sample_idx, center_idx)
            if min_dist == -1 or dist < min_dist:
                min_dist = dist
                labels[sample_idx] = center_idx

        if store_distances:
            distances[sample_idx] = min_dist
        inertia += min_dist

    return inertia

def k_subspaces(X, n_clusters=8, n_init=5, max_iter=300, 
                precompute_distances='auto', tol=1e-4, 
                n_jobs=1, return_n_iter=False):
    
    if precompute_distances == 'auto':
        n_samples = X.shape[0]
        precompute_distances = (n_clusters * n_samples) < 12e6
    elif isinstance(precompute_distances, bool):
        pass
    else:
        raise ValueError("precompute_distances should be 'auto' or True/False"
                         ", but a value of %r was passed" %
                         precompute_distances)
    
    x_squared_norms = xsquarednorms(X)
    best_labels, best_inertia, best_centers = None, None, None
    
    if n_jobs == 1:
        # For a single thread, less memory is needed if we just store one set
        # of the best results (as opposed to one set per run per thread).
        for it in range(n_init):
            # run a k-means once
            labels, inertia, centers, n_iter_ = ksubspaces_single(
                X, n_clusters, max_iter=max_iter,
                precompute_distances=precompute_distances, tol=tol,
                x_squared_norms=x_squared_norms)
            # determine if these results are the best so far
            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia
                best_n_iter = n_iter_
                
    if return_n_iter:
        return best_centers, best_labels, best_inertia, best_n_iter
    else:
        return best_centers, best_labels, best_inertia

pokerhands = np.array(LoadData())
#X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
#best_centers, best_labels, best_inertia = k_subspaces(X, n_clusters=2)

print "Data loaded"
ksub = KSubspaces(n_clusters=6).fit(pokerhands)
print "Labels"
print ksub.labels_
print "Cluster centers"
print ksub.cluster_centers_
