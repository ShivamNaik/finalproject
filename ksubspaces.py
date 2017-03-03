import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from numpy import bincount
from sklearn.preprocessing import scale

# eta should be between 0.2 and 0.5
eta = 0.2

class KSubspaces:

    def __init__(self, n_clusters=8, n_init=5,
                 max_iter=300, tol=1e-4, precompute_distances='auto',
                 n_jobs=1, algorithm='auto'):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.precompute_distances = precompute_distances
        self.n_jobs = n_jobs
        self.algorithm = algorithm

    def fit(self, X):
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = \
            k_subspaces(X, self.n_clusters, self.n_init, self.max_iter, return_n_iter=True)
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

def lineDistance(x, ck, eta, ak, bk, variance):
    alpha = np.dot((x - ck).T,ak)
    dist = x-ck-np.dot(alpha, ak)
    return np.dot(dist, dist)

def planeDistance(x, ck, eta, ak, bk, variance):
    alpha = np.dot((x - ck).T,ak)
    beta = np.dot((x - ck).T,bk)
    dist = x-ck-(np.dot(alpha, ak))-(np.dot(beta, bk))
    return np.dot(dist, dist)

def sphereDistance(x, ck, eta, ak, bk, variance):
    v = np.subtract(x, ck)
    dist = np.dot(v, v)
    final = dist-(eta*variance)
    return max(0, final)

def eucdist(point, center, eta, ak, bk, variance):
    dist = 0.0
    # hardcoded: minimize euclidean distance to cluster center:
    # ||a - b||^2 = ||a||^2 + ||b||^2 -2 <a, b>
    dist += np.dot(point, center)
    dist *= -2
    dist += np.dot(center, center)
    dist += np.dot(point, point)
    return dist

def findParams(X, n_samples, centers, n_clusters, labels):
    variances = [0] * n_clusters
    aks = [0] * n_clusters
    bks = [0] * n_clusters

    datapoints_per_cluster = [[] for i in range(n_clusters)]
    for idx in range(n_samples):
        point = X[idx]
        label = labels[idx]
        if label==-1:
            return variances, aks, bks
        center = centers[label]
        datapoints_per_cluster[label].append(point)
        dist = np.subtract(point, center)
        variances[label] += (np.dot(dist, dist))

    #print datapoints_per_cluster
    for cluster_idx in range(n_clusters):
        cluster = datapoints_per_cluster[cluster_idx]
        if len(cluster):
            cluster_std = scale(cluster)
            pca = PCA(n_components=2).fit(cluster_std)
            aks[cluster_idx] = pca.components_[0]
            if len(pca.components_) > 1:
                bks[cluster_idx] = pca.components_[1]
    return variances, aks, bks

def determinedistfuncs(X, n_samples, centers, n_clusters, labels, variances, aks, bks):
    clineDispersions = [0] * n_clusters
    cplaneDispersions = [0] * n_clusters
    csphereDispersions = [0] * n_clusters
    
    for idx in range(n_samples):
        point = X[idx]
        label = labels[idx]
        if label==-1:
            return [eucdist] * n_clusters
        center = centers[label]
        lineDist = lineDistance(point, center, eta, aks[label], bks[label], variances[label])
        # line dispersion per cluster
        clineDispersions[label] += lineDist
        planeDist = planeDistance(point, center, eta, aks[label], bks[label], variances[label])
        cplaneDispersions[label] += planeDist
        sphereDist = sphereDistance(point, center, eta, aks[label], bks[label], variances[label])
        csphereDispersions[label] += sphereDist
        eucDist = eucdist(point, center, eta, aks[label], bks[label], variances[label])
        #print "dists", lineDist, planeDist, sphereDist, eucDist

    distfunclist = [0] * n_clusters
    for center_idx in range(n_clusters):
        minimum = np.argmin([clineDispersions[center_idx], cplaneDispersions[center_idx], csphereDispersions[center_idx]])
        if minimum is 0:
            distfunclist[center_idx] = lineDistance
        elif minimum is 1:
            distfunclist[center_idx] = planeDistance
        else:
            distfunclist[center_idx] = sphereDistance
      
    return distfunclist

def xsquarednorms(X):
    return np.array([np.dot(x,x) for x in X])

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
        candidate_ids = np.searchsorted(np.cumsum(closest_dist_sq),
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

def _centers_dense(X, n_samples, labels, n_clusters, distances):
    n_features = X.shape[1]
    centers = np.zeros((n_clusters, n_features), dtype=np.float32)

    n_samples_in_cluster = bincount(labels, minlength=n_clusters)
    empty_clusters = np.where(n_samples_in_cluster == 0)[0]

    if len(empty_clusters):
        # find points to reassign empty clusters to
        far_from_centers = distances.argsort()[::-1]

        for i, cluster_id in enumerate(empty_clusters):
            new_center = X[far_from_centers[i]]
            centers[cluster_id] = new_center
            n_samples_in_cluster[cluster_id] = 1

    for i in range(n_samples):
        for j in range(n_features):
            centers[labels[i], j] += X[i, j]

    centers /= n_samples_in_cluster[:, np.newaxis]

    return centers

def ksubspaces_single(X, n_clusters, max_iter, precompute_distances, tol, x_squared_norms):
    centers = _k_init(X, n_clusters, x_squared_norms)
    best_labels, best_inertia, best_centers = None, None, None
    distances = np.zeros(shape=(X.shape[0],), dtype=X.dtype)
    n_samples = X.shape[0]

    # iterations
    for i in range(max_iter):
        print i
        centers_old = centers.copy()
        # labels assignment -  E-step of EM
        labels, inertia = \
            _labels_inertia(X, n_clusters, x_squared_norms, centers,
                            precompute_distances=precompute_distances,
                            distances=distances)
        centers = _centers_dense(X, n_samples, labels, n_clusters, distances)
        
        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

        center_shift = np.ravel(centers_old - centers)
        center_shift_total = np.dot(center_shift, center_shift)
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
    labels = np.random.randint(n_clusters, size=n_samples)
    if distances is None:
        distances = np.zeros(shape=(0,), dtype=X.dtype)
    # distances will be changed in-place
    inertia = _assign_labels_array(X, n_clusters, x_squared_norms, centers, labels, distances=distances)
    return labels, inertia



def _assign_labels_array(X, n_clusters, x_squared_norms, centers, labels, distances):
    n_samples = X.shape[0]
    if n_samples == distances.shape[0]:
        store_distances = 1
    else:
        store_distances = 0

    #center_squared_norms = [np.dot(x,x) for x in centers]
    
    # for each cluster, find which model has lowest dispersion
    variances, aks, bks = findParams(X, n_samples, centers, n_clusters, labels)
    distfunclist = determinedistfuncs(X, n_samples, centers, n_clusters, labels, variances, aks, bks)

    inertia = 0.0
    # for each cluster, choose model assignment and use that for distance
    for idx in range(n_samples):
        min_dist = -1
        point = X[idx]
        for center_idx in range(n_clusters):
            center = centers[center_idx]
            distfunc = distfunclist[center_idx]
            dist = distfunc(point, center, eta, aks[center_idx], bks[center_idx], variances[center_idx])
            if min_dist == -1 or dist < min_dist:
                min_dist = dist
                labels[idx] = center_idx

        if store_distances:
            distances[idx] = min_dist
        inertia += min_dist

    #print labels
    return inertia

def k_subspaces(X, n_clusters, n_init, max_iter, 
                precompute_distances='auto', tol=1e-4, 
                return_n_iter=False):
    
    if precompute_distances == 'auto':
        n_samples = X.shape[0]
        precompute_distances = (n_clusters * n_samples) < 12e6
    
    x_squared_norms = xsquarednorms(X)
    best_labels, best_inertia, best_centers = None, None, None
    
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


#X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
#ksub = KSubspaces(n_clusters=2).fit(X)
#best_centers, best_labels, best_inertia = k_subspaces(X, n_clusters=2)

pokerhands = np.array(LoadData())
print "Data loaded"
ksub = KSubspaces(n_clusters=6).fit(pokerhands)
print "Labels"
print ksub.labels_
print "Cluster centers"
print ksub.cluster_centers_
print "Inertia"
print ksub.inertia_
