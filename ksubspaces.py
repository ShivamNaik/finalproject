import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import gen_batches
import matplotlib.pyplot as plt
from numpy import bincount
from sklearn.preprocessing import scale
from experiments import *

# eta should be between 0.2 and 0.5
eta = 0.35

class KSubspaces:

    def __init__(self, n_clusters=8, n_init=5,
                 max_iter=300, tol=1e-4):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol

    def fit(self, X):
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = \
            k_subspaces(X, self.n_clusters, self.n_init, self.max_iter, self.tol)
        return self

    def predict(self, X):
        x_squared_norms = squarednorms(X)
        return _labels_inertia(X, x_squared_norms, self.cluster_centers_)[0]


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
    diff = np.subtract(x, ck)
    alpha = np.dot(diff.T,ak)
    dist = x-ck-np.dot(alpha, ak)
    return np.dot(dist, dist)

def planeDistance(x, ck, eta, ak, bk, variance):
    diff = np.subtract(x, ck)
    alpha = np.dot(diff.T,ak)
    beta = np.dot(diff.T,bk)
    dist = x-ck-(np.dot(alpha, ak))-(np.dot(beta, bk))
    return np.dot(dist, dist)

def sphereDistance(x, ck, eta, ak, bk, variance):
    v = np.subtract(x, ck)
    dist = np.dot(v, v)
    final = dist-(eta*variance)
    return max(0, final)

#def linedist2(point, center, eucdist, alpha):

def getalpha(diff, ak):
    return np.dot(diff.T,ak)

def line2(diff, ak):
    alpha = getalpha(diff, ak)
    a = np.dot(alpha, ak)
    v = np.subtract(diff, a)
    return np.dot(v, v)

def line3(eucdist, point, center, ak):
    diff = np.subtract(point, center)
    alpha = np.dot(diff.T,ak)
    al = np.dot(alpha, ak)
    v = eucdist
    v -= (2*np.dot(point, al))
    v +=(2*np.dot(center, al))
    v += np.dot(al, al)
    return v
    
    
def pairwise_distances(X, Y, xsquarednorms, ysquarednorms, labels, distances):
    norms = xsquarednorms.reshape(-1,1)
    n_clusters = Y.shape[0]
    n_samples = X.shape[0]
    clusters = [[] for i in range(n_clusters)]
    eucs = euclidean_distances(X, Y, Y_norm_squared=ysquarednorms, X_norm_squared=norms, squared=True)
    variances = [0 for i in range(n_clusters)]
    for ind, point in enumerate(X):
        bestcenterid, dist = min([(i[0], eucs[ind][i[0]]) \
                    for i in enumerate(Y)], key=lambda t:t[1])
        clusters[bestcenterid].append(point)
        variances[bestcenterid] += dist

    aks = [0] * n_clusters
    bks = [0] * n_clusters
    for cluster_idx in range(n_clusters):
        cluster = clusters[cluster_idx]
        if len(cluster):
            cluster_std = scale(cluster)
            pca = PCA(n_components=2).fit(cluster_std)
            aks[cluster_idx] = pca.components_[0]
            if len(pca.components_) > 1:
                bks[cluster_idx] = pca.components_[1]

    inertia = 0.0
    for ind, point in enumerate(X):
        min_dist = -1
        for center_idx, center in enumerate(Y):
            center = Y[cluster_idx]
            ak = aks[center_idx]
            #difft = (np.subtract(point,center)).T
            #a1 = np.dot(difft,ak)
            #aterm = np.dot(a1, ak)S
            bk = bks[center_idx]
            #beta = np.dot(np.dot(difft,bk), bk)
            eucdist = eucs[ind][center_idx]
            #linedist1 = eucdist-np.dot(point, aterm)-np.dot(aterm,point)+np.dot(aterm, center)+np.dot(center, aterm)+np.dot(aterm, aterm)
            #planedist = linedist - (2*np.dot(point, beta)) + (2*np.dot(center, beta)) + (2*np.dot(alpha, beta)) + np.dot(beta, beta)
            variance = variances[center_idx]
            #linedist3 = line3(eucdist, point, center, ak)
            #if linedist1 < 0:
            #    raise ValueError("Negative")
            #spheredist = max(0, (eucdist - eta*variance))
            ##
            linedist2 = lineDistance(point, center, eta, ak, bk, variance)
            planedist2 = planeDistance(point, center, eta, ak, bk, variance)
            spheredist2 = sphereDistance(point, center, eta, ak, bk, variance)
            if spheredist2==0:
                spheredist2 = np.infty
            #print "line", linedist1, linedist3
            #print "plane", planedist, planedist2
            #print "sphere", spheredist, spheredist2
            ##
            #if spheredist==0:
            #    spheredist = np.infty

            dist = min([linedist2, planedist2, spheredist2])
            if min_dist == -1 or dist < min_dist:
                min_dist = dist
                labels[ind] = center_idx
        distances[ind] = min_dist
        inertia += min_dist

    #print labels[:1000], inertia
    return inertia            
            
        
##    for cluster in clusters:
##        
##        
##    indices = np.empty(X.shape[0], dtype=np.intp)
##    
##    values = np.empty((X.shape[0],3))
##    values.fill(np.infty)
##
##    dotval = np.dot(X, Y)
##    euc = 
##    
##
##    min_indices = d_chunk.argmin(axis=1)
##    min_values = d_chunk[:, min_indices]
##    
##    bestmukey, dist = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
##                    for i in enumerate(mu)], key=lambda t:t[1])[0]

##def eucdist(X, Y, x_squared_norms, y_squared_norms):
##    dist = 0.0
##    # hardcoded: minimize euclidean distance to cluster center:
##    # ||a - b||^2 = ||a||^2 + ||b||^2 -2 <a, b>
##    dist += np.dot(point, center)
##    dist *= -2
##    dist += np.dot(center, center)
##    dist += np.dot(point, point)
##    return dist

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
    clineDispersions =  np.empty(n_clusters, dtype=X.dtype)
    cplaneDispersions = np.empty(n_clusters, dtype=X.dtype)
    csphereDispersions = np.empty(n_clusters, dtype=X.dtype)
    
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
        #eucDist = eucdist(point, center, eta, aks[label], bks[label], variances[label])
        #print "dists", lineDist, planeDist#, sphereDist, eucDist
    print "Plane dispersions", cplaneDispersions
    print "Line dispersions", clineDispersions
    

# csphereDispersions[center_idx]]
    distfunclist = [0] * n_clusters
    for center_idx in range(n_clusters):
        minimum = np.argmin([clineDispersions[center_idx], cplaneDispersions[center_idx]])
        print minimum
        if minimum==0:
            distfunclist[center_idx] = lineDistance
        elif minimum==1:
            distfunclist[center_idx] = planeDistance
        #else:
            #distfunclist[center_idx] = sphereDistance
      
    return distfunclist

def squarednorms(X):
    return (X**2).sum(axis=1)

def k_init(X, n_clusters, x_squared_norms, n_local_trials=None):
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
        # indices of where the values would be sorted within cumulative sum list
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

def _centers(X,labels, n_clusters, distances):
    n_samples, n_features = X.shape
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

def ksubspaces_single(X, n_clusters, max_iter, tol, x_squared_norms):
    centers = k_init(X, n_clusters, x_squared_norms)
    best_labels, best_inertia, best_centers = None, None, None
    n_samples = X.shape[0]
    distances = np.zeros(shape=(n_samples,), dtype=X.dtype)
    
    # iterations
    for i in range(max_iter):
        centers_old = centers.copy()
        # labels assignment -  E-step of EM
        labels, inertia = \
            _labels_inertia(X, x_squared_norms, centers,
                            distances=distances)
        # model estimation - M-step of EM
        centers = _centers(X, labels, n_clusters, distances)

        # keep track of best inertia/assignments
        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

        # check for convergence
        center_shift = np.ravel(centers_old - centers)
        center_shift_total = np.dot(center_shift, center_shift)
        #print center_shift_total
        if center_shift_total <= tol:
            print("Converged at iteration %d: "
                      "center shift %e within tolerance %e"
                      % (i, center_shift_total, tol))
            break

    return best_labels, best_inertia, best_centers, i + 1

def _tolerance(X, tol):
    """Return a tolerance which is independent of the dataset"""
    variances = np.var(X, axis=0)
    return np.mean(variances) * tol

def _labels_inertia(X, x_squared_norms, centers, distances):
    if distances is None:
        distances = np.zeros(shape=(n_samples,), dtype=X.dtype)
    n_clusters = centers.shape[0]
    n_samples = X.shape[0]
    labels = -np.ones(n_samples, np.int32)
    y_squared_norms = squarednorms(centers)
    inertia = pairwise_distances(X, centers, x_squared_norms, y_squared_norms, labels, distances)
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
    print "Distance functions:", distfunclist
    
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

def k_subspaces(X, n_clusters, n_init, max_iter, tol=1e-4):
    n_samples = X.shape[0]
    x_squared_norms = squarednorms(X)
    best_labels, best_inertia, best_centers = None, None, None
    tol = _tolerance(X, tol)
    
    for it in range(n_init):
        print "Iteration", it
        # run a k-means once
        labels, inertia, centers, n_iter_ = ksubspaces_single(
            X, n_clusters, max_iter=max_iter, tol=tol, x_squared_norms=x_squared_norms)
        # determine if these results are the best so far
        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia
            best_n_iter = n_iter_
                
    return best_centers, best_labels, best_inertia, best_n_iter


#X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
#ksub = KSubspaces(n_clusters=2).fit(X)
#best_centers, best_labels, best_inertia = k_subspaces(X, n_clusters=2)

#pokerhands = np.array(LoadData())
##print "Data loaded"
##ksub = KSubspaces(n_clusters=6).fit(pokerhands)
##print "Labels"
##print ksub.labels_
##print "Cluster centers"
##print ksub.cluster_centers_
##print "Inertia"
##print ksub.inertia_

data = ExperimentData(limit=True, limitNum=5000)
outcomes = []
for i in range(3):
    print "Dataset", i
    dat = np.array(data.dataSets[i])
    ksub = KSubspaces().fit(dat)
    kmeans = KMeans().fit(dat)
    ksubdata = (ksub.cluster_centers_, ksub.labels_, ksub.inertia_, ksub.n_iter_)
    kmeansdata = (kmeans.cluster_centers_, kmeans.labels_, kmeans.inertia_, kmeans.n_iter_)
    print "Dataset", i, "Ksub:", ksub.inertia_, "Kmeans", kmeans.inertia_
    outcomes.append([ksubdata, kmeansdata])
