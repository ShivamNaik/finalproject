import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import gen_batches
from numpy import bincount
from sklearn.preprocessing import scale
from experiments import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from plottingtools import *

# eta should be between 0.2 and 0.5
eta = 0.35

class KSubspaces:

    def __init__(self, n_clusters=8, n_init=3,
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
            #if linedist1 < 0:
            #    raise ValueError("Negative")
            #spheredist = max(0, (eucdist - eta*variance))
            ##
            linedist2 = lineDistance(point, center, eta, ak, bk, variance)
            planedist2 = planeDistance(point, center, eta, ak, bk, variance)
            spheredist2 = sphereDistance(point, center, eta, ak, bk, variance)
            if spheredist2==0:
                spheredist2 = np.infty
            #print "line", linedist1, linedist2
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

def plot(xs, ys, zs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.scatter(xs, ys, zs)
    plt.show()
    
def syntheticData2Lines():
    xs1 = []
    ys1 = []
    zs1 = []
    points1 = []
    start1 = [15, 5, 1]
    for i in range(50):
        idx = 1
        start1[idx] += 1
        xs1.append(start1[0])
        ys1.append(start1[1])
        zs1.append(start1[2])
        points1.append([start1[0], start1[1], start1[2]])
        
    xs2 = []
    ys2 = []
    zs2 = []
    points2 = []
    start2 = [5, 0, 0]
    for i in range(50):
        start2[0] += 0.5
        start2[1] += 0.5
        start2[2] -= 0.1
        xs2.append(start2[0])
        ys2.append(start2[1])
        zs2.append(start2[2])
        points2.append([start2[0], start2[1], start2[2]])

    l = points1
    l.extend(points2)
    #plot2(xs1, ys1, zs1, xs2, ys2, zs2)
    return np.array(l)

def plot2(xs1, ys1, zs1, xs2, ys2, zs2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.scatter(xs1, ys1, zs1, c='b')
    ax.scatter(xs2, ys2, zs2, c='r')
    plt.show()

    
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

# dat = syntheticData2Lines()
# ksub = KSubspaces(n_clusters=2).fit(dat)
# plotClusterData(dat, ksub.labels_, 2)
# kmeans = KMeans(n_clusters=2).fit(dat)
# plotClusterData(dat, kmeans.labels_, 2)
