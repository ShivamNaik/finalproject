import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from . import _k_means

def lineDistance(x, ck, ak):
    alpha = np.dot((x - ck).T,ak)
    dist = x-ck-np.dot(alpha, ak)
    return math.pow(np.linalg.norm(dist),2)

def planeDistance(x, ck, ak, bk):
    alpha = np.dot((x - ck).T,ak)
    beta = np.dot((x - ck).T,bk)
    dist = x-ck-(np.dot(alpha, ak))-(np.dot(beta, bk))
    return math.pow(np.linalg.norm(dist),2)

def centroid(Xk):
    # Xk is datapoints in Ck (equation 7)
    size = len(Xk)
    total = sum(Xk)
    return float(total)/float(size)

def firstDirection(Xk):
    # do PCA and return first direction

def secondDirection(Xk):
    # do PCA and return second direction

def sphereDistance(x, ck, eta, variance):
    # eta should be between 0.2 and 0.5
    v = x-ck
    dist = math.pow(np.linalg.norm(v),2)
    final = dist-(eta*variance)
    return max(0, final)

def findVariance(Xk, ck):
    total = 0
    for point in Xk:
        dist = point - ck
        total += math.pow(np.linalg.norm(dist),2)
    return total

def chooseModel(Xk):
    eta = 0.2
    ck = centroid(Xk)
    ak = firstDirection(Xk)
    bk = secondDirection(Xk)
    variance = findVariance(Xk, ck)
    lineDist = sum(lineDistance(x, ck, bk) for x in Xk)
    planeDist = sum(planeDistance(x, ck, ak, bk) for x in Xk)
    sphereDist = sum(sphereDistance(x, ck, eta, variance) for x in Xk)

    minimum = np.argmin([lineDist, planDist, sphereDist])
    if minimum is 1:
        lineCluster(Xk)
    else if minimum is 2:
        planeCluster(Xk)
    else:
        sphereCluster(Xk)

def clusterAssignment(X):
    # X is float64 array-like
    # E-step of EM
    # estimating the posterior probability for all data points belonging to different clusters
    # should return labels and inertia
        
    # set centers to -1 
    n_samples = X.shape[0]
    clusterLabels = -np.ones(n_samples, np.int32)
    if distances is None:
        distances = np.zeros(shape=(0,), dtype=X.dtype)
    inertia = assign_labels(X, x_squared_norms, centers, labels, distances=distances)



def modelEstimation():
    # M-step of EM

