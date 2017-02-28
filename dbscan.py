from sklearn.cluster import DBSCAN

import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


NOTVISITED = -2
NOISE = -1

class dbscan:

    def dist(self, point_a, point_b):
        sum = 0
        for i in xrange(len(point_a)):
            sum += (math.pow(point_a[i]-point_b[i], 2))
        return math.sqrt(sum)

    def regionQuery(self, dataMatrix, pointIndex, eps):
        neighbors = []
        for i in xrange(len(dataMatrix)):
            if (self.dist(dataMatrix[i], dataMatrix[pointIndex]) < eps):
                neighbors.append(i)
        return neighbors

    def expandCluster(self, dataMatrix, eps, minPoints, clusters, pointIndex, clusterIndex):
        clusters[pointIndex] = clusterIndex
        neighbors = self.regionQuery(dataMatrix, pointIndex, eps)
        for neighbors_iter in neighbors:
            if(clusters[neighbors_iter] < 0): #if its not visited or noise
                clusters[neighbors_iter] = clusterIndex
                neighbors_2 = self.regionQuery(dataMatrix, neighbors_iter, eps)
                if(len(neighbors_2) >= minPoints):
                    neighbors.extend(neighbors_2)


    def dbscan(self, dataMatrix, eps, minPoints):
        clusterIndex = 0
        clusters = [NOTVISITED] * len(dataMatrix)

        for i in xrange(len(dataMatrix)):
            if clusters[i] >= 0:
                continue
            neighbors = self.regionQuery(dataMatrix, i, eps)
            if(len(neighbors) < minPoints):
                clusters[i] = NOISE
            else:
                self.expandCluster(dataMatrix, eps, minPoints, clusters, i, clusterIndex)
            clusterIndex+=1
        return clusters




def test_dbscan():
    X = [[1, 1.1, 1], [1.2, .8, 1.1], [.8, 1, 1.2], [3.7, 3.5, 3.6], [3.9, 3.9, 3.5], [3.4, 3.5, 3.7],[15,15, 15]]
    eps = 0.5
    min_points = 2
    dbscanalgo = dbscan()
    dbscanalgo.dbscan(X, eps, min_points)

    X = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_points).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    labels = db.labels_
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)
        xy = X[class_member_mask & core_samples_mask]
        print xy
        print " "
        print xy[:, 0], xy[:, 1]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

test_dbscan()

