
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


NOTVISITED = -2
NOISE = -1

class DBSCAN:
    minPoints = 0
    eps = 0
    calcEps = True

    def __init__(self, calcEps = True, eps=0, min_points=4):
        self.eps = eps
        self.calcEps = calcEps
        self.min_points = min_points

    def dist(self, point_a, point_b):
        sum = 0
        for i in xrange(len(point_a)):
            sum += (math.pow(point_a[i]-point_b[i], 2))
        return math.sqrt(sum)

    def regionQuery(self, dataMatrix, pointIndex, eps):
        neighbors = []
        for i in xrange(len(dataMatrix)):
            if (self.dist(dataMatrix[i], dataMatrix[pointIndex]) <= eps):
                neighbors.append(i)
        neighbors.remove(pointIndex)
        return neighbors

    def expandCluster(self, dataMatrix, neighbors, eps, minPoints, clusters, pointIndex, clusterIndex):
        clusters[pointIndex] = clusterIndex
        for neighbors_iter in neighbors:
            if(clusters[neighbors_iter] < 0):
                clusters[neighbors_iter] = clusterIndex
                neighbors_2 = self.regionQuery(dataMatrix, neighbors_iter, eps)
                if(len(neighbors_2) > minPoints):
                    neighbors.extend(neighbors_2)

    def run_dbscan(self, dataMatrix, eps, minPoints):
        clusterIndex = 0
        clusters = [NOTVISITED] * len(dataMatrix)

        for i in xrange(len(dataMatrix)):
            if clusters[i] >= 0:
                continue
            neighbors = self.regionQuery(dataMatrix, i, eps)
            if(len(neighbors) < minPoints):
                for i in neighbors:
                    clusters[i] = NOISE
                continue
            else:
                self.expandCluster(dataMatrix, neighbors, eps, minPoints, clusters, i, clusterIndex)
                clusterIndex+=1
        return clusters

    def getMask(self, indexVal, array):
        mask = [False] * len(array)
        index = 0
        for i in array:
            if(indexVal==i):
                mask[index] = True
        return mask

    def calculateEps(self, dataMatrix):
        neigh = NearestNeighbors(3, 1)
        neigh.fit(dataMatrix)
        kNearestDist =  (neigh.kneighbors(dataMatrix, 3, return_distance=True)[0]).flatten()
        sortedDistance = sorted(kNearestDist[kNearestDist != 0])
        return sortedDistance[len(sortedDistance)*3/4]

    def run(self,dataMatrix, title="", show=False): #need to figure out how to automate this
        title = 'DBSCAN Clustering ' + title
        print title
        if(self.calcEps):
            self.eps = self.calculateEps(dataMatrix)
        start = datetime.datetime.now()
        clusters = self.run_dbscan(dataMatrix, self.eps, self.minPoints)
        end = datetime.datetime.now()
        print "start, end, end-start", start, end, end-start
        with open("timing", "a") as target:
            target = open("timing", 'a')
            newline = "\n"
            target.write(title + newline)
            target.write(str(start)+ newline)
            target.write(str(end)+ newline)
            target.write(str(end - start)+ newline)

        unique_cluster_id = set(clusters)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_cluster_id)))

        colorCombo = dict(zip(unique_cluster_id, colors))
        for i in xrange(len(clusters)):
            cluster_ind = clusters[i]
            color = colorCombo[cluster_ind]
            if cluster_ind == NOISE:
                color = 'black'

            plt.plot(dataMatrix[i][0], dataMatrix[i][1], 'o', markerfacecolor=color,
                     markeredgecolor='k', markersize=6)

        plt.title(title)
        plt.savefig("figure/" + title)
        plt.clf()
        plt.close('all')
        if (not show):
            return clusters

        plt.show()
        return clusters

    def scikitDbScan(self, X, eps, min_points, title="", show=False):
        print "scikitDbScan"

        start = datetime.datetime.now()

        print start
        X = StandardScaler().fit_transform(X)
        db = DBSCAN(eps=eps, min_samples=min_points).fit(X)
        end = datetime.datetime.now()
        print "start, end, end-start", start, end, end - start

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
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=14)

            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=6)

        title = 'DBSCAN Clustering: ' + title
        plt.title(title)
        plt.savefig("figure/" + title)
        if (not show):
            return
        plt.show()







