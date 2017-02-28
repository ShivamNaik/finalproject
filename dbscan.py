
import numpy as np
import matplotlib.pyplot as plt
import math

NOTVISITED = -1
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
                if(len(neighbors_2) > minPoints):
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
    m = [[1, 1.1], [1.2, .8], [.8, 1], [3.7, 3], [3.9, 4.9], [3.6, 4.1],[10,10]]
    eps = 0.5
    min_points = 2
    dbscanalgo = dbscan()
    print dbscanalgo.dbscan(m, eps, min_points)
    assert dbscanalgo.dbscan(m, eps, min_points) == [0, 0, 0, 1, 1, 1, NOISE]
test_dbscan()

