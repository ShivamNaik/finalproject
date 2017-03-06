import numpy as np
import matplotlib.pyplot as plt
import math
import datetime

class HAC:
    clusterLevel = 1

    def calculateDistance(self, sample_a, sample_b):
        sum = 0
        for i in xrange(len(sample_a)):
            sum += (math.pow(sample_a[i]-sample_b[i], 2))
        return math.sqrt(sum)

    def mergeClusters(self, clusters, index_1, index_2):
        clusters[index_1].extend(clusters[index_2])
        del clusters[index_2]
        return clusters

    def runAggClustering(self, X, cluster_level = 1):
        clusters = [[index] for index in range(0, len(X))]
        while(len(clusters) > cluster_level):
            indicies = (-1, -1)
            minDistance = float("Inf")

            for i in xrange(len(clusters)):
                for j in xrange(i+1, len(clusters)):
                    if(i==j):
                        continue
                    for index_i in clusters[i]:
                        for index_j in clusters[j]:
                            distance = self.calculateDistance(X[index_i], X[index_j])
                            if(distance < minDistance):
                                minDistance = distance
                                indicies = (i,j)

            self.mergeClusters(clusters, indicies[0], indicies[1])
        return clusters

    def __init__(self, clusterLevel = 3):
        self.clusterLevel = clusterLevel

    def run(self, dataMatrix, title="", show=False):  # need to figure out how to automate this
        title = 'Agglomerative Clustering ' + title
        print title

        start = datetime.datetime.now()
        clusters = self.runAggClustering(dataMatrix, self.clusterLevel)
        end = datetime.datetime.now()
        print "start, end, end-start", start, end, end - start

        with open("timing.txt", "a") as target:
            target = open("timing", 'a')
            newline = "\n"
            target.write(title + newline)
            target.write(str(start)+ newline)
            target.write(str(end)+ newline)
            target.write(str(end - start)+ newline)

        colors = plt.cm.Spectral(np.linspace(0, 1, len(clusters)))
        colorCombo = dict(zip(xrange(len(clusters)), colors))
        for i in xrange(len(clusters)):
            color = colorCombo[i]

            for j in xrange(len(clusters[i])):
                cluster_ind = clusters[i][j]

                plt.plot(dataMatrix[cluster_ind][0], dataMatrix[cluster_ind][1], 'o', markerfacecolor=color,
                     markeredgecolor='k', markersize=6)
        plt.title(title)
        plt.savefig("figure/" + title)
        plt.clf()
        plt.close('all')
        if not show:
            return clusters
        plt.show()
        return clusters

    def testHAC(self):
        test = [[1, 1.1, 1], [1.2, .8, 1.1], [.8, 1, 1.2], [3.7, 3.5, 3.6], [3.9, 3.9, 3.5], [3.4, 3.5, 3.7],[15,15, 15]]
        self.plotAggClustering("test", test)
        self.plotAggClustering("test2", test, 2)
        self.plotAggClustering("test3", test, 3)