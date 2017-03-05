import numpy as np
import matplotlib.pyplot as plt
import math

def calculateDistance(sample_a, sample_b):
    sum = 0
    for i in xrange(len(sample_a)):
        sum += (math.pow(sample_a[i]-sample_b[i], 2))
    return math.sqrt(sum)

def mergeClusters(clusters, index_1, index_2):
    clusters[index_1].extend(clusters[index_2])
    del clusters[index_2]
    return clusters

def runAggClustering(X, cluster_level = 1):
    clusters = [[index] for index in range(0, len(X))]
    while(len(clusters) >= cluster_level):
        indicies = (-1, -1)
        minDistance = float("Inf")

        for i in xrange(len(clusters)):
            for j in xrange(i+1, len(clusters)):
                if(i==j):
                    continue
                for index_i in clusters[i]:
                    for index_j in clusters[j]:
                        distance = calculateDistance(X[index_i], X[index_j])
                        if(distance < minDistance):
                            minDistance = distance
                            indicies = (i,j)

        mergeClusters(clusters, indicies[0], indicies[1])
    return clusters

def plotAggClustering(title, dataMatrix, clusterLevel = 1):
    clusters = runAggClustering(dataMatrix, clusterLevel)
    unique_cluster_id = set(clusters)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_cluster_id)))
    # need to know how to color things
    colorCombo = dict(zip(unique_cluster_id, colors))
    for i in xrange(len(clusters)):
        for j in xrange(len(clusters[i])):
            cluster_ind = clusters[i][j]
            color = colorCombo[cluster_ind]

            plt.plot(dataMatrix[cluster_ind][0], dataMatrix[cluster_ind][1], 'o', markerfacecolor=color,
                 markeredgecolor='k', markersize=14)

    plt.title('Agglomerative Clustering: ' + title)
    plt.show()
    return clusters
