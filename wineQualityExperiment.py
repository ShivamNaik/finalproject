from dbscan import myDBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import time
# https://archive.ics.uci.edu/ml/datasets/Wine+Quality
def LoadData():
    f = open("winequality-white.csv")
    pokerhands = []
    i = 0
    for line in f:
        if(i == 0):
            i+=1 #skip the first line
            continue
        if(i > 20):
            break
        splitline = line.split(";")
        pokerhands.append(map(float, map(str.strip, splitline)))
        i +=1
    f.close()
    return pokerhands

def pokerHandExperiment():
    X = LoadData()
    eps = 2
    min_points = 2
    dbscanalgo = myDBSCAN()

    start = time.time()
    print start
    clusters = dbscanalgo.dbscan(X, eps, min_points)
    end = time.time()
    print "start, end", start, end

    X = StandardScaler().fit_transform(X)
    start = time.time()
    print start

    db = DBSCAN(eps=eps, min_samples=min_points).fit(X)
    end = time.time()
    print "start, end", start, end


    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    labels = db.labels_

    print "clusters", clusters
    print "labels", labels

    # unique_labels = set(labels)
    # colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    # for k, col in zip(unique_labels, colors):
    #     if k == -1:
    #         # Black used for noise.
    #         col = 'k'
    #
    #     class_member_mask = (labels == k)
    #     xy = X[class_member_mask & core_samples_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
    #              markeredgecolor='k', markersize=14)
    #
    #     xy = X[class_member_mask & ~core_samples_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
    #              markeredgecolor='k', markersize=6)
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    #
    # plt.title('Estimated number of clusters: %d' % n_clusters_)
    # plt.show()

pokerHandExperiment()
