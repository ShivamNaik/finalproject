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
        if(i > 50):
            break
        splitline = line.split(";")
        pokerhands.append(map(float, map(str.strip, splitline)))
        i +=1
    f.close()
    return pokerhands

def winequalityExperiment():
    X = LoadData()
    eps = 10
    min_points = 2
    dbscanalgo = myDBSCAN()

    start = time.time()
    print start
    clusters = dbscanalgo.plot_dbscan(X, eps, min_points)
    end = time.time()
    print "start, end", start, end



winequalityExperiment()
