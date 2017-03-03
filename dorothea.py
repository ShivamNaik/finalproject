from dbscan import myDBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import time
# http://archive.ics.uci.edu/ml/datasets/Dorothea
def LoadData():
    f = open("dorothea_valid.txt")
    dorothea = []
    for line in f:
        splitline = line.split(" ")
        dorothea.append(map(int, map(str.strip, splitline)))
    f.close()
    return dorothea