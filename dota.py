from dbscan import myDBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import time
# https://archive.ics.uci.edu/ml/datasets/Dota2+Games+Results
def LoadData():
    f = open("dota2Train.txt")
    dota2 = []
    for line in f:
        splitline = line.split(",")
        dota2.append(map(int, map(str.strip, splitline)))
    f.close()
    return dota2

def LoadData():
    f = open("dota2Test.txt")
    dota2 = []
    for line in f:
        splitline = line.split(",")
        dota2.append(map(int, map(str.strip, splitline)))
    f.close()
    return dota2