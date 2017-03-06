
from sklearn.decomposition import PCA, IncrementalPCA
from dbscan import DBSCAN
from hac import HAC
import numpy as np

class ExperimentData:
    genome = []
    wine = []
    poker = []
    dotaTest = []
    dataSets = []

    def LoadData(self, fileName, splitVal, dtype=float, limit=False, limitNum = 5000):
        f = open(fileName)
        data = []
        limitIter = 0
        for line in f:
            if(limit & (limitIter > limitNum)):
                break
            splitline = line.split(splitVal)
            data.append(map(float,map(str.strip, splitline)))
            limitIter+=1
        f.close()
        return data

    def LoadGenomeData(self, limit = False, limitNum = 5000):
        f = open("genome-data.txt")
        genomeMatrix = []
        labels = []
        gender = []
        limitIter = 0

        for line in f:
            if (limit & limitNum > limitIter):
                break
            splitline = line.split(" ")
            gender.append(int(splitline[1]))
            labels.append(splitline[2])
            genomeMatrix.append(map(str.strip, splitline[3:]))
            limitIter+=1

        f.close()
        return genomeMatrix

    def _createNucleobaseWeights(self, genomeMatrix):
        modeNucleobase = []
        for i in xrange(len(genomeMatrix[0])):
            dictionary = {"0": 0, "A": 0, "G": 0, "T": 0, "C": 0}
            for j in xrange(len(genomeMatrix)):
                if (genomeMatrix[j][i] != "0"):
                    if dictionary.has_key(genomeMatrix[j][i]):
                        dictionary[genomeMatrix[j][i]] += 1
                    else:
                        dictionary[genomeMatrix[j][i]] = 1
            nucleoList = sorted(dictionary, key=dictionary.get)
            nucleoList.reverse()
            modeNucleobase.append(nucleoList)
        return modeNucleobase

    def __init__(self, limit=False, limitNum=5000):
        self.limit = limit
        self.limitNum = limitNum
        self.genome = self.LoadGenomeData()
        self.wine = self.LoadData("winequality-white.csv", ";", limit=self.limit, limitNum=self.limitNum)
        self.poker = self.LoadData("poker-hand-testing.txt", ",", int, limit=self.limit, limitNum=self.limitNum)
        self.dotaTest = self.LoadData("dota2Test.csv", ",", limit=self.limit, limitNum=self.limitNum)
        self.arrhythmia = self.LoadData("arrhythmia.data", ",", limit=self.limit, limitNum=self.limitNum)

        self.dataSets = [self.wine, self.poker, self.dotaTest, self.arrhythmia]
        self.labels = ["Wine Quality Data Set", "Poker Hand Data Set", "Dota Data Set"]

class Experiments:
    def run(self, algorithm, limit=False, limitNum=5000, dimension=5):
        experimentData = ExperimentData(limit, limitNum)

        algorithm.run(experimentData.wine, "Wine Quality Data Set")
        pca = PCA(n_components=dimension)
        algorithm.run(pca.fit_transform(experimentData.wine), "Wine Quality Data Set PCA")

        algorithm.run(experimentData.poker, "Poker Data Set")
        pca = PCA(n_components=dimension)
        algorithm.run(pca.fit_transform(experimentData.poker), "Poker Data Set PCA")

        algorithm.run(experimentData.dotaTest, "Dota Test Data Set")
        pca = PCA(n_components=dimension)
        algorithm.run(pca.fit_transform(experimentData.dotaTest), "Dota Test Data Set PCA")

        print "done"

def test_DBSCAN():
    X = np.array([[1, 1.1, 1], [1.2, .8, 1.1], [.8, 1, 1.2], [3.7, 3.5, 3.6], [3.9, 3.9, 3.5], [3.4, 3.5, 3.7],[15,15, 15]])
    eps = 0.5
    min_points = 2
    dbscanalgo = DBSCAN(eps=eps, min_points=min_points)
    dbscanalgo.run(X, "Synthetic Data")

def teset_HAC():
    test = [[1, 1.1, 1], [1.2, .8, 1.1], [.8, 1, 1.2], [3.7, 3.5, 3.6], [3.9, 3.9, 3.5], [3.4, 3.5, 3.7],[15,15, 15]]
    hac = HAC()
    for i in xrange(3):
        hac.clusterLevel = i
        hac.run(test, "Synthetic Data with Cluster Level " + str(i))

test_DBSCAN()
teset_HAC()

dbscan = DBSCAN()
hac = HAC()
experiment = Experiments()

experiment.run(dbscan, True, 5000, 3)
experiment.run(hac, True, 5000, 3)

