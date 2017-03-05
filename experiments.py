
from sklearn.decomposition import PCA, IncrementalPCA
from dbscan import DBSCAN
from hac import HAC

class ExperimentData:
    genome = []
    wine = []
    poker = []
    dotaTest = []
    dorothea = []
    dataSets = []

    def LoadData(self, fileName, splitVal, dtype=float, limit=False, limitNum = 5000):
        f = open(fileName)
        data = []
        limitIter = 0
        for line in f:
            if(limit & (limitIter > limitNum)):
                break
            splitline = line.split(splitVal)
            data.append(map(float,(map(lambda x: dtype(0) if x is '' else dtype(x), map(str.strip, splitline)))))
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
        self.dorothea = self.LoadData("dorothea_valid.txt", " ", int, limit=self.limit, limitNum=self.limitNum)
        #   self.dataSets = [self.wine, self.poker, self.dotaTest, self.dorothea]
        self.dataSets = [self.wine, self.poker, self.dotaTest]
        self.labels = ["Wine Quality Data Set", "Poker Hand Data Set", "Dota Data Set", "Dorothea Data Set"]

class Experiments:
    def run(self, algorithm, limit=False, limitNum=5000, dimension=5):
        experimentData = ExperimentData(limit, limitNum)

        for i in xrange(len(experimentData.dataSets)):
            print experimentData.labels[i]
            print "Not PCA"

            algorithm.run(experimentData.dataSets[i], experimentData.labels[i])
            print "PCA"
            pca = PCA(n_components=dimension)
            algorithm.run(pca.fit_transform(experimentData.dataSets[i]), experimentData.labels[i])

dbscan = DBSCAN()
hac = HAC()
experiment = Experiments()
experiment.run(dbscan, True, 100, 2)
experiment.run(hac, True, 100, 2)
