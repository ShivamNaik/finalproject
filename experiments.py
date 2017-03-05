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
            data.append(map(lambda x: dtype(0) if x is '' else dtype(x), map(str.strip, splitline)))
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
        self.dataSets = [self.wine, self.poker, self.dotaTest, self.dorothea]
