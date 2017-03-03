import numpy as np

def getModeNucleobases(genomeMatrix):
    modeNucleobase = []
    for i in xrange(len(genomeMatrix[0])):
        dictionary = {}
        for j in xrange(len(genomeMatrix)):
            if(genomeMatrix[j][i] != "0"):
                if dictionary.has_key(genomeMatrix[j][i]):
                    dictionary[genomeMatrix[j][i]] += 1
                else:
                    dictionary[genomeMatrix[j][i]] = 1
        modeNucleobase.append(max(dictionary.iterkeys(), key=lambda k: dictionary[k]))
    return modeNucleobase

def LoadData():
    f = open("genome-data.txt")
    genomeMatrix = []
    labels = []
    gender = []
    for line in f:
        splitline = line.split(" ")
        gender.append(int(splitline[1]))
        labels.append(splitline[2])
        genomeMatrix.append(map(str.strip, splitline[3:]))
    f.close()
    return genomeMatrix, labels, gender

def createBinaryMatrix(genomeMatrix, modeNucleobase):
    m = len(genomeMatrix)  # number of individuals
    n = len(genomeMatrix[0])  # number of dimensions
    X = np.ones((m,n))
    for i in xrange(m):
        for j in xrange(n):
            if(genomeMatrix[i][j] == modeNucleobase[j]):
                X[i][j] = 0
    return X

def createCountryRegionToColorDict(countryRegion):
    colors = ['navy', 'turquoise', 'darkorange', 'purple', 'green', 'yellow', 'magenta']
    countryRegionList = list(set(countryRegion))
    colorCombo = {}
    labelledList = {}
    for i in xrange(7):
        colorCombo[countryRegionList[i]] = colors[i]
        labelledList[countryRegionList[i]] = False
    return colorCombo, labelledList


def createNucleobaseWeights(genomeMatrix):
    modeNucleobase = []
    for i in xrange(len(genomeMatrix[0])):
        dictionary = {"0":0, "A":0, "G":0,"T":0,"C":0}
        for j in xrange(len(genomeMatrix)):
            if(genomeMatrix[j][i] != "0"):
                if dictionary.has_key(genomeMatrix[j][i]):
                    dictionary[genomeMatrix[j][i]] += 1
                else:
                    dictionary[genomeMatrix[j][i]] = 1
        nucleoList = sorted(dictionary, key=dictionary.get)
        nucleoList.reverse()
        modeNucleobase.append(nucleoList)
    return modeNucleobase
