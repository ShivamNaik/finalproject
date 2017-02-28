import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
import matplotlib.pyplot as plt

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

def oneB():
    genomeMatrix, countryRegion, gender = LoadData()
    modeNucleobase = getModeNucleobases(genomeMatrix)
    X = createBinaryMatrix(genomeMatrix, modeNucleobase)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    dictColors, labelledList = createCountryRegionToColorDict(countryRegion)
    codeToLabelMapping = {"ACB": "African Caribbeans in Barbados", "GWD": "Gambian in Western Divisions",
                          "ESN": "Esan in Nigeria", "MSL": "Mende in Sierra Leone", "YRI": "Yoruba in Ibadan, Nigeria",
                          "LWK": "Luhya in Webuye, Kenya", "ASW": "Americans of African Ancestry in SW USA"}

    for i in xrange(995):
        currColor = dictColors[countryRegion[i]]
        if(labelledList[countryRegion[i]] == False):
            plt.scatter(X_pca[i, 0], X_pca[i, 1], color=currColor, s = 15, label=codeToLabelMapping[countryRegion[i]])
            labelledList[countryRegion[i]] = True
        else:
            plt.scatter(X_pca[i, 0], X_pca[i, 1], color=currColor, s = 15)

    plt.legend(loc=9, bbox_to_anchor=(1, .5))
    plt.savefig("oneb")
    plt.show()

#oneB()

def oneD():
    genomeMatrix, countryRegion, gender = LoadData()
    modeNucleobase = getModeNucleobases(genomeMatrix)
    X = createBinaryMatrix(genomeMatrix, modeNucleobase)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    genderLabel = ["", "Male", "Female"]
    genderLabelYet = [False, False, False]
    genderCombo = {1:"green", 2:"darkorange"}
    for i in xrange(995):
        currColor = color=genderCombo[gender[i]]
        if(genderLabelYet[gender[i]] == False):
            plt.scatter(X_pca[i, 0], X_pca[i, 2], color=currColor, s = 15, label=genderLabel[gender[i]])
            genderLabelYet[gender[i]] = True
        else:
            plt.scatter(X_pca[i, 0], X_pca[i, 2], color=currColor, s = 15)

    plt.legend(loc=9, bbox_to_anchor=(1, .5))
    plt.savefig("oneD")
    plt.show()

#oneD()

def oneF():
    genomeMatrix, countryRegion, gender = LoadData()
    modeNucleobase = getModeNucleobases(genomeMatrix)
    X = createBinaryMatrix(genomeMatrix, modeNucleobase)

    pca = PCA(n_components=3)
    pca.fit_transform(X)
    # get the third pca component
    third_pca = pca.components_[2]
    maxIndex =  np.argpartition(abs(third_pca), -4)[-4:]
    for i in maxIndex:
        print modeNucleobase[i], i, third_pca[i]

    for i in xrange(995):
        print modeNucleobase[i], X[i][10095], X[i][10096], X[i][10099]


#oneF()

# By inspecting the third principal component itself, identify the indices of the three nucleobases
# that are most closely associated with the clustering you observed in part (e). In one sentence,
# what does a deviation from the mode indicate at these indices?

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


def createXMatrix(genomeMatrix, weightedNucleoBase):
    m = len(genomeMatrix)  # number of individuals
    n = len(genomeMatrix[0])  # number of dimensions
    X = np.ones((m,n))
    for i in xrange(m):
        for j in xrange(n):
            X[i][j] = weightedNucleoBase[j].index(genomeMatrix[i][j])
    return X

def oneH():
    genomeMatrix, countryRegion, gender = LoadData()
    weightedNucleoBases = createNucleobaseWeights(genomeMatrix)
    X = createXMatrix(genomeMatrix,weightedNucleoBases)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    # dictColors, labelledList = createCountryRegionToColorDict(countryRegion)
    # codeToLabelMapping = {"ACB": "African Caribbeans in Barbados", "GWD": "Gambian in Western Divisions",
    #                       "ESN": "Esan in Nigeria", "MSL": "Mende in Sierra Leone", "YRI": "Yoruba in Ibadan, Nigeria",
    #                       "LWK": "Luhya in Webuye, Kenya", "ASW": "Americans of African Ancestry in SW USA"}
    #
    # for i in xrange(995):
    #     currColor = dictColors[countryRegion[i]]
    #     if(labelledList[countryRegion[i]] == False):
    #         plt.scatter(X_pca[i, 0], X_pca[i, 1], color=currColor, s = 15, label=codeToLabelMapping[countryRegion[i]])
    #         labelledList[countryRegion[i]] = True
    #     else:
    #         plt.scatter(X_pca[i, 0], X_pca[i, 1], color=currColor, s = 15)

    genderLabel = ["", "Male", "Female"]
    genderLabelYet = [False, False, False]
    genderCombo = {1: "green", 2: "darkorange"}
    for i in xrange(995):
        currColor = color = genderCombo[gender[i]]
        if (genderLabelYet[gender[i]] == False):
            plt.scatter(X_pca[i, 0], X_pca[i, 2], color=currColor, s=15, label=genderLabel[gender[i]])
            genderLabelYet[gender[i]] = True
        else:
            plt.scatter(X_pca[i, 0], X_pca[i, 2], color=currColor, s=15)

    plt.legend(loc=9, bbox_to_anchor=(1, .5))
    plt.savefig("oneH")
    plt.show()

oneH()


def oneI():
    genomeMatrix, countryRegion, gender = LoadData()
    modeNucleobase = getModeNucleobases(genomeMatrix)
    X = createBinaryMatrix(genomeMatrix, modeNucleobase)

    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X)

    dictColors, labelledList = createCountryRegionToColorDict(countryRegion)
    codeToLabelMapping = {"ACB": "African Caribbeans in Barbados", "GWD": "Gambian in Western Divisions",
                          "ESN": "Esan in Nigeria", "MSL": "Mende in Sierra Leone", "YRI": "Yoruba in Ibadan, Nigeria",
                          "LWK": "Luhya in Webuye, Kenya", "ASW": "Americans of African Ancestry in SW USA"}

    for i in xrange(995):
        currColor = dictColors[countryRegion[i]]
        if(labelledList[countryRegion[i]] == False):
            plt.scatter(X_pca[i, 0], X_pca[i, 3], color=currColor, s = 15, label=codeToLabelMapping[countryRegion[i]])
            labelledList[countryRegion[i]] = True
        else:
            plt.scatter(X_pca[i, 0], X_pca[i, 3], color=currColor, s = 15)

    plt.legend(loc=9, bbox_to_anchor=(1, .5))
    plt.savefig("onei")
    plt.show()
#oneI()