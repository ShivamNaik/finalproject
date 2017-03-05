from dbscan import myDBSCAN

# https://archive.ics.uci.edu/ml/datasets/Wine+Quality
def LoadData():
    f = open("winequality-white.csv")
    winequality = []
    i = 0
    for line in f:
        if(i == 0):
            i+=1 #skip the first line
            continue
        splitline = line.split(";")
        winequality.append(map(float, map(str.strip, splitline)))
        i +=1
    f.close()
    return winequality

def winequalityExperiment():
    X = LoadData()
    eps = 10
    min_points = 2
    dbscanalgo = myDBSCAN()

    clusters = dbscanalgo.plot_dbscan(X, eps, min_points)


winequalityExperiment()
