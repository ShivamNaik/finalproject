import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

def plotClusterData(X, clusterLabels, n_clusters):
    '''

    Usage Example:
    
    from plottingtools import *
    dat = syntheticData2Lines()
    ksub = KSubspaces(n_clusters=2).fit(dat)
    plotClusterData(dat, ksub.labels_, 2)
    
    '''
    
    colors = iter(cm.rainbow(np.linspace(0, 1, n_clusters)))
    clusters = [[] for i in range(n_clusters)]
    n_samples = X.shape[0]
    for i in range(n_samples):
        point = X[i]
        label = clusterLabels[i]
        clusters[label].append(point)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    for j in range(n_clusters):
        # want different color/marking for each cluster
        xs = []
        ys = []
        zs = []
        for pt in clusters[j]:
            xs.append(pt[0])
            ys.append(pt[1])
            zs.append(pt[2])
        ax.scatter(xs, ys, zs, c=next(colors))
    plt.show()

def syntheticData2Lines():
    points1 = []
    start1 = [15, 5, 1]
    for i in range(50):
        idx = 1
        start1[idx] += 1
        points1.append([start1[0], start1[1], start1[2]])
        
    points2 = []
    start2 = [5, 0, 0]
    for i in range(50):
        start2[0] += 0.5
        start2[1] += 0.5
        start2[2] -= 0.1
        points2.append([start2[0], start2[1], start2[2]])

    l = points1
    l.extend(points2)
    return np.array(l)

def syntheticDataLineSphere():
    line = syntheticDataLine()
    sphere = syntheticDataSphere()
    dataset = [line, sphere]
    plotDataList(dataset)

def syntheticDataLine():
    points2 = []
    start2 = [-50, -50, -80]
    for i in range(50):
        start2[0] += 7
        start2[1] += 7
        start2[2] += 1
        points2.append([start2[0], start2[1], start2[2]])
    return np.array(points2)

def syntheticDataSphere():
    z = (2 * np.random.rand(400) - 1)
    t = 2 * np.pi * np.random.rand(400)
    x = 40*(np.sqrt(1 - z**2) * np.cos(t))
    y = 40*(np.sqrt(1 - z**2) * np.sin(t))
    sphere = zip(x, y, 40*z)
    return np.array(sphere)

def syntheticPlane():
    points = []
    start = [-10, -10, 40]
    for i in range(400):
        point = list(start)
        point[0] += 240*np.random.rand()
        point[1] += 240*np.random.rand()
        point[2] = 0.8*point[0]
        #points.append([point[1], point[0], point[2]])
        points.append([point[0], point[1], point[2]])
    return np.array(points)

def plotDataList(dataset):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    n_clusters = len(dataset)
    colors = iter(cm.rainbow(np.linspace(0, 1, n_clusters)))
    for data in dataset:
        x = data[:,0]
        y = data[:,1]
        z = data[:,2]
        ax.scatter(x,y,z)
    ax.set_aspect("equal")
    plt.show()

def lineSphere():
    line = syntheticDataLine()
    sphere = syntheticDataSphere()
    plotDataList([line, sphere])

def linePlane():
    line = syntheticDataLine()
    plane = syntheticPlane()
    plotDataList([line, plane])

def linePlaneSphere():
    line = syntheticDataLine()
    sphere = syntheticDataSphere()
    plane = syntheticPlane()
    plotDataList([line, plane, sphere])
