def score(X, centers, labels):
    n_samples = X.shape[0]
    n_clusters = centers.shape[0]
    xsquarednorms = squarednorms(X)
    norms = xsquarednorms.reshape(-1,1)
    ysquarednorms = squarednorms(centers)
    eucs = euclidean_distances(X, centers, Y_norm_squared=ysquarednorms, X_norm_squared=norms, squared=True)

    totalInertia = 0.0
    for i in range(n_samples):
        label = labels[i]
        inertia = eucs[i][label]
        totalInertia += intertia
        
    return (totalInertia / n_samples)

def squarednorms(X):
    return (X**2).sum(axis=1)
