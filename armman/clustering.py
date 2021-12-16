import numpy as np
from sklearn.cluster import KMeans

def kmeans_missing(X, n_clusters, max_iter=10,random_state=0):
    missing = ~np.isfinite(X)
    mu = np.nanmean(X, 0, keepdims=1)
    X_hat = np.where(missing, mu, X)

    prev_labels = None
    for i in range(max_iter):
        cls = KMeans(n_clusters, n_jobs=-1, random_state=random_state)


        labels = cls.fit_predict(X_hat)

        centroids = cls.cluster_centers_


        X_hat[missing] = centroids[labels][missing]

        if i > 0 and np.all(labels == prev_labels):
            break

        prev_labels = labels

    return labels, centroids, X_hat, cls, len(set(labels)), i
