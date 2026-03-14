import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA

def kmeans_cluster(data, k):
    model = KMeans(n_clusters=k)
    labels = model.fit_predict(data)
    return labels


def hierarchical_cluster(data, k):
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(data)
    return labels


def dbscan_cluster(data):
    model = DBSCAN()
    labels = model.fit_predict(data)
    return labels


def apply_pca(data):
    pca = PCA(n_components=2)
    return pca.fit_transform(data)
