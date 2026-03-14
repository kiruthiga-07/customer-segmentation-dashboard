import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA


def convert_to_numeric(df):
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes

    return df


def kmeans_cluster(data, k):
    data = convert_to_numeric(data)

    model = KMeans(n_clusters=k, n_init=10)
    labels = model.fit_predict(data)

    return labels


def hierarchical_cluster(data, k):
    data = convert_to_numeric(data)

    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(data)

    return labels


def dbscan_cluster(data):
    data = convert_to_numeric(data)

    model = DBSCAN()
    labels = model.fit_predict(data)

    return labels


def apply_pca(data):
    data = convert_to_numeric(data)

    pca = PCA(n_components=2)
    return pca.fit_transform(data)
