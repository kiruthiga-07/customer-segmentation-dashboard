import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from clustering import *
import os

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("Customer Segmentation Dashboard")

# -------- Load default dataset --------

default_path = os.path.join("data", "customers.csv")

file = st.file_uploader("Upload CSV (optional)", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

elif os.path.exists(default_path):
    df = pd.read_csv(default_path)
    st.info("Using default dataset")

else:
    st.warning("Upload a CSV file to continue")
    st.stop()


# -------- Show data --------

st.subheader("Dataset")
st.write(df.head())


# -------- Feature selection --------

features = st.multiselect(
    "Select Features",
    df.columns,
    default=list(df.columns)
)

if len(features) == 0:
    st.error("Select at least one column")
    st.stop()

data = df[features]


# -------- Algorithm --------

algo = st.selectbox(
    "Algorithm",
    ["KMeans", "Hierarchical", "DBSCAN"]
)

k = st.slider("Clusters", 2, 6, 3)


# -------- Run --------

if st.button("Run Clustering"):

    if algo == "KMeans":
        labels = kmeans_cluster(data, k)

    elif algo == "Hierarchical":
        labels = hierarchical_cluster(data, k)

    else:
        labels = dbscan_cluster(data)

    df["Cluster"] = labels

    st.success("Clustering Done")

    st.write(df)

    # PCA

    pca_data = apply_pca(data)

    plt.figure()

    plt.scatter(
        pca_data[:, 0],
        pca_data[:, 1],
        c=labels
    )

    st.pyplot(plt)

    # Stats

    st.subheader("Cluster Statistics")

    numeric_df = df.select_dtypes(include=["int64", "float64"])

st.write(
    numeric_df.groupby(df["Cluster"]).mean()
)

    # Download

    csv = df.to_csv(index=False)

    st.download_button(
        "Download Result",
        csv,
        "clustered.csv"
    )
