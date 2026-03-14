import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from clustering import *

st.set_page_config(page_title="Customer Segmentation", layout="wide")

def load_css():
    with open("assets/style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

st.title("Customer Segmentation Dashboard")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.write(df.head())

    features = st.multiselect(
        "Select Features",
        df.columns,
        default=["Age", "AnnualIncome", "SpendingScore"]
    )

    data = df[features]

    algo = st.selectbox(
        "Select Algorithm",
        ["KMeans", "Hierarchical", "DBSCAN"]
    )

    k = st.slider("Clusters", 2, 6, 3)

    if st.button("Run Clustering"):

        if algo == "KMeans":
            labels = kmeans_cluster(data, k)

        elif algo == "Hierarchical":
            labels = hierarchical_cluster(data, k)

        else:
            labels = dbscan_cluster(data)

        df["Cluster"] = labels

        st.success("Done")

        st.write(df)

        # PCA
        pca_data = apply_pca(data)

        plt.scatter(
            pca_data[:,0],
            pca_data[:,1],
            c=labels
        )

        st.pyplot(plt)

        # stats
        st.write(df.groupby("Cluster").mean())

        csv = df.to_csv(index=False)

        st.download_button(
            "Download CSV",
            csv,
            "clustered.csv"
        )
