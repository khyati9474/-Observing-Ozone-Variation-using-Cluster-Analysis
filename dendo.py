import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Reading the dataset
data = pd.read_csv("Combined Dataset (Jul'22 - Jun'23).csv")
#data = pd.read_csv("combined_datasets(2020-2021).csv")
#data = pd.read_csv("Combined_datasets(2018-2019).csv")


# Selecting features for clustering (ozone concentration and month)
X = data[['ColumnO3', 'Month']]

# Setting parameters for agglomerative clustering
n_clusters = 3
linkage_method = 'ward'
agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)


# Performing agglomerative clustering and obtaining cluster labels
cluster_labels = agglomerative.fit_predict(X)

# Creating linkage matrix for dendrogram
linkage_matrix = linkage(X, method=linkage_method)

# Generating dendrogram
dendrogram(linkage_matrix, orientation='top', labels=cluster_labels, distance_sort='descending', show_leaf_counts=True)

# Adding labels and title
plt.xlabel('Sample Index')
plt.ylabel('Cluster Distance')
plt.title('Dendrogram for Agglomerative Clustering(22-23)')

# Displaying the dendrogram
plt.show()
