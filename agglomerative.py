import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

#data = pd.read_csv("Combined Dataset (Jul'22 - Jun'23).csv")
#data = pd.read_csv("combined_datasets(2020-2021).csv")
data = pd.read_csv("Combined_datasets(2018-2019).csv")

X = data[['ColumnO3', 'Month']]
n_clusters = 3
linkage = 'ward'
agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
cluster_labels = agglomerative.fit_predict(X)

print(f"Number of clusters: {max(agglomerative.labels_) + 1}")
plt.scatter(X['ColumnO3'], X['Month'], c=cluster_labels, cmap='viridis')
plt.xlabel('ColumnO3')
plt.ylabel('Month')
plt.title('Agglomerative Clustering(22-23)')
plt.show()
