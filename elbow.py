import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("Combined Dataset (Jul'22 - Jun'23).csv")
#data = pd.read_csv("combined_datasets(2020-2021).csv")
#data = pd.read_csv("Combined_datasets(2018-2019).csv")



X = data[['ColumnO3', 'Month']]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


k_values = range(1, 11)
inertia = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.title('Elbow Method for Optimal k')
plt.show()