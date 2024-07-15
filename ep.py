import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


data = pd.read_csv("Combined Dataset (Jul'22 - Jun'23).csv")
#data = pd.read_csv("combined_datasets(2020-2021).csv")
#data = pd.read_csv("Combined_datasets(2018-2019).csv")



X = data[['ColumnO3', 'Month']]


min_samples_range = range(2, 11)
distances = []

for min_samples in min_samples_range:
    neigh = NearestNeighbors(n_neighbors=min_samples)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    distances = np.sort(distances, axis=0)[:, -1]


    sorted_distances = np.sort(distances)
    plt.plot(range(1, len(X) + 1), sorted_distances)

plt.xlabel('Data Point Index')
plt.ylabel('Epsilon')
plt.title('K-Distance Graph for Min_Samples Range')
plt.legend([f'Min_Samples = {min_samples}' for min_samples in min_samples_range])
plt.show()




optimal_min_samples = 4
optimal_epsilon = 0.3


dbscan = DBSCAN(eps=optimal_epsilon, min_samples=optimal_min_samples)
labels = dbscan.fit_predict(X)


'''plt.scatter(X['ColumnO3'], X['Month'], c=labels, cmap='viridis')
plt.xlabel('ColumnO3')
plt.ylabel('Month')
plt.title('Agglomerative Clustering')
plt.show()'''
