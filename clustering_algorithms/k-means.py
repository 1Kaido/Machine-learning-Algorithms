import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=500,n_features=2,centers=3,cluster_std=1.5,random_state=42)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Dataset (3 Clusters)")
plt.show()

print(X.shape)

#Step 1 random n_clustors
n_clusters = 3
indices = np.random.choice(len(X),n_clusters)
centroids = X[indices]
#print(centroids)
#Distances
cluster_group = []
for row in X:
    distances = []
    for centroid in centroids:
        distance = np.linalg.norm(row - centroid)
        distances.append(distance)
        #print(centroid,row,distance)
    min_distance = np.argmin(distances)
    cluster_group.append(min_distance)
    #print(distances,min_distance)
print(np.array(cluster_group).shape)
clusters = np.unique(cluster_group)

#now we got a cluster_group which tell us in
#in which centroid each row belongs
print(cluster_group)
