class Kmeans:
    def __init__(self,n_clusters,iterations=100):
        self.n_clusters = n_clusters
        self.iteration = iterations
    def fit(self,X_train):
        indices = np.random.choice(len(X_train),self.n_clusters)
        centroids = X_train[indices]

        for iteration in range(10):
            old_centroids = centroids.copy()
            new_centroids = []
            cluster_group = []
            for row in X_train:
                distances = []
                for centroid in centroids:
                    distance = np.linalg.norm(row - centroid)
                    distances.append(distance)
                    #print(centroid,row,distance)
                labels = np.argmin(distances)
                cluster_group.append(labels)
            cluster_group = np.array(cluster_group)
            clusters = np.unique(cluster_group)
            for cluster in clusters:
                points = X_train[cluster_group == cluster] #AI for masking
                centroid = np.mean(points,axis=0)
                new_centroids.append(centroid)


            if np.allclose(old_centroids, new_centroids):
                print(iteration,"Converged!")
                break
            centroids = new_centroids.copy()
            self.centroids = centroids
            self.labels = labels

    def predict(self,X_test): #AI help lil confuse how
        predictions = []
        for row in X_test:
            distances = []
            for centroid in self.centroids:
                distance = np.linalg.norm(row - centroid)
                distances.append(distance)
            labels = np.argmin(distances)
            predictions.append(labels)
        return predictions
