class KNN:
    def __init__(self,n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self,X_test):

        X_train = self.X_train.to_numpy()
        y_train = self.y_train.to_numpy()
        X_test = X_test.to_numpy()
        predictions = []

        for X_test_val in X_test:
            distances = []
            for X_train_val in X_train:
                distance = np.sqrt(np.sum((X_test_val - X_train_val)**2))
                distances.append(distance)

            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            nearest_classes = y_train[nearest_indices]
            values, counts = np.unique(nearest_classes, return_counts=True)
            prediction = np.argmax(counts)
            #print(nearest_classes)
            #print(values,counts,prediction)
            #print(values[prediction])

            predictions.append(values[prediction])
        #print(predictions)
        return predictions
