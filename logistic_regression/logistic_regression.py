class LogisticRegression:
    def __init__(self,epochs,learning_rate):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.coef_ = None
        self.intercept_ = None

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def fit(self,X_train,y_train):
        m,n = X_train.shape
        #print(X_train.shape)
        X_train = np.insert(X_train,0,1,axis=1)
        #print(X_train.shape)
        weights = np.ones(X_train.shape[1])

        for i in range(self.epochs):
            y_pred = self.sigmoid(np.dot(X_train,weights))
            error = y_pred - y_train

            weights_derivative = (-1/n)*np.dot(error,X_train)
            weights = weights - (self.learning_rate * weights_derivative)

            self.intercept_ = weights[0]
            self.coef_ = weights[1:]

    def predict(self,X_test):
        z = (X_test @ self.coef_) + self.intercept_
        y_pred = self.sigmoid(z)
        return np.where(y_pred >= 0.5, 1, 0)
