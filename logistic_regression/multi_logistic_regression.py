#Multi Class Logisic Regression OvR method

class multiLogisticsRegression:
    def __init__(self,epochs,learning_rate):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.coef_ = None
        self.intercept_ = None

    def softmax(self,z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self,X_train,y_train):
        m,n = X_train.shape
        encoder = OneHotEncoder(sparse_output=False)
        y_train = y_train.reshape(-1, 1)
        y_onehotencoded = encoder.fit_transform(y_train)
        X_train = np.insert(X_train, 0, 1, axis=1)
        rows = X_train.shape[1]
        num_classes = y_onehotencoded.shape[1]
        weights = np.random.randn(rows, num_classes) * 0.01

        for i in range(self.epochs):
           # print(X_train.shape)
            #print(weights.shape)
            z = X_train @ weights

            y_pred = self.softmax(z)

            loss = y_pred - y_onehotencoded
            gradient = (1/m) * (X_train.T @ loss)
            weights = weights - (self.learning_rate * gradient)

        self.intercept_ = weights[0]
        self.coef_ = weights[1:]

    def predict(self, X_test):
        z = (X_test @ self.coef_) + self.intercept_
        y_hat = self.softmax(z)
        return np.argmax(y_hat, axis=1)
