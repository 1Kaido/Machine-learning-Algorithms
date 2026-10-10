class MultipleLinearRegression:
    def __init__(self):
        self.beta = None
    def fit(self, X_train, y_train):
        ones = np.ones((X_train.shape[0], 1))
        X_B = np.hstack((ones, X_train))
        self.beta = (
            np.linalg.inv(X_B.T @ X_B) @ X_B.T @ y_train)
    def predict(self, X_test):
        ones = np.ones((X_test.shape[0], 1))
        X_B = np.hstack((ones, X_test))
        return X_B @ self.beta
