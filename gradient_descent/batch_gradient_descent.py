class BatchGradientDecent:
    def __init__(self,epochs, learning_rate):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.intercept_ = None
        self.coef_ = None

    def fit(self,X_train,y_train):
        self.intercept_ = 0
        self.coef_ = np.zeros(X_train.shape[1])
        m = X_train.shape[0]
        for i in range(self.epochs):
            y_pred = X_train @ self.coef_ + self.intercept_
            error = y_pred - y_train
            intercept_derivative =  np.mean(error)
            coef_derivative = (1/m) * (X_train.T @ error)
            self.coef_ -= self.learning_rate * coef_derivative
            self.intercept_ -= self.learning_rate * intercept_derivative
            if i % 100 == 0:
                print(error)
    def predict(self,X_test):
        return (X_test @ self.coef_) + self.intercept_
