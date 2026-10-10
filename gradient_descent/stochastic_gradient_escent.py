class StochasticGradientDescent:
    def __init__(self,epochs, learning_rate):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.coef_ = None
        self.intercept_ = None

    def fit(self,X_train,y_train):
        self.intercept_ = 0
        self.coef_ = np.zeros(X_train.shape[1])
        m = X_train.shape[0]
        for i in range(self.epochs):
            idx = np.random.randint(0,X_train.shape[0])
            x_i = X_train[idx]
            y_i = y_train[idx]

            y_pred = x_i @ self.coef_ + self.intercept_
            error = y_pred - y_i

            coef_derivative =   (error) * (x_i)
            intercept_derivative =  (error)

            self.coef_ = self.coef_ - self.learning_rate * coef_derivative
            self.intercept_ = self.intercept_ - self.learning_rate * intercept_derivative

    def predict(self,X_test):
        return X_test @ self.coef_ + self.intercept_
