class SoftMarginSVM:
    def __init__(self, learning_rate=0.001,epochs=10,c=1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coef_ = None
        self.intercept_ = None
        self.c = c

    def fit(self,X_train,y_train):
        self.coef_ = np.zeros(X_train.shape[1])
        self.intercept_ = 0


        for i in range(self.epochs):
            indices = np.random.permutation(X_train.shape[0])
            for idx in indices:
                x_i = X_train[idx]
                y_i = y_train[idx]

                if y_i * (self.coef_ @ x_i + self.intercept_) >= 1: # No Violation
                    d_w = self.coef_
                    d_b = 0 #cuz it 0 so line is stady
                    self.coef_ = self.coef_ - (self.learning_rate*d_w)
                    self.intercept_ = self.intercept_ - (self.learning_rate * d_b)
                else: # violation
                    d_w = self.coef_ -(self.c * y_i * x_i)
                    d_b = -(self.c*y_i)
                    self.coef_ = self.coef_ - (self.learning_rate*d_w)
                    self.intercept_ = self.intercept_ - (self.learning_rate * d_b)

    def predict(self,X_test):
        raw_preds = np.dot(X_test, self.coef_) + self.intercept_
        return np.where(raw_preds >= 0, 1, -1)
