#Ridge Regulator in Mini Batch Gradient Desent
class MiniBatchGradientDescent:
    def __init__(self,epochs=100,learning_rate=0.001,batch_size=32,alpha=0.1):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.alpha = alpha
        self.intercept_ = None
        self.coef_ = None
        self.loss = []
    def fit(self,X_train,y_train):
        m,n = X_train.shape
        self.intercept_ = 0
        self.coef_ = np.zeros(n)

        for i in range(self.epochs):
            indices = np.random.permutation(m)
            X_suffle = X_train[indices]
            y_suffle = y_train[indices]
            for batch in range(0,m,self.batch_size):
                X_batch = X_suffle[batch:batch+self.batch_size]
                y_batch = y_suffle[batch:batch+self.batch_size]
                batch_size = len(X_batch)

                y_pred = (X_batch @ self.coef_) + self.intercept_
                error = y_pred - y_batch
                intercept_derivative = np.sum(error) / batch_size
                coef_derivative = ((X_batch.T @ (X_batch @ self.coef_)) - (X_batch.T @ y_batch) + (self.alpha * self.coef_)) / batch_size

                self.coef_ -= self.learning_rate * coef_derivative
                self.intercept_ -= self.learning_rate * intercept_derivative
                if i % 10 == 0:
                    mse = np.mean(error**2)
                   # self.loss.append(mse)
        return self.loss
    def predict(self,X_test):
        return (X_test @ self.coef_) + self.intercept_
