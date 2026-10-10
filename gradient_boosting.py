class GradientBoosting:
    def __init__(self,learning_rate=0.1,n_estimators=100,max_depth=2):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None

    def fit(self,X_train,y_train):
       # X_train = X_train.to_array()
       # y_train = y_train.to_array()

        #Restart Tree
        self.trees = []
        self.initial_prediction = np.mean(y_train)
        #Ever val in Y_train become mean
        predictions = np.full(len(y_train),self.initial_prediction)

        for i in range(self.n_estimators):
            residual = y_train - predictions
            #Tree fit
            from sklearn.tree import DecisionTreeRegressor
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X_train, residual)
            self.trees.append(tree)
            pred = tree.predict(X_train)
            predictions = predictions + self.learning_rate*pred
        print(predictions)

    def predict(self,X_test):
        prediction = np.full(len(X_test),self.initial_prediction)
        for tree in self.trees:
            model = tree.predict(X_test)
            #Update
            prediction += self.learning_rate*model
        return prediction
