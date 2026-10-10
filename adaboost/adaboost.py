class AdaBoost:
    def __init__(self,n_estimators):
        self.models = []
        self.alphas = []
        self.n_estimators = n_estimators

    def fit(self,X_train,y_train):
        #Empty Arrays remember the mistake we made
        self.models = []
        self.alphas = []

        #initialize weights
        weights = np.ones(len(X_train)) / len(X_train)
        #Random Samples pass to next model and repeat
        for i in range(self.n_estimators):
            indices = np.random.choice(
                len(X_train),
                size=len(X_train),
                replace=True,
                p=weights#Dataset on Weights Base
            )
            X_new = X_train[indices]
            y_new = y_train[indices]
            stumps = DecisionTreeClassifier(max_depth=1)
            stumps.fit(X_new,y_new)
            self.models.append(stumps)
            pred = stumps.predict(X_train)
            #Calculate weights error
            wrong = pred != y_train
            correct = pred == y_train
            wrong_weights = weights[wrong]
            correct_weights = weights[correct] # or weights[~wrong]
            error = np.sum(wrong_weights)
            #Calculate Alpha
            alpha = (1/2)*np.log((1-error)/error)
            self.alphas.append(alpha)
            #Update weights according to classification
            weights[wrong] = weights[wrong] * np.exp(alpha)
            weights[correct] = weights[correct] * np.exp(-alpha)
            #Normalize weights
            weights = weights/np.sum(weights)

       # print(self.alphas)
       # print(self.models)

    def predict(self,X_test):
        final = np.zeros(X_test.shape[0])
        for model,alpha in zip(self.models,self.alphas):
            pred = model.predict(X_test)
            final += alpha*pred
        print(final)
        return np.sign(final)
