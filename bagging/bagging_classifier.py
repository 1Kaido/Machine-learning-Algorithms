
class MyBaggingClassifier:
    def __init__(self,model,n_estimators=10,max_samples=0.5,bootstrap=True):
        self.n_estimators = n_estimators
        self.model = model
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.trees = []

    def fit(self,X_train,y_train):

        for i in range(self.n_estimators):
            indices = np.random.choice(
                len(X_train),
                size=int(self.max_samples*len(X_train))
            )

            X_bootstrap = X_train.iloc[indices]
            y_bootstrap = y_train.iloc[indices] #cuz a df

            tree = copy.deepcopy(self.model)
            tree.fit(X_bootstrap, y_bootstrap)

            self.trees.append(tree)

    def predict(self,X_test):
        predictions = []

        for val in range(X_test.shape[0]):
            raw_predictions = []

            for tree in self.trees:
                pred = tree.predict(X_test.iloc[[val]])[0]
                raw_predictions.append(pred)

            final = max(
                set(raw_predictions),
                key=raw_predictions.count
            )

            predictions.append(final)

        return predictions
