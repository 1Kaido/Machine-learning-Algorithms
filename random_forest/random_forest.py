class DecisionTrees:
    def __init__(self,max_features=1):
        self.parent_entropy = 0 #Single val so no dict
        self.children_entropy = {}
        self.classes = []
        self.max_features = max_features
        self.tree = None

    def entropy(self,y):
        classes = np.unique(y)
        total = len(y)
        total_entropy = 0
        for cls in classes:
            count = np.sum(y == cls)
            p = count / total
            total_entropy += -p * np.log2(p)
        return total_entropy

    def entropy_for_all(self,X_train,y_train):
        self.children_entropy = {}
        self.classes = np.unique(y_train)
        self.parent_entropy = self.entropy(y_train)

        for features in X_train.columns:
            self.children_entropy[features] = {}
            values = np.unique(X_train[features])
            for val in values:
                child_y = y_train[X_train[features] == val]
                entropy = self.entropy(child_y)
                self.children_entropy[features][val] = entropy #AI

    def information_gain(self,X_train,y_train):
        self.entropy_for_all(X_train,y_train)
        #weighted entropy
        weighted_entropy = {}
        information_gain = {}
        #random features
        n_features = min(self.max_features, X_train.shape[1])
        X_features_columns = np.random.choice(X_train.columns,size=n_features,replace=False)
        for features in X_features_columns:
            weighted_ = 0
            values, counts = np.unique(X_train[features], return_counts=True)
            for value,count in zip(values,counts):
                weighted_ += ((count/X_train.shape[0])*self.children_entropy[features][value])
            weighted_entropy[features] = weighted_
        #Information Gain
        for features in X_features_columns:
            ig =  self.parent_entropy - weighted_entropy[features]
            information_gain[features] = ig
        best_feature = max(information_gain, key=information_gain.get)
        return best_feature

    def build_tree(self,X_train,y_train):
        majority_class = y_train.value_counts().idxmax()
        if len(np.unique(y_train)) == 1:
            return y_train.iloc[0]
        if X_train.shape[1] == 0:
             return majority_class
        best_feature = self.information_gain(X_train,y_train)
        tree = {best_feature: {}}
        values = np.unique(X_train[best_feature])
        for value in values:
            #AI help
            mask = X_train[best_feature] == value
            x_child = X_train[mask]
            y_child = y_train[mask]
            x_child = x_child.drop(columns=[best_feature])
            tree[best_feature][value] = self.build_tree(x_child, y_child)
        self.tree = tree
       # print(self.tree)
        return tree

    def fit(self, X_train, y_train):
        self.tree = None
        self.children_entropy = {}
        self.build_tree(X_train, y_train)

    def predict(self,X_test):
        predictions = []
        for _, row in X_test.iterrows():
            #AI help
            node = self.tree
            while isinstance(node, dict):
                feature = list(node.keys())[0]
                value = row[feature]
                if value not in node[feature]:
                    node = self.classes[0]
                    break
                node = node[feature][value]
            predictions.append(node)
        return predictions


import copy
class MyRandomForest:
    def __init__(self,max_features=1,n_estimators=10,max_samples=0.5,bootstrap=True):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.model = DecisionTrees(self.max_features)
        self.trees = []

    def fit(self,X_train,y_train):


        for i in range(self.n_estimators):
            if self.max_samples <= 0:
                print("No columns try increasing max_samples")
                break
            indices = np.random.choice(
                len(X_train),
                size=int(self.max_samples*len(X_train)),
                replace = self.bootstrap
            )
            X_bootstrap = X_train.iloc[indices]
            y_bootstrap = y_train.iloc[indices] #cuz a df
            tree = copy.deepcopy(self.model)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self,X_test):
        if len(self.trees) == 0:
            print("No tree Found")
            return
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
