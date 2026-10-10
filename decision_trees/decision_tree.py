class DecisionTrees:
    def __init__(self):
        self.parent_entropy = 0 #Single val so no dict
        self.children_entropy = {}
        self.classes = []
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
        self.classes = np.unique(y_train)
        self.parent_entropy = self.entropy(y_train)

        for features in X_train.columns:
            self.children_entropy[features] = {}
            values = np.unique(X_train[features])
            for val in values:
                child_y = y_train[X_train[features] == val]
                entropy = self.entropy(child_y)
                self.children_entropy[features][val] = entropy #AI
       # pprint(self.children_entropy)

    def information_gain(self,X_train,y_train):
        self.entropy_for_all(X_train,y_train)
        #weighted entropy
        weighted_entropy = {}
        information_gain = {}
        for features in X_train.columns:
            weighted_ = 0
            values, counts = np.unique(X_train[features], return_counts=True)
            for value,count in zip(values,counts):
                weighted_ += ((count/X_train.shape[0])*self.children_entropy[features][value])
            weighted_entropy[features] = weighted_
        #Information Gain
        for features in X_train.columns:
            ig =  self.parent_entropy - weighted_entropy[features]
            information_gain[features] = ig
        best_feature = max(information_gain, key=information_gain.get)
        return best_feature

    def build_tree(self,X_train,y_train):
        if len(np.unique(y_train)) == 1:
            return y_train.iloc[0]
        best_feature = self.information_gain(X_train,y_train)
        tree = {best_feature: {}}
        values = np.unique(X_train[best_feature])
        for value in values:
            #AI help
            mask = X_train[best_feature] == value
            x_child = X_train[mask]
            y_child = y_train[mask]
            x_child = x_child.drop(columns=[best_feature])
           # print(x_child)
            tree[best_feature][value] = self.build_tree(x_child, y_child)
        #pprint(tree)
        self.tree = tree
        return tree

    def fit(self,X_train,y_train):
        self.build_tree(X_train,y_train)

    def predict(self,X_test):
       # print(json.dumps(self.tree, indent=4))
       # print(X_test)
       #For X_test[0] is No
       # print(X_test.iloc[0])
       # print(self.tree["Outlook"]["Rain"]["Wind"]["Strong"])
        predictions = []

        for _, row in X_test.iterrows():
            #AI help
            node = self.tree
            while isinstance(node, dict):
                feature = list(node.keys())[0]
                value = row[feature]
                node = node[feature][value]
            predictions.append(node)
        return predictions
