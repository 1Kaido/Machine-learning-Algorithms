class XgboostTreeRegression:
    def __init__(self, lamda=1,gamma=0.5,max_depth=3):
        self.lamda = lamda
        self.gamma = gamma
        self.tree = None
        self.max_depth = max_depth

    def all_threshold(self,X_train):
        thresholds = []
        x_unique = np.unique(X_train)
        for i in range(len(x_unique)-1):
            thes = (x_unique[i] + x_unique[i+1])/2
            thresholds.append(thes)
     #   print(f"Thresholds: {thresholds}")
        return thresholds

    def similarity(self,gradients,hessians,left_nodes,right_nodes):
        #Similarity score
        root_similarity = (np.sum(gradients)**2 /(np.sum(hessians) + self.lamda))
       # print(f"gradients: {len(gradients)}")
       ## print(f"left_nodes: {len(left_nodes)}")
       # print(f"right_nodes: {len(right_nodes)}")
        left_gradient = gradients[left_nodes]
        right_gradient = gradients[right_nodes]
        G_left = np.sum(left_gradient)
        G_right = np.sum(right_gradient)
        H_left = np.sum(hessians[left_nodes])
        H_right = np.sum(hessians[right_nodes])
        score_left = G_left**2/(H_left + self.lamda)
        score_right = G_right**2/(H_right + self.lamda)
        score = (score_left + score_right) - root_similarity
       # print(f"score: {score}")
        return score

    def splitting(self,X_train, feature,threshold):
        left_nodes = X_train[:,feature] <= threshold
        right_nodes = X_train[:,feature] > threshold
        return left_nodes,right_nodes


    def gain(self, X_train, gradients, hessians):
        best_split = None
        best_gain = -np.inf
        for feature in range(X_train.shape[1]):
            thresholds = self.all_threshold(X_train[:, feature])
            for threshold in thresholds:
                left, right = self.splitting(X_train, feature, threshold)
                if not np.any(left) or not np.any(right):
                    continue
                score = self.similarity(gradients, hessians, left, right)
                gain = 0.5 * score
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature, threshold, gain)
        if best_split is None:
            return None, None, -np.inf
        return best_split

    def build_tree(self,X_train,gradients,hessians,depth=0):
        if depth >= self.max_depth or len(X_train) < 2:
            return {"leaf": -np.sum(gradients) / (np.sum(hessians) + self.lamda)}
       # print(f"len of gradient in tree: {len(gradients)}")
        #best_threshold = self.gain(X_train, gradients,hessians)
        feature, threshold, best_gain = self.gain(X_train, gradients, hessians)
        if feature is None or best_gain <= self.gamma:
            return {"leaf": -np.sum(gradients) /(np.sum(hessians) + self.lamda)}

        left_nodes, right_nodes = self.splitting(X_train, feature, threshold)
        left_gradient = gradients[left_nodes]
        right_gradient = gradients[right_nodes]
        X_left = X_train[left_nodes]
        X_right = X_train[right_nodes]
        left_hessian = hessians[left_nodes]
        right_hessian = hessians[right_nodes]
        return {
            "feature": feature,
            "threshold": threshold,
            "left": self.build_tree(X_left, left_gradient,left_hessian, depth + 1),
            "right": self.build_tree(X_right, right_gradient,right_hessian,depth + 1)
        }

    def fit(self,X_train, gradients, hessians):
        self.tree = self.build_tree(X_train, gradients, hessians)
        return self

    def predict_tree(self,X_test,node):
        if "leaf" in node:
            return np.full(len(X_test), node["leaf"])

        feature = node["feature"]
        threshold = node["threshold"]
        left_mask = X_test[:, feature] <= threshold
        right_mask = X_test[:, feature] > threshold
        predictions = np.zeros(len(X_test))
        predictions[left_mask] = self.predict_tree(
        X_test[left_mask], node["left"])
        predictions[right_mask] = self.predict_tree(
        X_test[right_mask], node["right"])
        return predictions

    def predict(self, X_test):
        return self.predict_tree(X_test,self.tree)

class XgboostRegression:
    def __init__(self,n_estimators=50,learning_rate=0.01,lamda=1,gamma=0.5,max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.lamda = lamda
        self.gamma = gamma
        self.max_depth = max_depth
        self.base_pred = None
        self.trees = []

    def fit(self,X_train,y_train):
        self.trees = []
        self.base_pred = np.mean(y_train)
        pred = np.full(len(y_train),self.base_pred)
        for i in range(self.n_estimators):
            gradients = pred - y_train
            hessains = np.ones(len(y_train))
            model = XgboostTreeRegression(self.lamda,self.gamma,self.max_depth)
            model.fit(X_train, gradients, hessains)
            tree_predict = model.predict(X_train)
            pred += self.learning_rate*tree_predict
            if i%20 == 0:
                mse = np.mean((y_train - pred) ** 2)
                print(i,mse)
            self.trees.append(model)
       # print(self.trees)
       # print(pred)

    def predict(self, X_test):
        pred = np.full(len(X_test),self.base_pred)
        for tree in self.trees:
            tree_pred = tree.predict(X_test)
            pred += self.learning_rate * tree_pred
        return pred