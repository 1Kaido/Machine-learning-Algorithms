class XgboostTreeRegression:
    def __init__(self,lamda=1,gamma=0.5,max_depth=0):
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

    def gain(self,X_train, gradients,hessians):
       
        best_gains = []
        for feature in range(X_train.shape[1]):
            gains = []
            thresholds = self.all_threshold(X_train[:,feature])
            for threshold in thresholds:
                left_nodes,right_nodes = self.splitting(X_train, feature,threshold)
                score = self.similarity(gradients, hessians,left_nodes,right_nodes)
                gain = 0.5 * score
                #print(f"Gain for {feature}: {gain} threshold: {threshold}")
                gains.append(gain)
                best_index = np.argmax(gains)
                best_threshold = thresholds[best_index]
           # print("gain",best_threshold)
                best_gains.append((feature, best_threshold, gains[best_index]))
                #print(f"Best gain: {best_threshold}")
       # return best_threshold
        best_split = max(best_gains, key=lambda x: x[2])
        return best_split

    def build_tree(self,X_train,gradients,hessians,depth=0):
        if depth >= 2 or len(X_train) < 2:
            return {"leaf": -np.sum(gradients) / (np.sum(hessians) + self.lamda)}
       # print(f"len of gradient in tree: {len(gradients)}")
        #best_threshold = self.gain(X_train, gradients,hessians)
        feature, threshold, best_gain = self.gain(X_train, gradients, hessians)
        if best_gain <= self.gamma:
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

    def fit(self,X_train,y_train):
        pred = np.full(len(y_train),np.mean(y_train))
        gradients = pred - y_train
        #print(f"len of pred: {len(pred),len(gradients)}")
        hessians = np.ones(len(y_train))
        tree = self.build_tree(X_train,gradients,hessians,self.max_depth)
        import json
        #print(json.dumps(tree, indent=4))
        #print("Y:", y_train.min(), y_train.max())
        #print("Gradient:", gradients.min(), gradients.max())
        #print("Hessian:", hessians.min(), hessians.max())
        self.tree = self.build_tree(X_train, gradients, hessians)
      

    def predict(self,X_test):
        #Code here