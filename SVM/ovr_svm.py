
class OnevRestSoftMarginSVM:
    def __init__(self, learning_rate=0.001,epochs=100,c=1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coef_ = None
        self.intercept_ = None
        self.c = c
        self.models = []
        self.classes = None

    def cal_score(self,X,coef_,intercept_):
        norm = np.linalg.norm(coef_) #Advice
        return (X @ coef_ + intercept_)/ (norm + 1e-12)

    def fit(self,X_train,y_train):
        self.models = []
    #This was the issue next time keep in mind to reset
    #Finally My model improve 0.666 ->
        self.classes = np.unique(y_train)
        #print(self.classes)
        #Itrate for each unique in y_train
        for each_val in self.classes:
            coef_ = np.zeros(X_train.shape[1])
            intercept_ = 0
            y_binary = np.where(y_train == each_val, 1, -1)
            for i in range(self.epochs):
                indices = np.random.permutation(X_train.shape[0])
                for idx in indices:
                    x_i = X_train[idx]
                    y_i = y_binary[idx]

                    if y_i * (coef_ @ x_i + intercept_) >= 1: # No Violation
                    #divide cuz even no violation the line moves it lil so to prevent
                        d_w = (coef_/ X_train.shape[0])
                        d_b = 0 #cuz it 0 so line is stady
                        coef_ = coef_ - (self.learning_rate*d_w)
                        intercept_ = intercept_ - (self.learning_rate * d_b)
                    else: # violation
                        d_w = (coef_/ X_train.shape[0]) -(self.c * y_i * x_i)
                        d_b = -(self.c*y_i)
                        coef_ = coef_ - (self.learning_rate*d_w)
                        intercept_ = intercept_ - (self.learning_rate * d_b)
            model_data = {
                'class': each_val,
                'coef': coef_,
                'intercept': intercept_
            }
            self.models.append(model_data)

            #prediction = self.cal_score(X_train,coef_,intercept_)
           # self.model.append(prediction)
       # pprint(self.models)
    def predict(self,X_test):

        scores = []
        for each_model in self.models:
           coef = each_model['coef']
           intercept = each_model['intercept']

           score = self.cal_score(X_test,coef,intercept)
           scores.append(score)
           #pprint(f"Class {each_model['class']} scores: {score}")
        #pprint(scores)
        scores = np.array(scores)
        class_indices = np.argmax(scores, axis=0)

        return self.classes[class_indices]
