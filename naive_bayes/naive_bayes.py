class NaiveBayes:
    def __init__(self):
        self.priors = {}
        self.likelihood = {}
        self.classes = []

    def fit(self,X_train,y_train):
         total_len = len(y_train)
         self.classes = np.unique(y_train)

         for cls in self.classes:
             total_sum = np.sum(y_train == cls)
             self.priors[cls] = total_sum/total_len
             self.likelihood[cls] = {}
             for features in range(X_train.shape[1]):
                 values = np.unique(X_train.iloc[:,features])
                 #print(X_train.columns[features])
                 self.likelihood[cls][X_train.columns[features]] = {}
                 number_of_values = len(values)
                 for val in values:
                     self.likelihood[cls][X_train.columns[features]][val] = {}
                     numerator = 0
                     denominator = 0
                     for i in range(X_train.shape[0]):
                         current_x = X_train.iloc[i,features]
                         current_y = y_train.iloc[i]

                         if current_y == cls:
                             denominator+=1
                             if current_x == val:
                                 numerator+=1
                     if denominator == 0:
                         return 0.0
                     else:
                         probability = (numerator + 1) / (denominator + number_of_values)
                         self.likelihood[cls][X_train.columns[features]][val] = probability
        # pprint(self.likelihood,sort_dicts=False)

    def predict(self,X_test):
        predictions = []

        for i in range(X_test.shape[0]):
            scores = {}
            for cls in self.classes:
                score = np.log(self.priors[cls])
                for features in range(X_test.shape[1]):
                    feature = X_test.columns[features]
                    val = X_test.iloc[i, features]

                    score += np.log(self.likelihood[cls][feature][val])
                scores[cls] = score
                 #   print(cls, features,val ,score)

            prediction = max(scores, key=scores.get)
            predictions.append(prediction)
       # print(self.prediction)
        return predictions
