class blending:
    def __init__(self,base_models,meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        self.base_model_predictions = None

    def fit(self,X_train,y_train):
        base_model_predictions = []
        #train test split again
        X, X_val, y, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=42)
        #Step 2 train the base models
        for model in self.base_models:
            model.fit(X_train,y_train)
            pred = model.predict(X_val)
            base_model_predictions.append(np.array(pred))
        # new columns with contain prediction on all models
        prediction = np.column_stack(base_model_predictions)
        self.base_model_predictions = prediction
        #meta model fit
        meta_model.fit(self.base_model_predictions,y_val)
        print("shape of x and X val by new split",X.shape,X_val.shape)
        print("shape of y and y_val by new split",y.shape,y_val.shape)
        print("shape of X_train and y_train by user",X_train.shape,y_train.shape)

    def predict(self,X_test):
        test_predictions = []
        for model in self.base_models:
            pred = model.predict(X_test)
            test_predictions.append(pred)
        test_predictions = np.column_stack(test_predictions)
        final_pred = meta_model.predict(test_predictions)
        return final_pred