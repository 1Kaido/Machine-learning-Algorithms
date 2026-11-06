class StackingRegressor:
    def __init__(self,base_models,meta_model,n_splits=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.oof_predictions = None
        self.n_splits = n_splits
    def fit(self,X_train,y_train):
        from sklearn.model_selection import KFold
        self.oof_predictions = np.zeros((len(X_train), len(self.base_models)))
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for model_idx,model in enumerate(self.base_models):
            for train_index, val_index in kf.split(X_train):
                X_fold_train = X_train[train_index]
                X_fold_val = X_train[val_index]
                y_fold_train = y_train[train_index]
                y_fold_val = y_train[val_index]
                #print("",X_fold_val.shape)

                #fit model
                model.fit(X_fold_train,y_fold_train)
                pred = model.predict(X_fold_val)
                self.oof_predictions[val_index, model_idx] = pred
            model.fit(X_train, y_train) #Sklearn do this #AI advice 
        self.meta_model.fit(self.oof_predictions,y_train)

    def predict(self,X_test):
        test_predictions = np.zeros((len(X_test), len(self.base_models)))
        for model_idx,model in enumerate(self.base_models):
            test_pred = model.predict(X_test)
            test_predictions[:, model_idx] += (test_pred) #AI

        pred = self.meta_model.predict(test_predictions)

        #print(np.array(pred))
        return pred