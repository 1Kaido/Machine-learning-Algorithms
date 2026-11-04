class blending:
    def __init__(self,base_model,meta_model):
        self.base_models = base_model
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
        test_prediction = np.column_stack(test_predictions)
        final_pred = meta_model.predict(test_predictions)


from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(42)

# 4 Features, 2000 samples
X = np.random.randint(1, 100, (2000, 4))
y = (
    5 * X[:, 0]
    + 3 * X[:, 1]
    - 2 * X[:, 2]
    + 4 * X[:, 3]
    + np.random.normal(0, 5, 2000)
)

# 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
base_models = [LinearRegression(),GradientBoostingRegressor(),KNeighborsRegressor()]
meta_model = LinearRegression()
b = blending(base_models,meta_model)
b.fit(X_train,y_train)
b.predict(X_test)
