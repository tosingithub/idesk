# create the test setup
import lightgbm as lgb
import pickle as pkl
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.preprocessing import LabelEncoder

import pandas as pd

#df['x1']= LabelEncoder().fit_transform(df['x1'])

data= {
 'x': [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
 'q': [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
 'b': [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
 'target': [0.0, 2.0, 1.5, 0.0, 5.1, 4.0, 0.0, 1.0, 2.0, 0.0, 2.1, 1.5]
}
df= pd.DataFrame(data) 
X, y=df.iloc[:, :-1], df.iloc[:, -1]
X= X.astype('float32')

# create two models                 
model1= LinearRegression()
model2 = lgb.LGBMRegressor(n_estimators=5, num_leaves=10, min_child_samples=1) 
ser_model1= X['x']==0.0
model1.fit(X[ser_model1], y[ser_model1])
model2.fit(X[~ser_model1], y[~ser_model1])

# define a class that mocks the model interface
class CombinedModel:
    def __init__(self, model1, model2):
        self.model1= model1
        self.model2= model2
        
    def predict(self, X, **kwargs):
        ser_model1= X['x']==0.0
        return pd.concat([
                pd.Series(self.model1.predict(X[ser_model1]), index=X.index[ser_model1]),
                pd.Series(self.model2.predict(X[~ser_model1]), index=X.index[~ser_model1])
            ]
        ).sort_index()

# create a model with the two trained sum models
# and pickle it
model= CombinedModel(model1, model2)
model.predict(X)
with open('model.pkl', 'wb') as fp:
    pkl.dump(model, fp)
model= model1= model2= None

# test load it
with open('model.pkl', 'rb') as fp:
    model= pkl.load(fp)
model.predict(X)