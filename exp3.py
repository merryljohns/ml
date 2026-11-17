import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression,LassoCV,RidgeCV
from sklearn.model_selection import train_test_split

data=load_diabetes()
#X=pd.DataFrame(data.data,columns=data.feature_names)
X=data.data
y=data.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)
y_predlin=model.predict(X_test)
print("linear model:")
print("r2 score:",r2_score(y_test,y_predlin))
print("mse score:",mean_squared_error(y_test,y_predlin))

model=LassoCV(cv=5,max_iter=10000)
model.fit(X_train,y_train)
y_predlas=model.predict(X_test)
print("lasso cv:")
print("r2 score:",r2_score(y_test,y_predlas))
print("mse score:",mean_squared_error(y_test,y_predlas))

model=RidgeCV(alphas=np.logspace(-3,3,50),cv=5)
model.fit(X_train,y_train)
y_predlri=model.predict(X_test)
print("ridge:")
print("r2 score:",r2_score(y_test,y_predlri))
print("mse score:",mean_squared_error(y_test,y_predlri))

plt.scatter(y_test,y_predlin)
plt.show()

