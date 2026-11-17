import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error


df=pd.read_csv("auto-mpg.csv")
print(df.head().to_string())
print(df.isnull().sum())


X=df[['mpg']]
y=df['displacement']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("linear regression:")
print("r2 score:",r2_score(y_test,y_pred))
print("mse:",mean_squared_error(y_test,y_pred))


poly2=PolynomialFeatures(degree=2)
X_train2=poly2.fit_transform(X_train)
X_test2=poly2.transform(X_test)
model2=LinearRegression()
model2.fit(X_train2,y_train)
y_pred2=model2.predict(X_test2)
print("polynomial regression degree 2")
print("r2 score:",r2_score(y_test,y_pred2))
print("mse:",mean_squared_error(y_test,y_pred2))

#degree 3
poly3=PolynomialFeatures(degree=3)
X_train3=poly3.fit_transform(X_train)
X_test3=poly3.transform(X_test)
model3=LinearRegression()
model3.fit(X_train3,y_train)
y_pred3=model3.predict(X_test3)
print("polynomial regression degree 3")
print("r2 score:",r2_score(y_test,y_pred3))
print("mse:",mean_squared_error(y_test,y_pred3))




X_range=np.linspace(X.min(),X.max(),100).reshape(-1,1)
plt.scatter(X,y,color='red',label='actual data')
plt.plot(X_range,model.predict(X_range),color='blue',label="linear regression")

plt.plot(X_range,model2.predict(poly2.fit_transform(X_range)),color='green',label='degree 2')
plt.plot(X_range,model3.predict(poly3.fit_transform(X_range)),color='pink',label='degree 3')
plt.grid()
plt.xlabel("displacement")
plt.ylabel("mpg")
plt.title("linear vs polynomial regression")
plt.legend()
plt.show()





















