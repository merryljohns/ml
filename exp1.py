import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score
df=pd.read_csv("housing.csv")
print(df.head())
X=df["Avg. Area Number of Rooms"].values
y=df["Price"].values

mask=~np.isnan(X) & ~np.isnan(y)
X,y=X[mask],y[mask]

X_mean=np.mean(X)
X_std=np.std(X)
X=(X-X_mean)/X_std

n=len(X)

theta0=0
theta1=0

learning_rate=0.01
iterations=1000
cost_history=[]

for i in range(iterations):
    y_pred=theta0+theta1*X
    error=y-y_pred

    d_theta0=(-2/n)*np.sum(error)
    d_theta1=(-2/n)*np.sum(error*X)

    theta0-=d_theta0*learning_rate
    theta1-=d_theta1*learning_rate
    cost_history.append(np.mean(error**2))
y_pred_gd=theta0+theta1*X

#intercept=theta0 slope=theta1

X_b=np.c_[np.ones(n),X]
theta=np.linalg.pinv(X_b)@y
theta0_ne,theta1_ne=theta
y_pred_ne=X_b@theta

print("gradient descent metrics:")
print("r2 score:",r2_score(y,y_pred_gd))
print("mse:",mean_squared_error(y,y_pred_gd))


print("normal eqn metrics:")
print("r2 score:",r2_score(y,y_pred_ne))
print("mse:",mean_squared_error(y,y_pred_ne))


idx=np.argsort(X)
plt.figure(figsize=(10,6))
plt.scatter(X,y,color='blue',alpha=0.3,label="actual data")
plt.plot(X[idx],y_pred_gd[idx],color='red',label='gradient descent')
plt.plot(X[idx],y_pred_ne[idx],color='green',linestyle='--',label='normal eqn')

plt.xlabel("normalized total rooms")
plt.ylabel("median house value")
plt.title("linear regression")
plt.legend()
plt.grid()
plt.show()

plt.plot(cost_history,color='green')
plt.title("cost functions convergence")
plt.xlabel("iterations")
plt.ylabel("cost")
plt.grid()
plt.show()

new_rooms = 3000
new_rooms = (new_rooms - X_mean) / X_std

print("\nPREDICTION")
print("Gradient Descent:", theta0 + theta1 * new_rooms)
print("Normal Equation:", theta0_ne + theta1_ne * new_rooms)















