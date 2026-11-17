import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.svm import SVC

iris=datasets.load_iris()
X=iris.data[:,:2]
y=iris.target
y=np.where(y==0,1,-1)

model=SVC(kernel='linear',C=100)
model.fit(X,y)
w=model.coef_[0]
b=model.intercept_[0]

plt.scatter(X[:,0],X[:,1],c=y)
x=np.linspace(4,8,100)
plt.plot(x,-(w[0]*x+b)/w[1],'k-',label='decision boundary')
plt.plot(x,-(w[0]*x+b-1)/w[1],'r--',label='margins')
plt.plot(x,-(w[0]*x+b+1)/w[1],'r--',label='margins')

plt.scatter(model.support_vectors_[:,0],
            model.support_vectors_[:,1],
            s=100,facecolors='none',edgecolors='black')
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.title("linear svm")

plt.show()
