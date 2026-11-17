#k means clustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data=load_digits()
X=data.data
scaler=StandardScaler()
X=scaler.fit_transform(X)

inertia=[]
silhouettes=[]
K_range=range(2,15)
for i in K_range:
    model=KMeans(n_clusters=i,random_state=0,n_init=10)
    labels=model.fit_predict(X)
    inertia.append(model.inertia_)
    silhouette=silhouette_score(X,labels)
    silhouettes.append(silhouette)
    print("k: ",i)
    print("inertia, silhouetter", inertia, silhouette)
plt.plot(K_range,inertia)
plt.title("elbow method")
plt.show()

plt.plot(K_range,silhouettes)
plt.show()
    
