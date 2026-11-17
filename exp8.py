import numpy as np
import pandas as pd
#from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import accuracy_score,f1_score,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#(X_train,y_train),(X_test,y_test)=fashion_mnist.load_data()
#X_train=np.reshape(len(X_train),-1)/255.0
#X_test=np.reshape(len(X_test),-1)/255.0
df=pd.read_csv("fashion-mnist_train.csv")
X=df.drop('label',axis=1)
y=df['label']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
                                               
for k in [1,3,5]:
    knn=KNeighborsClassifier(n_neighbors=k,n_jobs=-1)
    knn.fit(X_train[:10000],y_train[:10000])
    y_pred=knn.predict(X_test[:2000])
    print("k=",k)
    print("accuracy score:",accuracy_score(y_test[:2000],y_pred))

