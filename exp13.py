import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data=pd.read_csv("WineQT.csv")
X=data.drop("quality",axis=1)
y=data['quality']

label=LabelEncoder()
y=label.fit_transform(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
results=[]
architectures=[[8],
               [32,16],
               [64,32,16]]
#
for layers in architectures:
    model=Sequential()
    model.add(Dense(layers[0],activation='relu',input_shape=(X_train.shape[1],)))

    for neurons in layers[1:]:
        model.add(Dense(neurons,activation='relu'))
    model.add(Dense(len(np.unique(y)),activation='softmax'))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(X_train,y_train,epochs=50,batch_size=32,verbose=0)
    loss,accuracy=model.evaluate(X_test,y_test,verbose=0)
    results.append(loss)
    print("layers:",layers)
    print("accuracy,loss:",accuracy,loss)
plt.bar(['[8]','32,16','64,32,16'],results)
plt.show()




















    
    
    
