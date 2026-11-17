#multinomial nb and bernoulli nb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score,f1_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

data=fetch_20newsgroups(subset='all',categories=['sci.space','rec.sport.baseball','comp.graphics'],
                        remove=('headers','footers','quotes'))
X=data.data
y=data.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

v=CountVectorizer(stop_words='english',max_features=5000)
X_trainmb=v.fit_transform(X_train)
X_testmb=v.transform(X_test)
mb=MultinomialNB()
mb.fit(X_trainmb,y_train)
y_predmb=mb.predict(X_testmb)
print("multinomial mb:")
print("f1_score:",f1_score(y_test,y_predmb,average='macro'))
print("accuracy score:",accuracy_score(y_test,y_predmb))

v=CountVectorizer(stop_words='english',binary=True,max_features=5000)
X_trainmb=v.fit_transform(X_train)
X_testmb=v.transform(X_test)
mb=BernoulliNB()
mb.fit(X_trainmb,y_train)
y_predmb=mb.predict(X_testmb)
print("bernoulli nb:")
print("f1_score:",f1_score(y_test,y_predmb,average='macro'))
print("accuracy score:",accuracy_score(y_test,y_predmb))
