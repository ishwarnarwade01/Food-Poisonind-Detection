from lib2to3.pgen2.pgen import DFAState
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

data=pd.read_csv('Book2.csv')

#cdf = data[['diarreah','stomachpain','vomiting','nausea','bodypain','fever','headache','skinrashes','acidity','stoolcolor','output']]


data.describe()
data.info()
data.head()
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
encoder=LabelEncoder()
data.agent=encoder.fit_transform(data.agent)
data.diarreah=encoder.fit_transform(data.diarreah)
data.condition=encoder.fit_transform(data.condition)
data.stomachpain=encoder.fit_transform(data.stomachpain)
data.vomiting=encoder.fit_transform(data.vomiting)
data.nausea=encoder.fit_transform(data.nausea)
data.bodypain=encoder.fit_transform(data.bodypain)
data.fever=encoder.fit_transform(data.fever)
data.headache=encoder.fit_transform(data.headache)
data.skinrashes=encoder.fit_transform(data.skinrashes)
data.acidity=encoder.fit_transform(data.acidity)
data.stoolcolor=encoder.fit_transform(data.stoolcolor)
data.output=encoder.fit_transform(data.output)

#data.class=encoder.fit_transform(data.class)

data.head()
data.head()
data.info()
data=data.drop(columns=['id','condition','height','weight','agent','onset(hrs)','age'])
data.head()
data.head()
data.info()
X=data.drop('output',axis=1)
y=data['output']
X
y

#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.05,random_state=0)
#X_train=data.drop('output',axis=1)
#y_train=data['output']
#X_train=data.drop(data.columns[[10]], axis=1, inplace=True)

X_train=X#data.iloc[0:143]
y_train=data.iloc[:,-1]

#X_train=data.drop[10] ,axis
#y_train=data['output']
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)# training Model
#pred=model.predict(X_test) #predicting 'Exited'
#from sklearn.metrics import (accuracy_score,confusion_matrix)
#print(accuracy_score(y_test,pred))
#print(confusion_matrix(y_test,pred))

pickle.dump(model, open('model.pkl','wb'))

