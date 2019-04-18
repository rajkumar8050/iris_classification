# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 22:20:42 2019

@author: RAJ KUMAR GAUTAM
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 04:50:36 2019

@author: RAJ KUMAR GAUTAM
"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')
#%matplotlib inline
np.random.seed(0)  #for reproducibility
iris=pd.read_csv('D:\iris.csv')
iris.head()
iris["Species"].value_counts() #check how many unique species of iris ara in the dataset
val_plot=iris.Species.value_counts().plot(kind="pie",autopct='%.1f%%',figsize=(8,8))
val_plot.savefig('D:\\file6.png')
sns_plot=sns.pairplot(iris.drop("Id",axis=1),hue="Species",size=3) #visualize the relationship between pairs of features
sns_plot.savefig('D:\\file5.png')
y=iris.Species    #set target variable
X=iris.drop(["Species","Id"],axis=1) #select feature variable
le=LabelEncoder() #load the label encoder
y=le.fit_transform(y)
encoder=OneHotEncoder(sparse=False)
encoder.fit_transform(y.reshape(-1,1))
sc=StandardScaler() #load the standard scaler
sc.fit(X) #compute the mean and std of the feature data
X_scaled=sc.transform(X) #Scale the feature data to be of mean 0 and variance 1
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=.3,random_state=1) #split the dataset into 30% testing and 70%training
model=KNeighborsClassifier(n_neighbors=3) #load our classifier
model.fit(X_train,y_train)  #fit our model on the training data
prediction=model.predict(X_test)#make prediction with our trained model on the data
accuracy=accuracy_score(y_test,prediction)*100 #compare accuracy of predicted classes with test data
print('k-Nearest Neighbours accuracy | '+str(round(accuracy,4))+'%.')
plt.figure(figsize=(12,10))
plt.scatter(X_train[:,1],y_train)
plt.savefig('D:\\file1.png')
plt.scatter(X_test[:,1],prediction,color='green')
plt.savefig('D:\\file2.png')
plt.plot(X_test[:,1],prediction,color='pink')
plt.xlabel('id')
plt.ylabel('binary notation of iris_species')
plt.title('KNN CLASSIFIER IRIS DATASET')
plt.savefig('D:\\file3.png')
X_train.shape
neighRange = range(1,30)
# We can create Python dictionary using [] or dict()
scores = []
for k in neighRange:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    scores.append(accuracy_score(y_test, pred))
print('mean accuracy while k varies from (1 to 30) | '+str(round(sum(scores)/len(scores)*100,2))+'%.')