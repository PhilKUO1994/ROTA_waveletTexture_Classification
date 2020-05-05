import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,Perceptron
from sklearn import metrics
from sklearn.svm import SVC

def score(y_pred,y_test):
	counter = 0
	for i in range(len(y_pred)):
		counter += abs(y_pred[i]-y_test[i][0])
	specifity = 0
	for i in range(len(y_test)):
		if y_test[i][0]==0 and y_pred[i] ==0:
			specifity+=1
	specifity = specifity/(len(y_test)-sum(y_test))

	sensitivity = 0
	for i in range(len(y_test)):
		if y_test[i][0]==1 and y_pred[i] ==1:
			sensitivity+=1
	sensitivity = sensitivity/sum(y_test)
	return 1-counter/len(y_pred),specifity,sensitivity

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train.ravel())
y_pred = logreg.predict(X_test)
print()
acc,spe,sen =score(y_pred,y_test)
print('Accuracy, specifity, sensitivity of logistic regression classifier on test set: ',round(acc,4),round(spe[0],4),round(sen[0],4))

#Perceptron
for i in range(len(y_train)):
	y_train[i] =y_train[i][0]
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X_train, y_train.ravel())
y_pred = clf.predict(X_test)
acc,spe,sen =score(y_pred,y_test)
print('Accuracy, specifity, sensitivity of Perceptron on test set: ',round(acc,4),round(spe[0],4),round(sen[0],4))

#SVM
for i in range(len(y_train)):
	y_train[i] =y_train[i][0]
clf = SVC(kernel='linear')
clf.fit(X_train, y_train.ravel())
y_pred = clf.predict(X_test)
acc,spe,sen =score(y_pred,y_test)
print('Accuracy, specifity, sensitivity of Support Vector Classification on test set: ',round(acc,4),round(spe[0],4),round(sen[0],4))
