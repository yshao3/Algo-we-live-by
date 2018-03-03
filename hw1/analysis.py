import csv
from scipy.stats.stats import pearsonr
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def importdata(path):
	res = []
	read = csv.reader(open(path))
	read.next()
	for row in read:
		if len(res) == 0:
			for i in range(0, len(row)-1):
				res.append([])
		for i in range(0, len(row)-1):
			if row[i+1] != '':
				res[i].append(int(row[i+1]))
	return res

def Devide(X):
	Xtrain = []
	Xtest = []
	for i in X:
		Xtrain.append(i[:15])
		Xtest.append(i[15:])
	return Xtrain, Xtest

def getkNeighbors(k, Xtrain):
	similarity = Pearson(Xtrain)
	kneighbors = []
	for s in similarity:
		s =np.reshape(s, (-1,1))
		neigh = NearestNeighbors(n_neighbors=k+1)
		neigh.fit(s) 
		kneighbors.append(neigh.kneighbors([[1]],return_distance=False).tolist()[0])
	return kneighbors

def prediction(kneighbors, Xtest):
	prediction = []
	for neigh in kneighbors:
		pred = []
		for i in range(len(neigh)-1):
			pred.append(Xtest[neigh[i+1]])
		prediction.append(np.mean(np.array(np.array(pred)),axis= 0).tolist())
	return prediction
def mse_error(pred, real):
	mse = ((np.array(pred) - np.array(real))**2).mean(axis = 1)
	return mse
def Pearson(x):
	return np.corrcoef(x)
def knn_prediction(k, Xtrain, Xtest):
	kneighbors = getkNeighbors(i, Xtrain)
	train_prediction = prediction(kneighbors, Xtrain)
	test_prediction = prediction(kneighbors, Xtest)
	train_mse = mse_error(train_prediction, Xtrain)
	test_mse =  mse_error(test_prediction, Xtest)
	return train_mse.mean(axis = 0),test_mse.mean(axis = 0)
			
X = importdata("mattersOfTaste.csv")	
Xtrain, Xtest = Devide(X)
train_mse = []
test_mse = []
for i in range(1, 25):
	mse_tr, mse_te = knn_prediction(i, Xtrain, Xtest)
	train_mse.append(mse_tr)
	test_mse.append(mse_te)


