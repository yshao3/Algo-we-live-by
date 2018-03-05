import csv
from scipy.stats.stats import pearsonr
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import data
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

# Cross validation
def Leaveoneout(X, j):
	Xtrain = []
	Xtest = []
	for i in X:
		Xtrain.append(i[:j]+i[j+1:])
		Xtest.append(i[j])
	return Xtrain, Xtest

# get k nearest kneighbors according to similarity
def getkNeighbors(k, Xtrain):
	similarity = Pearson(Xtrain)
	kneighbors = []
	for s in similarity:
		s =np.reshape(s, (-1,1))
		neigh = NearestNeighbors(n_neighbors=k+1)
		neigh.fit(s) 
		kneighbors.append(neigh.kneighbors([[1]],return_distance=False).tolist()[0])
	return kneighbors

# make prediction according to k nearest kneighbors
def prediction(kneighbors, Xtest):
	prediction = []
	for neigh in kneighbors:
		pred = []
		for i in range(len(neigh)-1):
			pred.append(Xtest[neigh[i+1]])
		prediction.append(np.mean(np.array(np.array(pred)),axis= 0).tolist())
	return prediction

# calculate mean square error
def mse_error(pred, real):
	# print np.array(pred)- np.array(real)
	try:
		mse = ((np.array(pred) - np.array(real))**2).mean(axis = 1)
	except:
		mse = ((np.array(pred) - np.array(real))**2)
	return mse

# calculate correlation
def Pearson(x):
	return np.corrcoef(x)

# entire knn prediction according to k training set and test set, 
# and return train and test mean square error
def knn_prediction(k, Xtrain, Xtest):
	kneighbors = getkNeighbors(i, Xtrain)
	train_prediction = prediction(kneighbors, Xtrain)
	test_prediction = prediction(kneighbors, Xtest)
	train_mse = mse_error(train_prediction, Xtrain)
	
	test_mse =  mse_error(test_prediction, Xtest)
	return train_mse,test_mse


X = importdata("mattersOfTaste.csv")
similarity = Pearson(X)
Index_max = similarity.mean(axis = 1).argmax(axis = 0)
Index_min = similarity.mean(axis = 1).argmin(axis = 0)
print Index_max, Index_min
train_mse = [0]*24
test_mse = [0]*24
max_sim = [0]*24
min_sim = [0]*24

for j in range(0, 19):	
	Xtrain, Xtest = Leaveoneout(X, j)
	# calculate mean square error according to number of nearest neighbors
	for i in range(1, 25):
		mse_tr, mse_te = knn_prediction(i, Xtrain, Xtest)
		train_mse[i-1] += mse_tr.mean(axis = 0)
		test_mse[i-1] += mse_te.mean(axis = 0)
		max_sim[i-1] += mse_te[Index_max]
		min_sim[i-1] += mse_te[Index_min]
train_mse = (np.array(train_mse)/20).tolist()
test_mse = (np.array(test_mse)/20).tolist()
max_sim = (np.array(max_sim)/20).tolist()
min_sim = (np.array(min_sim)/20).tolist()

print train_mse[0],train_mse[4], train_mse[23]
print test_mse[0],test_mse[4], test_mse[23]
print max_sim[0],max_sim[4], max_sim[23]
print min_sim[0], min_sim[4], min_sim[23]
print train_mse
# Plot Image
k = []
for i in range (1,25):
	k.append(i)
plta=plt.plot(k,train_mse, '-', label='Train')
pltb=plt.plot(k,test_mse,'-', label='Test')
pltc=plt.plot(k,max_sim, '-', label='Highest simi person')
pltd=plt.plot(k,min_sim, '-', label='Lowest simi person')
plt.ylabel('Mean Square Error')
plt.xlabel('k nearest neighbor')
plt.title("Learning Curve Analysis")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()

