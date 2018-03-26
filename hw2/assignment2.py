import csv
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize
import numpy as np
import math


# Import data
def importdata(path):
	X = []
	Y = []
	read = csv.reader(open(path))
	read.next()
	for row in read:
		# no transformation
		# X.append([int(row[9]), int(row[10]), float(row[11]), float(row[12])])
		# use log transformation
		X.append([float(row[11]), float(row[12]), math.log(float(row[9])), math.log(float(row[10]))])
		# Add log(nb1/nb2), nb1-nb2, r1-r2 to features
		# X.append([float(row[11]),float(row[12]), math.log(float(row[9])/float(row[10]))/math.log(2), float(row[9])-float(row[10]),float(row[11])-float(row[12]), math.log(float(row[9])), math.log(float(row[10]))])
		Y.append([int(row[8])])
	return X, Y

# Select features most related feature for given model
def feature_selection(X, Y, estimator):
	selector = RFE(estimator)
	choosen = selector
	score  =  0
	for i in range(1,7):
		selector = RFE(estimator,i)
		selector = selector.fit(X, Y)
		if selector.score(X,Y) > score:
			choosen = selector
			score = selector.score(X, Y)
	print choosen.ranking_
	return choosen.transform(X)

# Perform cross validation test
def cross_validation(X, Y, estimator):
	scores = cross_val_score(estimator, X, Y, cv = 4)
	return scores

X, Y = importdata("choiceWithSocialCues.csv")

# Using logistic regression to select feature and predict
estimator1 = LogisticRegression()
x = normalize(X, norm = 'l2', axis = 1)
# x = normalize(X, norm = 'l1', axis = 1)
# x = normalize(X, norm = 'max', axis = 1)
x = feature_selection(x, Y, estimator1)
scores = cross_validation(x, Y, estimator1)
print scores.mean()

# Using DecisionTree to select feature and predict
estimator1 = DecisionTreeClassifier()
x = feature_selection(X, Y, estimator1)
scores = cross_validation(x, Y, estimator1)
print scores.mean()

# Using Nueral net with to select feature and predict
estimator1 = MLPClassifier(hidden_layer_sizes=(50,50,))
# x = normalize(X, norm = 'l2', axis = 1)
scores = cross_validation(X, Y, estimator1)
print scores.mean()
