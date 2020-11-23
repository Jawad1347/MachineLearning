#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 02:51:22 2020

@author: julia
"""

import pandas as pd
# train = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/annealing/anneal.data")
# test = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/annealing/anneal.test")
train = pd.read_csv('anneal.data')
test = pd.read_csv('anneal.test')

columns = ["family", "product-type", "steel", "carbon", "hardness", "temper_rolling",
"condition", "formability", "strength", "non-ageing", "surface-finish", "surface-quality",
"enamelability", "bc", "bf", "bt", "bw/me", "bl", "m", "chrom", "phos", "cbond", "marvi",
"exptl", "ferro", "corr", "blue/bright/varn/clean", "lustre", "jurofm", "s", "p", "shape",
"thick", "width", "len", "oil", "bore", "packing", "classes"]

W = pd.concat([train, test], ignore_index=True)
W = W.iloc[:,:-3]
W.columns = columns

X = W.loc[:, W.isin(["?"]).sum()<len(W)*.20]
X.isin(['?']).sum(axis=0)

import numpy as np
X = X.replace('?', np.nan)
X = X.dropna()

#Xt = X.iloc[:,:-1]
y = X.iloc[:,-1]

X = X[["steel", "shape", "thick", "width", "len"]]

import sklearn

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Feature Scaling cheating by selecting last three columns only
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xscal = X.iloc[:,2:]
#Xscal_test = X.iloc[:,2:]
Xscal = sc.fit_transform(Xscal)
#Xscal_test = sc.transform(Xscal_test)

# One hot encoder applied by cheating by selecting last three columns only
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categories='auto')
Xenc = X.iloc[:,0:2]
Xend = onehotencoder.fit_transform(Xenc).toarray()

# Xp =pd.merge(Xend,Xscal) # can't concatenate/merge np.ndarray to pd.df
# Xp = pd.concat(Xend,Xscal) # only seies and dataframs objs are valid
# Xp = np.concatenate(Xend,Xscal) # only integer scaler arrays can be convertd to scaler index
Xp = np.concatenate((Xend, Xscal), axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xp, y, test_size = 0.2, random_state = 0, stratify=y)

"""
# PCA would not be applied to categorical data 
from sklearn.decomposition import PCA
# pca = PCA(n_components = None)
# pca = PCA(n_components = 2)
pca = PCA(n_components = 4)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
"""

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
# y_pred = classifier.predict(X_test)
# y_test.groupby().size()

# Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# np.unique(y_test) so 5X5 matrix will be generates
sklearn.metrics.plot_confusion_matrix(classifier, X_test, y_test)