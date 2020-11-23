#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 10:21:22 2020

@author: julia
"""
import pandas as pd
import numpy as np
import sklearn
import urllib3
import matplotlib.pyplot as plt


# =============================================================================
# urlinfo = 'https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.names'
# pool = urllib3.PoolManager()
# resp = pool.request('GET',urlinfo )
# f = open('habermanz.info', 'wb')
# f.write(resp.data)
# f.close()
# resp.release_conn()
# 
# urldata = 'https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data'
# pool = urllib3.PoolManager()
# resp = pool.request('GET',urldata )
# f = open('habermanz.data', 'wb')
# f.write(resp.data)
# f.close()
# resp.release_conn()
# 
# =============================================================================

# df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data')

df = pd.read_csv('habermanz.data')
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# test train split
from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X,y, stratify = y, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(X_train)
sc.transform(X_test)

# feature selection
# =============================================================================
# from sklearn.decomposition import PCA
# pca = PCA(n_components=None)
# X_train = pca.fit_transform(X_train)
# explanation = pca.explained_variance_ratio_
# X_test = pca.transform(X_test)
# pca.singular_values_
# pca.get_covariance()
# print(pca.get_precision())
# 
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure(1, figsize=(8, 6))
# ax = Axes3D(fig, elev=-150, azim=110)
# X_reduced = PCA(n_components=3).fit_transform(df)
# ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
#            cmap=plt.cm.Set1, edgecolor='k', s=40)
# ax.set_title("First three PCA directions")
# ax.set_xlabel("1st eigenvector")
# ax.w_xaxis.set_ticklabels([])
# ax.set_ylabel("2nd eigenvector")
# ax.w_yaxis.set_ticklabels([])
# ax.set_zlabel("3rd eigenvector")
# ax.w_zaxis.set_ticklabels([])
# plt.show()
# =============================================================================

""" this only works if you have more than 3 features
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_train)
"""
# confusing matrix
from sklearn.metrics import ConfusionMatrixDisplay as cdm
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression(random_state=0)
logReg.fit(X_train, y_train)
y_logReg = logReg.predict(X_test)
cm = confusion_matrix(y_test, y_logReg)
cdm(cm).plot()

from sklearn.cluster import KMeans
# kmeanpp = KMeans(init='k-means++', n_clusters = len(y_train.unique()))
kmeanpp = KMeans(init='k-means++', n_clusters = 2)
kmeanpp.fit(X_train)
y_kmean = kmeanpp.predict(X_test)
cm_kmean = confusion_matrix(y_test,y_kmean)
cdm(cm_kmean).plot()


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
y_knn = neigh.predict(X_test)
cm_knn = confusion_matrix(y_test,y_knn)
cdm(cm_knn).plot()

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_train, y_train)
y_dtree = dtree.predict(X_test)
cm_dtree = confusion_matrix(y_test, y_dtree)
cdm(cm_dtree).plot()
from sklearn import tree
tree.plot_tree(dtree)

from sklearn.ensemble import RandomForestClassifier
rmf = RandomForestClassifier(n_estimators=300)
rmf.fit(X_train, y_train)
y_forest = rmf.predict(X_test)
cm_forest = confusion_matrix(y_test, y_forest)
cdm(cm_forest).plot()

from sklearn.svm import SVC
svm = SVC(C=1, kernel='rbf', gamma=.1)
svm.fit(X_train, y_train)
y_svm = svm.predict(X_test)
cm_svc = confusion_matrix(y_test, y_forest)
cdm(cm_svc).plot()

