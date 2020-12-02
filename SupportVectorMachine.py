# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:47:20 2020

@author: kacie
"""

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
#from scipy.sparse import csc_matrix
#from scipy.sparse.linalg import eigs

mnist_train_file = pd.read_csv("fashion-mnist_train.csv")
mnist_test_file = pd.read_csv("fashion-mnist_test.csv")

#Put training data X,y into numpy arrays for manipulation. 10000 data point arrays
labels_column = mnist_train_file.loc[:,'label']
labels = labels_column.to_numpy()

X_columns = mnist_train_file.drop(['label'], axis=1)
X = X_columns.to_numpy()

#Put test data into X,y arrays
labels_column_test = mnist_test_file.loc[:,'label']
labels_test = labels_column_test.to_numpy()

X_columns_test = mnist_test_file.drop(['label'], axis=1)
X_test = X_columns_test.to_numpy()

#One-v-all USING SUPPORT VECTORS:
n_eval = np.size(labels_test)
x_eval_1 = X_test
n_train = np.size(labels)
x_train_1 = X

# Train classifier using linear SVM from SK Learn library
clf = LinearSVC(random_state=0, tol=1e-8,multi_class='ovr',max_iter=2000)
clf.fit(x_train_1, np.squeeze(labels))
w_opt = clf.coef_.transpose()

y_hat_k=x_eval_1@w_opt #find the Xw for this one-vs-rest classifier
labels_classified = np.argmax(y_hat_k,axis=1)
#print(labels_classified) #find the y_hat by argmin of all X*f_k, as given by one-vs-rest
#print(labels_test)

error_vec = [0 if i[0]==i[1] else 1 for i in np.matrix.transpose(np.vstack((labels_classified,labels_test)))]
error = sum(error_vec)
print(error)