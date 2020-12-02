# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 22:24:33 2020

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


#Separating data into 10000 point subsets (7 datasets)
## For cross validation
x_1=X[0:10000,:]
x_2=X[10000:20000,:]
x_3=X[20000:30000,:]
x_4=X[30000:40000,:]
x_5=X[40000:50000,:]
x_6=X[50000:60000,:]
x_7=X_test[0:10000,:]

y_1=labels[0:10000]
y_2=labels[10000:20000]
y_3=labels[20000:30000]
y_4=labels[30000:40000]
y_5=labels[40000:50000]
y_6=labels[50000:60000]
y_7=labels_test[0:10000]

#One-v-all. 10 classifiers (0-9), each one has it's own f_k (w) value
#USING SUPPORT VECTORS:
n_eval = np.size(labels_test)
x_eval_1 = X_test
n_train = np.size(labels)
x_train_1 = X

# Train classifier using linear SVM from SK Learn library
clf = LinearSVC(random_state=0, tol=1e-4,multi_class='ovr',max_iter=10000)
clf.fit(x_train_1, np.squeeze(labels))
w_opt = clf.coef_.transpose()

y_hat_k=x_eval_1@w_opt #find the Xw for this one-vs-rest classifier
labels_classified = np.argmax(y_hat_k,axis=1)
#print(labels_classified) #find the y_hat by argmin of all X*f_k, as given by one-vs-rest
#print(labels_test)

error_vec = [0 if i[0]==i[1] else 1 for i in np.matrix.transpose(np.vstack((labels_classified,labels_test)))]
error = sum(error_vec)
print(error)

