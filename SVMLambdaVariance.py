# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:25:12 2020

@author: kacie
"""

import timeit
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

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

X_bias = np.hstack((np.ones((60000,1)),X))
X_test_bias = np.hstack((np.ones((10000,1)),X_test))

start = timeit.default_timer()

#One-v-all USING SUPPORT VECTORS:
n_eval = np.size(labels_test)
n_train = np.size(labels)

c_vals = np.logspace(-6, np.log10(5000), num=25)

sq_err_r=np.zeros((25,1))
err_total=np.zeros((25,1))

for j in range(25):
    c_use = c_vals[j]
    # Train classifier using linear SVM from SK Learn library
    clf = LinearSVC(random_state=0,C=c_use,tol=1e-4,multi_class='ovr',max_iter=20000)
    clf.fit(X_test_bias, np.squeeze(labels_test))
    w_opt = clf.coef_.transpose()

    y_hat_k=X_test_bias@w_opt #find the Xw for this one-vs-rest classifier
    labels_classified = np.argmax(y_hat_k,axis=1)
    #print(labels_classified) #find the y_hat by argmin of all X*f_k, as given by one-vs-rest
    #print(labels_test)

    error_vec = [0 if i[0]==i[1] else 1 for i in np.matrix.transpose(np.vstack((labels_classified,labels_test)))]
    error = sum(error_vec)
    print(error)
    err_total[j]=error
    
    sq_err_r[j] = np.sum(np.square(labels_classified - labels_test))

stop = timeit.default_timer()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(c_vals,sq_err_r)
ax.set_xlabel('$\lambda$', fontsize=16)
ax.set_ylabel('$||Aw*-d||_2^2$', fontsize=16)
ax.set_title('SVM Regularization', fontsize=18)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(c_vals,sq_err_r)
ax.set_xlabel('$\lambda$', fontsize=16)
ax.set_ylabel('$||Aw*-d||_2^2$', fontsize=16)
ax.set_title('SVM Regularization', fontsize=18)
plt.xscale("log")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(c_vals,err_total)
ax.set_xlabel('$\lambda$', fontsize=16)
ax.set_ylabel('Misclassifications', fontsize=16)
ax.set_title('SVM Regularization', fontsize=18)
plt.xscale("log")
plt.show()

print('Time: ', stop - start)  