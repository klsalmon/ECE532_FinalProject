# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 12:06:05 2020

@author: kacie
"""

import numpy as np
import pandas as pd
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

#One-v-all. 10 classifiers (0-9), each one has it's own f_k (w) value

#Step 1: construct label vectors z for each classifier (z[:,#] is for that # number classifier label)
z = np.zeros((len(labels),10))
for i in range(10):
    for j in range(len(labels)):
        if labels[j] == i:
            z[j,i] = 1
        else:
            z[j,i] = -1
            
            
#Step 2: find f_k for each classifier (f_k[:,#] is for that # classifier label)
## USING LEAST SQUARES W/ RIDGE REGRESSION:

lam_vals = np.logspace(-6, np.log10(5000), num=25)

# RIDGE REGRESSION:
U,s,VT=np.linalg.svd(X_bias,full_matrices=False)
V=np.matrix.transpose(VT)
UT=np.matrix.transpose(U)

sq_err_r=np.zeros((25,1))
err_total=np.zeros((25,1))

for j in range(25):
    print(lam_vals[j])
    f_k=np.zeros((785,10))
    #error=np.zeros((10,1))
    y_hat_k=np.zeros((10000,10))
    d=s/(s**2+lam_vals[j])
    D=np.diag(d)
    V=np.matrix.transpose(VT)
    UT=np.matrix.transpose(U)
    for i in range(10):
        z_use = z[:,[i]]
        f_k_hat=V@D@UT@z_use #find the 'w' term for this one-vs-rest classifier
        f_k[:,[i]]=f_k_hat
        y_hat_k[:,[i]]=X_test_bias@f_k_hat #find the Xw for this one-vs-rest classifier

    
    labels_classified = np.argmax(y_hat_k,axis=1)
    #print(labels_classified) #find the y_hat by argmin of all X*f_k, as given by one-vs-rest
    #print(labels_test)
    
    error_vec = [0 if i[0]==i[1] else 1 for i in np.matrix.transpose(np.vstack((labels_classified,labels_test)))]
    error = sum(error_vec)
    print(error)
    err_total[j]=error
    
    sq_err_r[j] = np.sum(np.square(labels_classified - labels_test))
    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(lam_vals,sq_err_r)
ax.set_xlabel('$\lambda$', fontsize=16)
ax.set_ylabel('$||Aw*-d||_2^2$', fontsize=16)
ax.set_title('Least Squares Regression', fontsize=18)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(lam_vals,sq_err_r)
ax.set_xlabel('$\lambda$', fontsize=16)
ax.set_ylabel('$||Aw*-d||_2^2$', fontsize=16)
ax.set_title('Least Squares Regression', fontsize=18)
plt.xscale("log")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(lam_vals,err_total)
ax.set_xlabel('$\lambda$', fontsize=16)
ax.set_ylabel('Misclassifications', fontsize=16)
ax.set_title('Least Squares Regression', fontsize=18)
plt.xscale("log")
plt.show()