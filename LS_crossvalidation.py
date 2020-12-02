# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 12:06:05 2020

@author: kacie
"""

import numpy as np
import pandas as pd
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

#Step 1: construct label vectors z for each classifier (z[:,#] is for that # number classifier label)
z = np.zeros((len(labels),10))
for i in range(10):
    for j in range(len(labels)):
        if labels[j] == i:
            z[j,i] = 1
        else:
            z[j,i] = 0
            
            
#Step 2: find f_k for each classifier (f_k[:,#] is for that # classifier label)
## USING LEAST SQUARES W/ RIDGE REGRESSION:

lam_val=.01

# RIDGE REGRESSION:
U,s,VT=np.linalg.svd(X,full_matrices=False)
V=np.matrix.transpose(VT)
UT=np.matrix.transpose(U)

f_k=np.zeros((784,10))
#error=np.zeros((10,1))
y_hat_k=np.zeros((10000,10))
d=s/(s**2+lam_val)
D=np.diag(d)
V=np.matrix.transpose(VT)
UT=np.matrix.transpose(U)
for i in range(10):
    z_use = z[:,[i]]
    f_k_hat=V@D@UT@z_use #find the 'w' term for this one-vs-rest classifier
    f_k[:,[i]]=f_k_hat
    y_hat_k[:,[i]]=X_test@f_k_hat #find the Xw for this one-vs-rest classifier

    
labels_classified = np.argmax(y_hat_k,axis=1)
#print(labels_classified) #find the y_hat by argmin of all X*f_k, as given by one-vs-rest
#print(labels_test)

error_vec = [0 if i[0]==i[1] else 1 for i in np.matrix.transpose(np.vstack((labels_classified,labels_test)))]
error = sum(error_vec)
print(error)