# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 22:45:06 2020

@author: kacie
"""

import timeit
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

X_test_bias = np.hstack((np.ones((10000,1)),X_test))

z = np.zeros((len(labels),10))
for i in range(10):
    for j in range(len(labels)):
        if labels[j] == i:
            z[j,i] = 1
        else:
            z[j,i] = -1
            
            
start = timeit.default_timer()

epoch_vals = np.logspace(0, 4, num=20)

sq_err_r=np.zeros((20,1))
err_total=np.zeros((20,1))
epoch_used=np.zeros((20,1))

for j in range(20):
    epoch_use = int(epoch_vals[j])
    epoch_used[j] = epoch_use
    #Neural Network:
    ## Train NN
    X_bias = np.hstack((np.ones((60000,1)),X))
    X_test_bias = np.hstack((np.ones((10000,1)),X_test))
    q = np.shape(z)[1] #number of classification problems
    M = 20 #number of hidden nodes
    p = 784
    n = int(60000) #examples

    ## initial weights
    V = np.random.randn(M+1, q); 
    W = np.random.randn(p+1, M);

    alpha = 0.1 #step size
    L = epoch_use #number of epochs
    print(epoch_use)
    def logsig(_x):
        return 1/(1+np.exp(-_x))
        
    for epoch in range(L):
        ind = np.random.permutation(n)
        for i in ind:
        # Forward-propagate
            H = logsig(np.hstack((np.ones((1,1)), X_bias[[i],:]@W)))
            Yhat = logsig(H@V)
         # Backpropagate
            delta = (Yhat-z[[i],:])*Yhat*(1-Yhat)
            Vnew = V-alpha*H.T@delta
            gamma = delta@V[1:,:].T*H[:,1:]*(1-H[:,1:])
            Wnew = W - alpha*X_bias[[i],:].T@gamma
            V = Vnew
            W = Wnew
            #print(epoch)
    

    ## Final predicted labels (on training data)
    H = logsig(np.hstack((np.ones((10000,1)), X_test_bias@W)))
    Yhat = logsig(H@V)


    yhat=np.zeros((10000))
    for i in range(10000):
        yhat[i]=np.argmin(Yhat[i,:])
    
    error_vec = [0 if i[0]==i[1] else 1 for i in np.matrix.transpose(np.vstack((np.transpose(yhat),labels_test)))]
    error = sum(error_vec)
    print(error)
    err_total[j]=error
    
    sq_err_r[j] = np.sum(np.square(np.transpose(yhat) - labels_test))

stop = timeit.default_timer()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(epoch_used,sq_err_r)
ax.set_xlabel('$Epoch$', fontsize=16)
ax.set_ylabel('$||Aw*-d||_2^2$', fontsize=16)
ax.set_title('Neural Network Epoch', fontsize=18)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(epoch_used,sq_err_r)
ax.set_xlabel('$Epoch$', fontsize=16)
ax.set_ylabel('$||Aw*-d||_2^2$', fontsize=16)
ax.set_title('eural Network Epoch', fontsize=18)
plt.xscale("log")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(epoch_used,err_total)
ax.set_xlabel('$Epoch$', fontsize=16)
ax.set_ylabel('Misclassifications', fontsize=16)
ax.set_title('eural Network Epoch', fontsize=18)
plt.xscale("log")
plt.show()

print('Time: ', stop - start)  