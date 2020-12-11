# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 12:06:05 2020

@author: kacie
"""

import numpy as np
import pandas as pd
from itertools import permutations
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

X_bias = np.hstack((np.ones((60000,1)),X))
X_test_bias = np.hstack((np.ones((10000,1)),X_test))

#Separating data into 10000 point subsets (7 datasets)
## For cross validation
x_1=X_bias[0:10000,:]
x_2=X_bias[10000:20000,:]
x_3=X_bias[20000:30000,:]
x_4=X_bias[30000:40000,:]
x_5=X_bias[40000:50000,:]
x_6=X_bias[50000:60000,:]
x_7=X_test_bias[0:10000,:]

y_1=labels[0:10000]
y_2=labels[10000:20000]
y_3=labels[20000:30000]
y_4=labels[30000:40000]
y_5=labels[40000:50000]
y_6=labels[50000:60000]
y_7=labels_test[0:10000]

## Cross validation
count=[0,1,2,3,4,5,6]
perm=permutations(count,2)
number_42=0
error_tot_v=np.zeros(42)

for i in list(perm):
    x_list = [x_1,x_2,x_3,x_4,x_5,x_6,x_7]
    x_train=x_list
    x_choosew=x_list[i[0]]
    x_testerror=x_list[i[1]]
    
    y_list = [y_1,y_2,y_3,y_4,y_5,y_6,y_7]
    y_train=y_list
    y_choosew=y_list[i[0]]
    y_testerror=y_list[i[1]]

    for index in sorted(i, reverse=True):
        del x_train[index]
        del y_train[index]
        
    X_train=np.vstack(x_train)
    Y_train=np.vstack(y_train)
    
    
    #One-v-all. 10 classifiers (0-9), each one has it's own f_k (w) value

    #Step 1: construct label vectors z for each classifier (z[:,#] is for that # number classifier label)
    z = np.zeros((len(Y_train),10))
    for x in range(10):
        for j in range(len(Y_train)):
            if labels[j] == x:
                z[j,x] = 1
            else:
                z[j,x] = 0
            
            
    #Step 2: find f_k for each classifier (f_k[:,#] is for that # classifier label)
    ## USING LEAST SQUARES W/ RIDGE REGRESSION:
    
    lam_val=[0,0.1,0.5,1,2,10,200]

    # RIDGE REGRESSION:
    U,s,VT=np.linalg.svd(X_train,full_matrices=False)
    V=np.matrix.transpose(VT)
    UT=np.matrix.transpose(U)
    W=[] #list of f_ks that can be used
   # W.append(i)

    error=np.zeros((7,1))

    for i in range(7):
        lam_use=lam_val[i]
        f_k=np.zeros((784,10))
        #error=np.zeros((10,1))
        y_hat_k=np.zeros((10000,10))
        d=s/(s**2+lam_use)
        D=np.diag(d)
        for j in range(10):
            z_use = z[:,[j]]
            f_k_hat=V@D@UT@z_use #find the 'w' term for this one-vs-rest classifier
            f_k[:,[j]]=f_k_hat
            y_hat_k[:,[j]]=x_choosew@f_k_hat #find the Xw for this one-vs-rest classifier

    
        labels_classified = np.argmax(y_hat_k,axis=1)
        #print(labels_classified) #find the y_hat by argmin of all X*f_k, as given by one-vs-rest
        #print(labels_test)

        error_vec = [0 if i[0]==i[1] else 1 for i in np.matrix.transpose(np.vstack((labels_classified,y_choosew)))]
        error[i] = sum(error_vec)
        print(error)
        W.append(f_k)
        
    fk_chosen=W[i[np.argmin(error)]]
    y_hat_test=np.zeros((10000,10))
    for j in range(10):
        f_k_hat=fk_chosen[:,[j]]
        y_hat_test[:,[j]]=x_testerror@f_k_hat #find the Xw for this one-vs-rest classifier
        
    labels_classified = np.argmax(y_hat_test,axis=1)
    error_vec_final = [0 if i[0]==i[1] else 1 for i in np.matrix.transpose(np.vstack((labels_classified,y_testerror)))]
    
    error_rate=sum(error_vec_final)/16
    error_tot_v[number_42]=error_rate
    number_42=number_42+1
    
#averaging errors from all permutations (42)
print('Average Error for Ridge Regression=',np.round(sum(error_tot_v)/42,3))
print('Average Error for Ridge Regression (Percentage)=',np.round(100*sum(error_tot_v)/42,3),'%')