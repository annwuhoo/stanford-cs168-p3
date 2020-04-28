# Project: CS 168 SP2020 - Miniproject 3 - Question 1
# Author: Ann Wu
# Date: 04/28/2020

import numpy as np
import matplotlib.pyplot as plt
import sys
import csv
from datetime import datetime
from collections import defaultdict
from scipy.sparse.linalg import norm as snorm
from scipy.sparse import coo_matrix
from sklearn.metrics import pairwise as pw

d = 100 # dimensions of data
n = 1000 # number of data points

def total_sq_err(a, X, y):
    return np.sum((np.dot(X,a)-y)**2)

def sq_err(a, X, y):
    return ((np.dot(X,a)-y)**2)

def leastSquares(X, y):
    XT = X.transpose()
    return np.dot(np.dot(np.linalg.inv(np.dot(XT,X)),XT), y)

def gradientDescent(X, y_noise, y_true, alpha):
    niter = 20
    a_gd = np.zeros((d,1))
    for i in range(niter):
        error = sq_err(a_gd, X, y_noise)
        gradient = np.dot(X.transpose(), error)/n
        a_gd = a_gd - alpha*gradient
        print("i:", i, "; total error:",total_sq_err(a_gd, X, y_true))
    return total_sq_err(a_gd, X, y_true)

def SGD(X, y_noise, y_true, alpha):
    niter = 1000
    a_gd = np.zeros((d,1))
    for i in range(niter):
        idx = np.random.randint(d, size=1) # pick a random datapoint
        error = sq_err(a_gd[idx], X[idx], y_noise[idx])
        #gradient = np.dot(X.transpose(), error)/n
        a_gd[idx] = a_gd[idx] - alpha*X[idx]*error
        if (i%100 == 0):
            print("i:", i, "; total error:",total_sq_err(a_gd, X, y_true))
    return total_sq_err(a_gd, X, y_true)

def main():
    # Generate data (given)
    X = np.random.normal(0,1, size=(n,d))
    a_true = np.random.normal(0,1, size=(d,1))
    y_true = X.dot(a_true)
    y_noise = X.dot(a_true) + np.random.normal(0,0.5,size=(n,1))

    # --- Part a
    a_ls = leastSquares(X, y_noise)
    a_zeros = np.zeros((d,1))
    print("err_ls:", total_sq_err(a_ls, X, y_true))
    print("err_zeros:", total_sq_err(a_zeros, X, y_true))

    # --- Part b
    niter = 20
    #err_gd_1 = gradientDescent(X, y_noise, y_true, 0.00005)
    err_gd_2 = gradientDescent(X, y_noise, y_true, 0.0005)
    #err_gd_3 = gradientDescent(X, y_noise, y_true, 0.0007)

    #print("1:", err_gd_1)
    print("2:", err_gd_2)
    #print("3:", err_gd_3)

    #plt.bar([1,2,3], [err_gd_00005,err_gd_0005,err_gd_0007])
    #plt.show()

    # --- Part c


main()
