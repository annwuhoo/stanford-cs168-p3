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

def leastSquares(X, y):
    XT = X.transpose()
    return np.dot(np.dot(np.linalg.inv(np.dot(XT,X)),XT), y)

def gradientDescent(X, y_noise, alpha):
    niter = 20
    a = np.zeros((d,1))
    ret_lst = []
    for i in range(niter):
        # 2 * xT * (aTx - y)
        gradient = 2 * (X.T).dot( ((a.T).dot(X.T) - y_noise.T ).T)
        a = a - alpha*gradient
        ret_lst.append(total_sq_err(a, X, y_noise))
    return ret_lst

def SGD(X, y_noise, alpha):
    niter = 1000
    a = np.zeros((d,1))
    ret_lst = []
    for i in range(niter):
        idx = np.random.randint(d, size=1) # pick a random datapoint
        # 2 * x[idx] * (a[idx]x[idx] - y[idx])
        gradient = 2 * ((a.T).dot(X[idx].T) - y_noise[idx])
        a[idx] = a[idx] - alpha*gradient
        ret_lst.append(total_sq_err(a, X, y_noise))
    return ret_lst

def main():
    # Generate data (given)
    X = np.random.normal(0,1, size=(n,d))
    a_true = np.random.normal(0,1, size=(d,1))
    y_true = X.dot(a_true)
    y_noise = X.dot(a_true) + np.random.normal(0,0.5,size=(n,1))

    # --- Part a
    a_ls = leastSquares(X, y_noise)
    a_zeros = np.zeros((d,1))
    print("err_ls:", total_sq_err(a_ls, X, y_noise))
    print("err_zeros:", total_sq_err(a_zeros, X, y_noise))

    # --- Part b
    gd1 = gradientDescent(X, y_noise, 0.00005)
    gd2 = gradientDescent(X, y_noise, 0.0005)
    gd3 = gradientDescent(X, y_noise, 0.0007)
    #print("1:", gd1)
    #print("2:", gd2)
    #print("3:", gd3)
    t = np.arange(0, 20, 1)
    plt.semilogy(t, gd1, 'r--', label="alpha=0.00005")
    plt.semilogy(t, gd2, 'bs', label="alpha=0.0005")
    plt.semilogy(t, gd3, 'g^', label="alpha=0.0007")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=3, mode="expand", borderaxespad=0.)
    #plt.title("Gradient Descent")
    plt.xlabel("Iterations")
    plt.ylabel("Convergence")
    plt.show()

    # --- Part c
    gd1 = SGD(X, y_noise, 0.0005)
    gd2 = SGD(X, y_noise, 0.005)
    gd3 = SGD(X, y_noise, 0.01)
    #print("1:", gd1)
    #print("2:", gd2)
    #print("3:", gd3)
    t = np.arange(0, 1000, 1)
    plt.semilogy(t, gd1, 'r--', label="alpha=0.0005")
    plt.semilogy(t, gd2, 'bs', label="alpha=0.005")
    plt.semilogy(t, gd3, 'g^', label="alpha=0.01")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=3, mode="expand", borderaxespad=0.)
    #plt.title("Gradient Descent")
    plt.xlabel("Iterations")
    plt.ylabel("Convergence")
    plt.show()

main()
