# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 14:15:24 2019

@author: tchat
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import math

hf1 = h5py.File('D:\EE 599 - Deep Learning\HW 2\lms_fun-v2.hdf5','r')

print(list(hf1.keys()))

mismatched_v = np.array(hf1["mismatched_v"][:])
mismatched_x = np.array(hf1["mismatched_x"][:])
mismatched_y = np.array(hf1["mismatched_y"][:])

eta = 0.15

w1 = np.zeros((502,3))
e1 = np.zeros((502,1))

for i in range(600):
    w = np.zeros((1,3))
    e = np.zeros((1,1))
    weight = np.zeros((1,3))
    
    for j in range(501):
        
        zhat = np.dot(weight,mismatched_v[i,j])
        error = mismatched_y[i,j] - zhat
        weight = weight + eta*error*mismatched_v[i,j]
        w = np.vstack((w,weight))
        err = error**2
        e = np.vstack((e,err))
        
    w1 = w1+w
    e1 = e1+e

w1 = w1/600
error = e1/600
e1 = 10*np.log10(e1/600)
x = np.arange(502)

plt.figure()
plt.plot(x,w[:,0],'r--', x,w[:,1],'b--', x,w[:,2],'g--')
plt.ylabel('w-estimates')
plt.xlabel('updates')
plt.title('Coefficients - One Run')
plt.show()

plt.figure()
plt.plot(x,w1[:,0],'r--', x,w1[:,1],'b--', x,w1[:,2],'g--')
plt.ylabel('w-estimates')
plt.xlabel('updates')
plt.title('Coefficients(averaged)')
plt.show()

plt.figure()
plt.plot(x,e1[:,0],'r--')
plt.ylabel('MSE(dB)')
plt.xlabel('updates')
plt.title('Learning Curve - eta = 0.15') 
plt.show()

# Rvn

mismatchedv = np.reshape(mismatched_v,(300600,3))
Rvn = (np.matmul(mismatchedv.T,mismatchedv))/300600
print(Rvn)     

# rn

mismatchedy = np.reshape(mismatched_y,(300600,1))
rn = (np.matmul(mismatchedy.T,mismatchedv))/300600
print(rn)

# LLSE

wllse = np.matmul(rn,np.linalg.inv(Rvn))
Ry = (np.matmul(mismatchedy.T,mismatchedy))/300600
Rxy = (np.matmul(mismatchedv.T,mismatchedy))/300600
final = Ry - np.matmul(wllse,Rxy)
print(final)