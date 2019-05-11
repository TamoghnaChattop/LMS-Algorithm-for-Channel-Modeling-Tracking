# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 13:44:51 2019

@author: tchat
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import math

hf1 = h5py.File('D:\EE 599 - Deep Learning\HW 2\lms_fun-v2.hdf5','r')

print(list(hf1.keys()))

matched_10_v = np.array(hf1["matched_10_v"][:])
matched_10_x = np.array(hf1["matched_10_x"][:])
matched_10_y = np.array(hf1["matched_10_y"][:])
matched_10_z = np.array(hf1["matched_10_z"][:])
matched_3_v = np.array(hf1["matched_3_v"][:])
matched_3_x = np.array(hf1["matched_3_x"][:])
matched_3_y = np.array(hf1["matched_3_y"][:])
matched_3_z = np.array(hf1["matched_3_z"][:])
mismatched_v = np.array(hf1["mismatched_v"][:])
mismatched_x = np.array(hf1["mismatched_x"][:])
mismatched_y = np.array(hf1["mismatched_y"][:])

#weight = np.random.rand(1,3) 
weight = np.zeros((1,3))
w = np.zeros((1,3))
e = np.zeros((1,1))
eta = 0.05

# eta = 0.05 and SNR = 10 dB

for i in range(600):
    esq = 0
    for j in range(501):
        
        zhat = np.dot(weight,matched_3_v[i,j])
        error = matched_3_z[i,j] - zhat
        weight = weight + eta*error*matched_3_v[i,j]
        esq = esq + error**2
    w = np.vstack((w,weight))
    esq = esq/501
    e = np.vstack((e,esq))
    
#w = np.delete(w, (0), axis=0)       
#e = np.delete(e, (0), axis=0)
x = np.arange(601)
e = 10*np.log10(e)

plt.figure()
plt.plot(x,w[:,0],'r--', x,w[:,1],'b--', x,w[:,2],'g--')
plt.ylabel('w-estimates')
plt.xlabel('updates')
plt.title('Coefficients(averaged)')
plt.show()

plt.figure()
plt.plot(x,e[:,0],'r--')
plt.ylabel('MSE(dB)')
plt.xlabel('updates')
plt.title('Learning Curve') 
plt.show()
        