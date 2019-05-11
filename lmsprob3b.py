# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 19:46:57 2019

@author: tchat
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import math

hf1 = h5py.File('D:\EE 599 - Deep Learning\HW 2\lms_fun_v2.hdf5','r')

print(list(hf1.keys()))

matched_10_v = np.array(hf1["matched_10_v"][:])
matched_10_x = np.array(hf1["matched_10_x"][:])
matched_10_y = np.array(hf1["matched_10_y"][:])
matched_10_z = np.array(hf1["matched_10_z"][:])

matched_3_v = np.array(hf1["matched_3_v"][:])
matched_3_x = np.array(hf1["matched_3_x"][:])
matched_3_y = np.array(hf1["matched_3_y"][:])
matched_3_z = np.array(hf1["matched_3_z"][:])
'''
# eta = 0.05 and SNR = 10 dB
eta = 0.25

w1 = np.zeros((502,3))
e1 = np.zeros((502,1))

for i in range(600):
    w = np.zeros((1,3))
    e = np.zeros((1,1))
    weight = np.zeros((1,3))
    
    for j in range(501):
        
        zhat = np.dot(weight,matched_10_v[i,j])
        error = matched_10_z[i,j] - zhat
        weight = weight + eta*error*matched_10_v[i,j]
        w = np.vstack((w,weight))
        err = error**2
        e = np.vstack((e,err))
        
    w1 = w1+w
    e1 = e1+e

w1 = w1/600
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
plt.title('Learning Curve - eta = 0.05 and SNR = 10 dB') 
plt.show()

'''    
# eta = 0.15 and SNR = 10 dB 
eta = 0.25

w1 = np.zeros((502,3))
e1 = np.zeros((502,1))

for i in range(600):
    w = np.zeros((1,3))
    e = np.zeros((1,1))
    weight = np.zeros((1,3))
    
    for j in range(501):
        
        zhat = np.dot(weight,matched_10_v[i,j])
        error = matched_10_z[i,j] - zhat
        weight = weight + eta*error*matched_10_v[i,j]
        w = np.vstack((w,weight))
        err = error**2
        e = np.vstack((e,err))
        
    w1 = w1+w
    e1 = e1+e

w1 = w1/600
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
plt.title('Learning Curve - eta = 0.15 and SNR = 10 dB') 
plt.show()     
'''
# eta = 0.05 and SNR = 3 dB   
eta = 0.05

w1 = np.zeros((502,3))
e1 = np.zeros((502,1))

for i in range(600):
    w = np.zeros((1,3))
    e = np.zeros((1,1))
    weight = np.zeros((1,3))
    
    for j in range(501):
        
        zhat = np.dot(weight,matched_3_v[i,j])
        error = matched_3_z[i,j] - zhat
        weight = weight + eta*error*matched_3_v[i,j]
        w = np.vstack((w,weight))
        err = error**2
        e = np.vstack((e,err))
        
    w1 = w1+w
    e1 = e1+e

w1 = w1/600
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
plt.title('Learning Curve - eta = 0.05 and SNR = 3 dB') 
plt.show()  

# eta = 0.15 and SNR = 3 dB   
eta = 0.15

w1 = np.zeros((502,3))
e1 = np.zeros((502,1))

for i in range(600):
    w = np.zeros((1,3))
    e = np.zeros((1,1))
    weight = np.zeros((1,3))
    
    for j in range(501):
        
        zhat = np.dot(weight,matched_3_v[i,j])
        error = matched_3_z[i,j] - zhat
        weight = weight + eta*error*matched_3_v[i,j]
        w = np.vstack((w,weight))
        err = error**2
        e = np.vstack((e,err))
        
    w1 = w1+w
    e1 = e1+e

w1 = w1/600
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
plt.title('Learning Curve - eta = 0.05 and SNR = 3 dB') 
plt.show()  
'''