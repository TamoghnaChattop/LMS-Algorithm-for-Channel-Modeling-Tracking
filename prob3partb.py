# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 13:53:13 2019

@author: tchat
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import math

hf1 = h5py.File('D:\EE 599 - Deep Learning\HW 2\lms_fun_v2.hdf5','r')

print(list(hf1.keys()))

time_varying_coefficents = np.array(hf1["timevarying_coefficents"][:])
time_varying_v = np.array(hf1["timevarying_v"][:])
time_varying_x = np.array(hf1["timevarying_x"][:])
time_varying_y = np.array(hf1["timevarying_y"][:])
time_varying_z = np.array(hf1["timevarying_z"][:])

time_varying_x = np.reshape(time_varying_x,(501,1))
time_varying_y = np.reshape(time_varying_y,(503,1))
time_varying_z = np.reshape(time_varying_z,(503,1))

time_varying_z = time_varying_z[:-2,:]

x = np.arange(501)
plt.figure()
plt.plot(x,time_varying_coefficents[:,0],'r--', x,time_varying_coefficents[:,1],'b--', x,time_varying_coefficents[:,2],'g--')
plt.ylabel('time_varying_coefficents')
plt.xlabel('time(n)')
plt.title('True Coefficients')
plt.show()

eta = 0.05

w = np.zeros((1,3))
e = np.zeros((1,1))
weight = np.zeros((1,3))

for i in range(501):
    
    zhat = np.dot(weight,time_varying_v[i,:])
    error = time_varying_z[i,:] - zhat
    weight = weight + eta*error*time_varying_v[i,:]
    w = np.vstack((w,weight))
    err = error**2
    e = np.vstack((e,err))
    
e1 = 10*np.log10(e)    

x = np.arange(502)

plt.figure()
plt.plot(x,w[:,0],'r--', x,w[:,1],'b--', x,w[:,2],'g--')
plt.ylabel('w-estimates')
plt.xlabel('updates')
plt.title('Estimated Coefficients')
plt.show()

plt.figure()
plt.plot(x,e1[:,0],'r--')
plt.ylabel('MSE(dB)')
plt.xlabel('updates')
plt.title('Learning Curve') 
plt.show()

