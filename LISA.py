######################################################################################################
#Simulates LISAs movement around the Sun
######################################################################################################

import numpy as np
from matplotlib import cm
import random
import matplotlib.pyplot as plt
import cPickle as pickle

def l(x_b, y_b, z_b, phi_b, t, T, alpha0, i):
    alpha_i= 2*np.pi*t/T - np.pi/12 -(i-1)*np.pi/3 + alpha0
    return np.cos(alpha_i)*(np.cos(phi_b*y_b)-np.sin(phi_b*x_b)) + np.sin(alpha_i)*(np.sqrt(3)/2 *z_b + 1/2 *(np.cos(phi_b*x_b)+ np.sin(phi_b*y_b)))

def phi_bar(phi0, T, t):
    return phi0 +2*np.pi*t/T

def z(phi_b,x_b,y_b):
    return -np.sqrt(3)/2 * (np.cos(phi_b*x_b)+np.sin(phi_b*y_b))

T=3.15e7		#seconds in a year
N=100			#number of time intervals
dt=T/N			#time interval
phi0=0	
alpha0=0
R=1.496e11		#earth sun distance
phi_b=np.zeros((N,3))
l_i=np.zeros((N,3))
n=3			#number of arms
theta_b=np.pi/2
omega=2*np.pi/T
time = np.arange(0,T,dt)

for j,t in enumerate(time):
    x_b= R*np.sin(omega*t)
    y_b= R*np.cos(omega*t)
    z_b=0
    for i in np.arange(n):
	print j, i	
	phi_b[j,i]= phi_bar(phi0, T, t)
	l_i[j,i] = l(x_b, y_b, z_b, phi_b[j,i], t, T, alpha0, i)

exit()

