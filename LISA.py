######################################################################################################
#Simulates LISAs movement around the Sun
######################################################################################################

import numpy as np
from matplotlib import cm
import random
import cPickle as pickle
from matplotlib import pyplot as plt
from matplotlib import animation
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

#Equation to give vector of a LISA arm
def l(phi_b, t, T, alpha0, i):
    alpha_i= 2*np.pi*t/T - np.pi/12 -(i-1)*np.pi/3 + alpha0
    x_b= -np.cos(alpha_i)*np.sin(phi_b) + 1/2 * np.sin(alpha_i)*np.cos(phi_b)
    y_b= np.cos(alpha_i)*np.cos(phi_b) + np.sin(alpha_i)*np.cos(phi_b)
    z_b= np.sqrt(3)/2 *np.sin(alpha_i)
    return [x_b, y_b, z_b]

#Equation to give phi_b
def phi_bar(phi0, T, t):
    return phi0 +2*np.pi*t/T

def init():
    r = 1.49e11
    x = np.linspace(-r,r,1000)
    y = np.sqrt(-x**2+r**2)
    plt.plot(x, y,'b')
    plt.plot(x,-y,'b')


T=3.15e7		#seconds in a year
N=10			#number of time intervals
dt=T/N			#time interval
phi0=0			#starting phi value	
alpha0=0		#starting alpha value
R=1.496e11		#earth sun distance
L=5.0e6
r=L/2 * np.sin(np.pi/12)
phi_b=np.zeros((N,3))
l_i=np.zeros((N,3,3))
V=np.zeros((N,3,3))	#vertices coordinates
n=3			#number of arms
theta_b=np.pi/2		
omega=2*np.pi/T		#angular speed of orbit
time = np.arange(0,T,dt)#time array
z_c=0

#Fill in l_i vector at each point in time for each arm
for j,t in enumerate(time):
    for i in np.arange(n):	
	phi_b[j,i]= phi_bar(phi0, T, t)
	l_i[j,i] = l(phi_b[j,i], t, T, alpha0, i+1)
    r_12=-(l_i[j,0]+1/2* l_i[j,1])#trying to get vector to vertex
    r_12=r*r_12/np.linalg.norm(r_12)	
    C=[R*np.cos(omega*t),R*np.sin(omega*t),z_c]	#coordinates of centre of circle
    v_12=C+r_12 	#vertex at arm 1 and 2
    v_23=v_12+l_i[j,0]	#vertes at arm 2 and 3
    v_31=v_12+l_i[j,1]  #vertex at arm 3 and 1
    V[j,:]=[v_12, v_23, v_31]

#orbit coordinates
u = np.linspace(0,  2*np.pi, 100)
x = R*np.cos(u)
y = R*np.sin(u)

#plot LISA with orbit
fig = plt.figure()
tri = zip(V[1,:,0],V[1,:,1],V[1,:,2])
ax = fig.add_subplot(111, projection='3d')
ax.scatter(V[1,:,0],V[1,:,1],V[1,:,2], 'go-')
ax.add_collection3d(Poly3DCollection([tri], facecolors='w', edgecolor='black'))
#ax.plot(x,y,0)
plt.show()


exit()
#animation

#set up figure, 
#fig=plt.figure()
#ax = fig.add_subplot(111, projection='3d')

#anim = animation.FuncAnimation(fig, update_LISA, 25, fargs=(data, lines),
#                                   interval=50, blit=False)
exit()

