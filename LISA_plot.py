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
    x_b= -np.cos(alpha_i)*np.sin(phi_b) + 0.5*np.sin(alpha_i)*np.cos(phi_b)
    y_b= np.cos(alpha_i)*np.cos(phi_b) + 0.5*np.sin(alpha_i)*np.sin(phi_b)
    z_b= (np.sqrt(3)/2.0)*np.sin(alpha_i)
    return [x_b, y_b, z_b]

#Equation to give phi_b
def phi_bar(phi0, T, t):
    return phi0 +2*np.pi*t/T

#Function to give response
def response(A0, Omega, t, Phi0, psi, iota, l1, l2):
    h_plus = A0*np.sin(Omega*t - Phi0) #plus response from WD
    h_x = A0*np.cos(Omega*t - Phi0)	#cross response from WD
    h_ab= np.array([[h_plus, h_x, 0],[h_x, -h_plus, 0], [0, 0, 0]]) #response tensor
    R=np.array([[psi,iota, 0.0], [-iota, psi, 0.0], [0,0.0,1.0]])
    h_ab= np.transpose(R)*h_ab*R
    h_i = 0.5*h_ab*(np.outer(l1,l1)- np.outer(l2,l2)) #response at LISA for WD q at time j
    return h_i

#Load WD positions and parameter files
with open('WD_positions.sav') as data:
    WD_pos=pickle.load(data)
with open('WD_parameters.sav') as data:
    WD_params=pickle.load(data)

T=3.15e7		#seconds in a year
N=1000			#number of time intervals
dt=T/N			#time interval
df=1.0/T		#frequency interval
phi0=0			#starting phi value	
alpha0=0		#starting alpha value
R=1.496e11		#earth sun distance
L=5.0e10		#length of LISA arms
r=(L/2.0)/np.cos(np.pi/6.0) #length of vector to vertex
phi_b=np.zeros((N,3))	#phi_b array
l_i=np.zeros((N,3,3))	#l_i array
V=np.zeros((N,3,3))	#vertices coordinates
h_I=np.zeros((N,3,3))   #total response
h_mod=np.zeros((N))	#modulus of response
n=3			#number of arms
theta_b=np.pi/2		
omega=2*np.pi/T		#angular speed of orbit
time = np.arange(0,T,dt)#time array
z_c=0.0			#LISA orbit plane in x, y
#WD parameters
psi=WD_params[0] 
iota=WD_params[1]
A0=WD_params[2]
Omega=WD_params[3]
Phi0=WD_params[4]
N_WD=len(WD_pos[0,0]) 	#Number of WDs
h_ab=np.zeros((N_WD,N,3,3))#signal tensor for WDs
h_i=np.zeros((3,3))	#Individual response of WD binary on LISA


#Fill in l_i vector at each point in time for each arm
for j,t in enumerate(time):
    for i in np.arange(n):	
	phi_b[j,i]= phi_bar(phi0, T, t) 
	l_i[j,i] = l(phi_b[j,i], t, T, alpha0, i+1)
    r_12=-(l_i[j,0]+l_i[j,1])#trying to get vector to vertex
    r_12=r*r_12/np.linalg.norm(r_12)
    C=[R*np.cos(omega*t),R*np.sin(omega*t),z_c]	#coordinates of centre of circle
    v_12=C+r_12 	#vertex at arm 1 and 2
    v_23=v_12+L*l_i[j,0]	#vertes at arm 2 and 3
    v_31=v_12+L*l_i[j,1]  #vertex at arm 3 and 1
    V[j,:]=[v_12, v_23, v_31]	#array of vertices
    
    #Go through each WD binary and calulate response at any given time
    for q in np.arange(N_WD):
	h_i=response(A0[q,0], Omega[q,0], t, Phi0[q,0], psi[q,0], iota[q,0], l_i[j,0], l_i[j,1])
	h_I[j] = h_I[j] + h_i #sum all WD responses at time j
    h_mod[j] = np.linalg.norm(h_I[j]) #modulus od response tensor at time 

h_modf=np.fft.rfft(h_mod)*dt
freq=np.arange(0, df*len(h_modf), df)
fig = plt.figure()
plt.plot(time, h_mod)
plt.xlabel('t (s)')
plt.ylabel('|h|')
plt.show()

exit()
#orbit coordinates
u = np.linspace(0,  2*np.pi, 100)
x = R*np.cos(u)
y = R*np.sin(u)


a=3 			#iteration to plot

#plot LISA with orbit
fig = plt.figure()
tri = [V[a,0],V[a,1],V[a,2]]	#triangle from vertex points
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')
ax.scatter(V[a,0,0],V[a,0,1],V[a,0,2],'r') #plot vertex points
ax.scatter(V[a,1,0],V[a,1,1],V[a,1,2],'b')
ax.scatter(V[a,2,0],V[a,2,1],V[a,2,2],'g')
ax.add_collection3d(Poly3DCollection([tri], alpha=0.1, edgecolor='black'))
ax.plot(x,y,0)
#ax.scatter(0,0,0)

# Create cubic bounding box to simulate equal aspect ratio
max_range = np.array([V[:,:,0].max()-V[:,:,0].min(), V[:,:,1].max()-V[:,:,1].min(), V[:,:,2].max()-V[:,:,2].min()]).max()
Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(V[:,:,0].max()+V[:,:,0].min())
Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(V[:,:,1].max()+V[:,:,1].min())
Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(V[:,:,2].max()+V[:,:,2].min())
# Comment or uncomment following both lines to test the fake bounding box:
for xb, yb, zb in zip(Xb, Yb, Zb):
   ax.plot([xb], [yb], [zb], 'w')

plt.grid()
plt.show()


exit()

