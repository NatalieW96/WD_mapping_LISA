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
import time as tm

#Equation to give vector of a LISA arm
def l(phi_b, t, T, alpha0, i):
    alpha_i= 2*np.pi*t/T - np.pi/12 -(i-1)*np.pi/3 + alpha0
    x_b= -np.cos(alpha_i)*np.sin(phi_b) + 0.5*np.sin(alpha_i)*np.cos(phi_b)
    y_b= np.cos(alpha_i)*np.cos(phi_b) + 0.5*np.sin(alpha_i)*np.sin(phi_b)
    z_b= (np.sqrt(3)/2.0)*np.sin(alpha_i)
    return np.transpose(np.array([x_b, y_b, z_b]))

#Equation to give phi_b
def phi_bar(phi0, T, t):
    return phi0 +2*np.pi*t/T

#Function to give response
def response(A0, Omega, t, Phi0, psi, iota, l1, l2, h_ab, alpha, beta):
    h_plus = 0.5*(1+np.cos(iota)**2)*A0*np.sin(Omega*t - Phi0) #plus response from WD
    h_x = np.cos(iota)*A0*np.cos(Omega*t - Phi0)	#cross response from WD
    h_ab[:,0,0] = h_plus
    h_ab[:,0,1] = h_x
    h_ab[:,1,0] = h_x
    h_ab[:,1,1] = -h_plus
    R = np.array([[np.cos(psi),np.sin(psi), 0.0], [-np.sin(psi), np.cos(psi), 0.0], [0,0.0,1.0]])
    h_ab = np.matmul(np.matmul(np.transpose(R),h_ab),R)
    Ry=np.array([[np.cos(beta), 0, -np.sin(beta)],[0,1,0],[np.sin(beta),0,np.cos(beta)]]) #unnecessary?
    Rz=np.array([[np.cos(alpha), np.sin(alpha), 0],[-np.sin(alpha), np.cos(alpha), 0],[0,0,1]])
    R_ec = np.matmul(Ry,Rz) #Euler angle transformation matrix from WD frame into ecliptic frame
    h_ab=np.matmul(np.matmul(np.transpose(R_ec),h_ab),R_ec)
    h_ab3 = np.einsum('kil,lj->kij',np.einsum('ij,kjl->kil',np.transpose(R_ec),h_ab),R_ec) #Rotation of ecliptic
    D_ij = (np.einsum('ki,kj->kij', l1, l1) - np.einsum('ki,kj->kij', l2, l2)) #response at LISA for WD q at time j
    h_t = np.einsum('kij,kij->k',D_ij,h_ab3)
    return h_t

#Function to give shape
def expectation(A0, psi, iota, l1, l2, h_ab, alpha, beta):
    h_plus = 0.5*(1+np.cos(iota)**2)*A0*1/np.sqrt(2) #plus response from WD
    h_x = np.cos(iota)*A0*1/np.sqrt(2)	#cross response from WD
    h_ab[:,0,0] = h_plus
    h_ab[:,0,1] = h_x
    h_ab[:,1,0] = h_x
    h_ab[:,1,1] = -h_plus
    R = np.array([[np.cos(psi),np.sin(psi), 0.0], [-np.sin(psi), np.cos(psi), 0.0], [0,0.0,1.0]])
    h_ab = np.matmul(np.matmul(np.transpose(R),h_ab),R)
    Ry=np.array([[np.cos(beta), 0, -np.sin(beta)],[0,1,0],[np.sin(beta),0,np.cos(beta)]]) #unnecessary?
    Rz=np.array([[np.cos(alpha), np.sin(alpha), 0],[-np.sin(alpha), np.cos(alpha), 0],[0,0,1]])
    R_ec = np.matmul(Ry,Rz) #Euler angle transformation matrix from WD frame into ecliptic frame
    h_ab=np.matmul(np.matmul(np.transpose(R_ec),h_ab),R_ec)
    h_ab3 = np.einsum('kil,lj->kij',np.einsum('ij,kjl->kil',np.transpose(R_ec),h_ab),R_ec) #Rotation of ecliptic
    D_ij = (np.einsum('ki,kj->kij', l1, l1) - np.einsum('ki,kj->kij', l2, l2)) #response at LISA for WD q at time j
    h_t = np.einsum('kij,kij->k',D_ij,h_ab3)
    return h_t

#Function to give SNR
def SNR(h,f):
    alpha=10**(-22.79)*(f/10**-3)**(-7/3)
    beta=10**(-24.54)*(f/10**-3)
    gamma=10**(-23.04)
    noise=5.049e5*(alpha**2+beta**2+gamma**2)
    return h/noise

#Load WD positions and parameter files
with open('WD_positions_1000000_const_iota_psi.sav') as data:
    WD_pos=pickle.load(data)
with open('WD_parameters_1000000_const_iota_psi.sav') as data:
    WD_params=pickle.load(data)

T=3.15e7		#seconds in a year
N=100			#number of time intervals
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
h_I1=np.zeros(N)   #total response interferomenter 1
h_I2=np.zeros(N)   #total response interferomenter 2
h_I3=np.zeros(N)   #total response interferomenter 3
h_ab2=np.zeros((N,3,3))
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
r_WD=WD_pos[3,0]
N_WD=len(WD_pos[0,0]) 	#Number of WDs
h_ab=np.zeros((N_WD,N,3,3))#signal tensor for WDs
#Euler angles for ecliptic transformation of WDs
Z_WD=WD_pos[2]
Z_WD=-Z_WD/np.sqrt(Z_WD[0]**2 + Z_WD[1]**2 + Z_WD[2]**2) #normalised WD Z axis vector in ecliptic frame
alpha = np.arccos(-Z_WD[1]/np.sqrt(1-Z_WD[2]**2)) 
beta = np.arccos(Z_WD[2])
h_a_h_b_exp=np.zeros((3,3,N))
signal=np.zeros((N_WD))

#Fill in l_i vector at each point in time for each arm
for i in np.arange(n):	
    phi_b[:,i]= phi_bar(phi0, T, time) 
    l_i[:,i] = l(phi_b[:,i], time, T, alpha0, i+1)
r_12=-(l_i[0]+l_i[1])#trying to get vector to vertex
r_12=r*r_12/np.linalg.norm(r_12)
C=[R*np.cos(omega*time),R*np.sin(omega*time),z_c]	#coordinates of centre of circle
v_12=C+r_12 	#vertex at arm 1 and 2
v_23=v_12+L*l_i[0]	#vertes at arm 2 and 3
v_31=v_12+L*l_i[1]  #vertex at arm 3 and 1
V=[v_12, v_23, v_31]	#array of vertices

#Go through each WD binary and calulate response at any given time
fig=plt.figure()
for q in np.arange(N_WD):
    h_i1=response(A0[q,0], Omega[q,0], time, Phi0[q,0], psi[q,0], iota[q,0], l_i[:,0], l_i[:,1], h_ab2, alpha[q,0], beta[q,0])
    h_I1 = h_I1 + h_i1 #sum all WD responses at time j
    h_i2=response(A0[q,0], Omega[q,0], time, Phi0[q,0], psi[q,0], iota[q,0], l_i[:,1], l_i[:,2], h_ab2, alpha[q,0], beta[q,0])
    h_I2 = h_I2 + h_i2 #sum all WD responses at time j
    h_i3=response(A0[q,0], Omega[q,0], time, Phi0[q,0], psi[q,0], iota[q,0], l_i[:,2], l_i[:,0], h_ab2, alpha[q,0], beta[q,0])
    h_I3 = h_I3 + h_i3 #sum all WD responses at time j
    h_i1exp=expectation(A0[q,0], psi[q,0], iota[q,0], l_i[:,0], l_i[:,1], h_ab2, alpha[q,0], beta[q,0])
    h_i2exp=expectation(A0[q,0], psi[q,0], iota[q,0], l_i[:,1], l_i[:,2], h_ab2, alpha[q,0], beta[q,0])
    h_i3exp=expectation(A0[q,0], psi[q,0], iota[q,0], l_i[:,2], l_i[:,0], h_ab2, alpha[q,0], beta[q,0])
    signal[q] = SNR(max(h_i1exp), Omega[q,0])
    h_a_h_b_exp= h_a_h_b_exp + np.array([[h_i1exp*h_i1exp,h_i1exp*h_i2exp,h_i1exp*h_i3exp],[h_i2exp*h_i1exp,h_i2exp*h_i2exp,h_i2exp*h_i3exp], [h_i3exp*h_i1exp,h_i3exp*h_i2exp,h_i3exp*h_i3exp]])
#    print '{}: done {}/{}'.format(tm.asctime(),q+1,N_WD)
    if q > 1 and np.log10(q+1)%1 == 0:
        print q+1
        plt.plot(time, h_a_h_b_exp[0,0]/(q+1), alpha=1, label= q+1)
#expectation value calculations
plt.xlabel('Time (s)')
plt.ylabel('<$h^{I}(t)><h^{I}(t)$>')
plt.legend()
plt.show()

pickle.dump(h_a_h_b_exp, open('expectation_values_const_iota_psi.sav', 'wb'))
pickle.dump(signal, open('SNR_{}WDs_const_iota_psi.sav'.format(N_WD), 'wb'))

plt.hist(A0, bins='auto')
plt.show()
exit()

h_a_h_b_exp=h_a_h_b_exp/N_WD
fig = plt.figure()
plt.plot(time, h_a_h_b_exp[0,0])
#for j in np.arange(N):
#    h_mod[j]=np.linalg.norm(h_I[j])

h_f1=np.fft.rfft(h_I1)*dt
h_f2=np.fft.rfft(h_I2)*dt
h_f3=np.fft.rfft(h_I3)*dt
Nf = N//2 + 1
freq=df*np.arange(Nf)
fig = plt.figure()
plt.plot(time, h_I1)
plt.plot(time, h_I2)
plt.plot(time, h_I3)
plt.xlabel('t (s)')
plt.ylabel('h(t)')
fig = plt.figure()
plt.plot(time, h_I1**2)
#plt.plot(time, h_I2**2)
#plt.plot(time, h_I3**2)
plt.xlabel('t (s)')
plt.ylabel('|h(t)|^2')
fig = plt.figure()
plt.loglog(freq, h_f1*np.conj(h_f1))
plt.loglog(freq, h_f2*np.conj(h_f2))
plt.loglog(freq, h_f3*np.conj(h_f3))
plt.xlabel('f (Hz)')
plt.ylabel('|h(f)|^2')
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
