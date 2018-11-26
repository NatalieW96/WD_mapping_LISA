######################################################################################################
#Generates a model of the Milky Way for white dwarf binaries
######################################################################################################

import healpy
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import random
import matplotlib.pyplot as plt
import cPickle as pickle

#function of plot the cylinder defined as MW
def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid

#Define Milky Way parameters
h0=1000			#height of MW
r0=100000  		#radius of MW
N=1000   		#number of binary WDs

#Sun's parameters
rhoS=25000
thetaS=0.0
zS=500
beta= np.pi*60.2/180 #angle of ecliptic

#Initialise coordinates
theta=np.zeros((N,1))
rho=np.zeros((N,1))
z=np.zeros((N,1))
x=np.zeros((N,1))
y=np.zeros((N,1))
#Initialise parameters
psi=np.zeros((N,1))
iota=np.zeros((N,1))
A0=np.zeros((N,1))
phi0=np.zeros((N,1))
omega=np.zeros((N,1))


#Generate random coordinates
for i in np.arange(N):
    theta[i]=random.uniform(0, 2*np.pi)
    rho[i]=r0*np.sqrt(random.uniform(0,1.0))
    z[i]=random.uniform(0,h0)
    psi[i]=random.uniform(0, 2*np.pi)
    iota[i]=random.uniform(0, 2*np.pi)
    phi0[i]=random.uniform(0, 2*np.pi)
    omega[i]=random.uniform(10e-3, 10e-2)
    
x=rho*np.cos(theta)	#x coordinates of WDs
y=rho*np.sin(theta)	#y coordinates of WD
xS=rhoS*np.cos(thetaS)	#x coordinate of Sun
yS=rhoS*np.cos(thetaS)	#y coordinate of Sun

#Cartesian coordinaes in system where Sun is origin and X-Y plane is galactic plane
X=x-xS
Y=y-yS
Z=z-zS

#Cartesian coordinaes in system where Sun is origin and X-Y plane is ecliptic plane
R=[[1, 0, 0],[0, np.cos(beta), np.sin(beta)],[0, -np.sin(beta), np.cos(beta)]] #Rotational matrix using Euler angles 
coords=np.transpose(np.array([X,Y,Z])).reshape(1000,3,1) #vectors of X,Y,Z
ecliptic=np.transpose(np.einsum("ij,kjl->kil",R,coords)) #Batch matric multipication to get new X, Y, Z

#Setting Galactic coordinates
r = np.sqrt(X**2 +Y**2 +Z**2)
theta_gal= np.arccos(Z/r)
phi_gal = np.arctan(Y/X) 

#Setting Ecliptic coordinates
theta_ec= np.array([np.arccos(ecliptic[0][2]/r)])
phi_ec= np.array([np.arctan(ecliptic[0][1]/ecliptic[0][0])])

A0= 1.0/r #gravitational wave amplitude

#Plot 3D graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')
ax.scatter(x, y, z)
ax.scatter(xS, yS, zS, color = 'red')

#Making axes same length
max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(x.max()+x.min())
Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(y.max()+y.min())
Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(z.max()+z.min())

for xb, yb, zb in zip(Xb, Yb, Zb):
   ax.plot([xb], [yb], [zb], 'w')

#Adding cyclinder defined as Milky Way
Xc,Yc,Zc = data_for_cylinder_along_z(0,0,r0,h0)
ax.plot_surface(Xc, Yc, Zc, alpha=0.5)

#Plotting Ecplitc coordinates
fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')
ax.scatter(ecliptic[0][0], ecliptic[0][1],ecliptic[0][2])
ax.scatter(0, 0, 0, color = 'red')

#Making axes same length
max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
Xd = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
Yd = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
Zd = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())

for xd, yd, zd in zip(Xd, Yd, Zd):
   ax.plot([xd], [yd], [zd], 'w')


plt.show()

positions = np.array([[X, Y, Z],[r, theta_gal, phi_gal], [r, theta_ec, phi_ec]])
pickle.dump(positions, open('WD_positions.sav', 'wb'))
parameters = np.array([psi, iota, A0, omega, phi0]) 
pickle.dump(parameters, open('WD_parameters.sav', 'wb'))
exit()
