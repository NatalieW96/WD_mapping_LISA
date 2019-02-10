######################################################################################################
#Generates a model of the Milky Way for white dwarf binaries
######################################################################################################

from astropy import units as u
from astropy.coordinates import SkyCoord
import healpy
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import random
import matplotlib.pyplot as plt
import cPickle as pickle
import time as tm

#function of plot the cylinder defined as MW
def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid

#Define Milky Way parameters
h0=1000*9.461e15			#height of MW
r0=100000*9.461e15 		#radius of MW
N=1000000	   		#number of binary WDs

#Sun's parameters
rhoS=25000*9.461e15
thetaS=0.0
zS=500*9.461e15
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
    psi[i]=np.pi/2
    iota[i]=np.pi/2
    phi0[i]=random.uniform(0, 2*np.pi)
    omega[i]=random.uniform(10e-3, 10e-2)
    print '{}: done {}/{}'.format(tm.asctime(),i+1,N)

x=rho*np.cos(theta)	#x coordinates of WDs
y=rho*np.sin(theta)	#y coordinates of WD
xS=rhoS*np.cos(thetaS)	#x coordinate of Sun
yS=rhoS*np.cos(thetaS)	#y coordinate of Sun

#Cartesian coordinates in system where Sun is origin and X-Y plane is galactic plane
X=x-xS
Y=y-yS
Z=z-zS

#Setting Galactic coordinates
r = np.sqrt(X**2 +Y**2 +Z**2)
print r
theta_gal=np.arccos(Z/r)
lon_gal=np.arctan2(Y,X)
lat_gal=np.pi/2-theta_gal

#Setting Ecliptic coordinates
coords_gal=SkyCoord(lon_gal*u.radian, lat_gal*u.radian, frame='galactic', unit= (u.radian, u.radian))
coords_ec=coords_gal.barycentrictrueecliptic
lon_ec=coords_ec.lon.radian
lat_ec=coords_ec.lat.radian
theta_ec=np.pi/2-lat_ec
X_ec=r*np.sin(theta_ec)*np.cos(lon_ec)
Y_ec=r*np.sin(theta_ec)*np.sin(lon_ec)
Z_ec=r*np.cos(theta_ec)

#Amplitude calculation
M_WD=(1.989e30)*0.5	#WD mass
D_WD=10**8	#WD seperation distance
G=6.67e-11
c=3e8
A0=(G*(M_WD**2))/(D_WD*(c**4))
A= A0/r  #gravitational wave amplitude

#Plot 3D graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')
ax.scatter(X, Y, Z)
ax.scatter(0, 0, 0, color = 'red')

#Making axes same length
max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), z.max()-z.min()]).max()
Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())

for xb, yb, zb in zip(Xb, Yb, Zb):
   ax.plot([xb], [yb], [zb], 'w')

#Adding cyclinder defined as Milky Way
#Xc,Yc,Zc = data_for_cylinder_along_z(0,0,r0,h0)
#ax.plot_surface(Xc, Yc, Zc, alpha=0.5)

#Plotting Ecplitc coordinates
fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')
ax.scatter(X_ec, Y_ec, Z_ec)
ax.scatter(0, 0, 0, color = 'red')

#Making axes same length
max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
Xd = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
Yd = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
Zd = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())

for xd, yd, zd in zip(Xd, Yd, Zd):
   ax.plot([xd], [yd], [zd], 'w')


plt.show()

print 'saving files'
positions = np.array([[X, Y, Z],[r, lon_gal, lat_gal],[X_ec, Y_ec, Z_ec], [r, lon_ec, lat_ec]])
pickle.dump(positions, open('WD_positions_{}_const_iota_psi.sav'.format(N), 'wb'))
parameters = np.array([psi, iota, A, omega, phi0]) 
pickle.dump(parameters, open('WD_parameters_{}_const_iota_psi.sav'.format(N), 'wb'))
print 'done'
exit()
