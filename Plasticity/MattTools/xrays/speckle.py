import numpy as np
import scipy as sp
import pylab as pl
import matplotlib as mpl
from Plasticity import TarFile
from collections import defaultdict
import PIL

FF = 1

# create some standard tensors
delta = np.eye(3, dtype='int')
levi  = np.array([[[-(i-j)*(j-k)*(k-i)/2 for i in range(3)] for j in range(3)] for k in range(3)])

# make sure we made delta, levi-civita correctly
assert (np.einsum('imn,jmn->ij', levi, levi) == 2*delta).all()
assert (np.einsum('ijk,imn->jkmn', levi, levi) == \
        (np.einsum('jm,kn->jkmn', delta, delta) - np.einsum('jn,km->jkmn', delta, delta))).all()


mu,nu = 0.5,0.3
lamb = 2.*mu*nu/(1.-2.*nu)
def ExternalStrain(sigma,primaryStrain):
    x,y,z = 'x','y','z'
    strains = {x:primaryStrain[0],y:primaryStrain[1],z:primaryStrain[2]}
    strain_trace = strains[x]+strains[y]+strains[z]
    for i in [x,y,z]:
        sigma[i,i] += strains[i] #lamb*strain_trace + 2.*mu*strains[i]
    return sigma

def CalculateTrueStrain(sig, rate, t, initial=np.array([0,0,0])):
    return ExternalStrain(sig, initial + rate*t)

#==========================================================
# these functions apply to the lattice vectors 
# before casting them across hkl's to form real Bragg peaks
#==========================================================
def generateHKLs(largest=4):
    import itertools
    return ( np.array(x) for x in itertools.product(xrange(-largest,largest+1), repeat=3))

def generateLatticeVectors(lattice='fcc', a0=3.62, shape=(1,)):
    if lattice=='fcc':
        a = a0/2 * np.array([[0,1,1],[1,0,1],[1,1,0]]).T 
    else:
        a = a0/2 * np.eye(3)

    return np.squeeze( np.ones(shape)[..., np.newaxis, np.newaxis] * a )

def strainLattice(a, eps):
    return np.einsum('jk,XYik->XYij', delta,a) + np.einsum('XYjk,XYik->XYij', eps, a)

def rotateLatticeVectorByRod(v, rod):
    theta = np.sqrt(np.einsum('XYi,XYi->XY', rod, rod))
    axis  = np.einsum('XYi,XY->XYi',rod, 1/theta)
    return np.einsum('XYij,XY->XYij', v,np.cos(theta)) + \
           np.einsum('jkl,XYk,XYil,XY->XYij', levi, axis, v, np.sin(theta)) + \
           np.einsum('XYj,XYk,XYik,XY->XYij', axis, axis, v, (1-np.cos(theta)))

def latticeToReciprocal(a):
    acubed = 1/np.einsum('XYi,XYi->XY', a[...,0,:], np.einsum('ijk,XYj,XYk->XYi', levi, a[...,1,:], a[...,2,:]))
    return 2*np.pi * np.einsum('ilk,jmn,XYlm,XYkn,XY->XYij', levi*(1+levi)/2, levi, a, a, acubed)

def projectLatticeToScreen(rod, eps, a0=3.82, lattice="fcc", extent=2):
    print "precalculating"
    a = generateLatticeVectors(lattice, a0, rod.shape[:-1])
    ap = strainLattice(a, eps)
    ap = rotateLatticeVectorByRod(ap, rod)
    a = ap

    freq = np.fft.fftfreq(256)+0.001
    t = np.random.random((3,)) - 0.5
    t /= np.sqrt(sum(t*t))
    rod[...,:] += t

    print "projecting"
    screen = np.zeros((256,256)).astype("complex")
    x = []
    y = []
    z = []
    for i in xrange(0,12):
        print i
        for j in xrange(0,12):
            tscreen = np.zeros((256,256))
            for hkl in generateHKLs(extent):
                g = np.einsum('i,ij->j', hkl, a[i,j])
                px = 32*g[0] / 16 + 128
                py = 32*g[1] / 16 + 128
                pz = 32*g[2] / 16 + 128
                x.append(px); y.append(py); z.append(pz);
                if px < 256 and py < 256 and px > 0 and py >0:
                    tscreen[px,py] += 1
            screen += tscreen #np.fft.fftn(100*tscreen)  # * np.exp(-8*np.pi*(i**2+j**2)/(128.**2))
    return x,y,z,np.fft.fftn(screen)



