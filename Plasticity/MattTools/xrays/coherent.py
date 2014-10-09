from Plasticity import TarFile
import numpy as np
import scipy as sp
import pylab as pl
import scipy.ndimage as nd

t,s = TarFile.LoadTarState("/media/scratch/plasticity/lvp2d1024_s2_d0.tar",0) 
#t,s = TarFile.LoadTarState("/home/m-bierbaum/lvp2d1024_s2_d0.tar", 0)

u = s.CalculateDisplacementField()

pl.rcParams.update({'xtick.labelsize':24,
    'xtick.major.size':20,
    'xtick.minor.size':10,
    'ytick.labelsize':24,
    'ytick.major.size':20,
    'ytick.minor.size':10,
    'lines.markersize':5,
    'axes.labelsize':24,
    'figure.figsize':[10.,8.],
    'text.usetex':False,
    'legend.fontsize':24,
    'legend.columnspacing':1.5,
    })

# create some standard tensors
delta = np.eye(3, dtype='int')
levi  = np.array([[[-(i-j)*(j-k)*(k-i)/2 for i in range(3)] for j in range(3)] for k in range(3)])
def generateLatticeVectors(lattice='fcc', a0=3.62, shape=(1,)):
    if lattice=='fcc':
        a = a0/2 * np.array([[0,1,1],[1,0,1],[1,1,0]]).T 
    else:
        a = a0/2 * np.eye(3)
    return a

def latticeToReciprocal(a):
    acubed = 1/np.einsum('i,i->', a[0,:], np.einsum('ijk,j,k->i', levi, a[1,:], a[2,:]))
    return 2*np.pi * np.einsum('ilk,jmn,lm,kn->ij', levi*(1+levi)/2, levi, a, a)* acubed

def generateHKLs(largest=4):
    import itertools
    return ( np.array(x) for x in itertools.product(xrange(-largest,largest+1), repeat=3))

lattice = generateLatticeVectors()
recip = latticeToReciprocal(lattice)

f2 = np.zeros(s.gridShape)

h = 4
uu = h*recip[0][0]*u['x']+h*recip[0][1]*u['y']+h*recip[0][2]*u['z']
#eu = np.sqrt(uu**2) #np.exp(2*np.pi*1.j*uu)
eu = np.exp(2*np.pi*1.j*uu)
ft = np.fft.fftn(eu)
f2 += np.real_if_close(ft*ft.conj())
f2 = np.fft.fftshift(f2)

dd = np.mgrid[-0.5:0.5:s.gridShape[0]*1j, -0.5:0.5:s.gridShape[1]*1j]
rr = np.sqrt(dd[0]*dd[0] + dd[1]*dd[1])

factor = 4
nbins = s.gridShape[0]/4
#bins = np.digitize(rr.flatten(), np.linspace(0,1.0, nbins))
bins = np.digitize(rr.flatten(), np.logspace(-3,0,nbins/factor))
binc = np.bincount(bins, minlength=nbins/factor)
rad  = nd.sum(f2.flatten(), labels=bins, index=np.arange(0,len(binc)))

xs = np.logspace(-3,0,nbins/factor) 
ys = rad/binc

pl.loglog(xs, ys, 'o', label='I(r)')
pl.show()

#good = xs > 0.15 
good = ((xs > 0.01402) * (xs < 0.03827)) == 1
#good = ((xs > 0.002828) * (xs < 0.005623)) == 1
good = good * ~np.isnan(np.log(ys))

exp, intercept = np.polyfit(np.log(xs)[good], np.log(ys)[good], 1)
pl.loglog(xs, xs**exp * np.exp(intercept), '-', label="Fit "+r"$\beta = %0.2f$" % exp)
pl.legend()
pl.xlabel("q", fontsize=30)
pl.ylabel("Intensity", fontsize=30)
