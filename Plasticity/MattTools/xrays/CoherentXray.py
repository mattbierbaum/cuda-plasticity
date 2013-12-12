import sys
import scipy.weave as W

from numpy import fromfunction, arctan, sin, pi, sqrt ,cos, fabs, random, arcsin, array, average, zeros, fft, sort, exp, roll, ones, finfo
import numpy

ME = finfo(float).eps

x='x'
y='y'
z='z'

class CoherentXrayDiffraction:
    def __init__(self,g=None):
        self.g = g 

    def SecondOrderCorrFuncfCoherentXrayScattering(self,state,gridShape):
        """
        Returns the intensity of scattered light as a function
		of the reciprocal lattice vector q

		.. math::
			I(q) = \\frac{1}{V} \\tilde{f}(q) \\tilde{f}(-q)  
        
		where 
		
		.. math::
			f(x) = e^{2 \pi i g*u(x)}

        Assume that there is a cubic symmetry, so that g = (h/a, k/a, l/a)  (set a=1)
        """
        V = float(array(gridShape).prod())
        u = state#.CalculateDisplacementField()
        ug = self.g[0]*u[x]+self.g[1]*u[y]+self.g[2]*u[z] 
        Kug = fft.fftn(ug) 
        u2g = ug*ug
        Ku2g = fft.fftn(u2g)
        I_0 = 1.
        I_1 = (1.j)*2.*pi*(Kug-Kug.conj())
        I_2 = 2.*pi**2*(2*Kug*Kug.conj()-Ku2g-Ku2g.conj())
        I_S = (I_0+I_2)/V
        I_A = I_1/V
        return I_S.real, I_A.real

    def FullCorrFuncfCoherentXrayScattering(self,state,gridShape):
        """
        Returns the intensity of scattered light as a function of 
		reciprocal lattice vector without making symmetry assumptions

		.. math::
			I(q) = \\frac{1}{V} \widetilde{e^{2 \pi i g*u(x)}} \widetilde{e^{2 \pi i g*u(x)}}^{*} 

        """
        V = float(array(gridShape).prod())
        u = state#.CalculateDisplacementField()
        u_dot_g = self.g[0]*u[x]+self.g[1]*u[y]+self.g[2]*u[z]
        f = exp((1.j)*2.*pi*u_dot_g)
        kf = numpy.fft.fftn(f)
        I = kf*(kf.conj())
        return I.real/V

def debug():
    pass


if __name__=="__main__":
    import numpy
    rod1 = {}
    N = 128 
    gridShape = tuple([N,N])
    rod1[x] = (random.random(gridShape)-0.5)*1.
    rod1[y] = (random.random(gridShape)-0.5)*2.
    rod1[z] = (random.random(gridShape)-0.5)*3.
    icx = IncoherentXrayScattering(axis='x',symmetry='fcc')
    data = icx.SimulateXrayDiffractionImage(rod1,k0=18,beamwidth=0.01)
    import pylab
    """
    pylab.figure(0)
    pylab.imshow(data,interpolation='nearest',cmap=pylab.cm.gist_gray_r)
    pylab.title("Random orientation")
    """
    def getrandomstrain(gridShape, maxstrain=.1):
        strain = {}
        strain[x,x] = (random.random(gridShape)-0.5)*maxstrain
        strain[y,y] = (random.random(gridShape)-0.5)*maxstrain
        strain[z,z] = (random.random(gridShape)-0.5)*maxstrain
        strain[x,y] = strain[y,x] = (random.random(gridShape)-0.5)*maxstrain
        strain[y,z] = strain[z,y] = (random.random(gridShape)-0.5)*maxstrain
        strain[x,z] = strain[z,x] = (random.random(gridShape)-0.5)*maxstrain
        return strain
    strain = getrandomstrain(gridShape)
    hkl = icx.ListPlanes(1,24)
    i = 1
    for item in hkl:
        data1 = icx.SimulateXrayDiffractionImage(rod1, strain=strain,  k0=24, beamwidth=0.01, hkl=item)
        if data1.max()<1.e-10:
            print str(i),item
            i += 1
    print len(hkl),hkl
    data1 = icx.SimulateXrayDiffractionImage(rod1, strain=strain,  k0=24, beamwidth=0.01)
    vMax = numpy.max(data1)
    """
    region = (data1>0.).astype(int).nonzero()
    pos = [(region[0][i],region[1][i]) for i in range(len(region[0])) if region[1][i]<128]
    mask = icx.RetrieveSpatialInformation((0,-1,0), pos, rod1, strain=strain, k0=18, beamwidth=0.01, systemshape=gridShape)
    rod2 = rod1.copy()
    rod2[x] *= mask
    rod2[y] *= mask
    rod2[z] *= mask
    strain2 = strain.copy() 
    for i in [x,y,z]:
        for j in [x,y,z]:
            strain2[i,j] *= mask
    data2 = icx.SimulateXrayDiffractionImage(rod2, strain=strain2, k0=24, beamwidth=0.01)
    """
    pylab.figure(0)
    #pylab.subplot(121)
    pylab.imshow(data1,interpolation='nearest',cmap=pylab.cm.gist_gray_r)
    pylab.colorbar()
    pylab.title("Random with small strain with window")
    #pylab.subplot(122)
    #pylab.imshow(data2,interpolation='nearest',cmap=pylab.cm.gist_gray_r,vmax=vMax)
    #pylab.title("Debug")
    #pylab.figure(1)
    #pylab.imshow(mask,interpolation='nearest')
    #pylab.title("Mask")
    pylab.show()
