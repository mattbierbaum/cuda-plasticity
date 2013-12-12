import sys
import scipy.weave as W

from numpy import fromfunction, arctan, sin, pi, sqrt ,cos, fabs 
from numpy import random, arcsin, array, average, zeros, fft
from numpy import sort, exp, roll, ones, finfo
import numpy

# machine epsilon
ME = finfo(float).eps

# Definitions for the dictionary of components
x='x'
y='y'
z='z'

class IncoherentXrayScattering:
    def __init__(self,D=1,L=1,axis='z',symmetry='scc'):
        """
        Create the experimental setup under which these x-rays will be measured.

        Apparatus geometry:
        D: distance from sample to detector
        L: side length of square screen of dector 
        axis: direction of the incident X-ray beam 
              one of {'scc', 'fcc', 'bcc'}
        """
        self.D = D
        self.L = L
        self.axis = axis
        self.symmetry = symmetry

    def RodriguesToSpots(self, rodrigues, strain=None, scale=None, k0=7, beamwidth=0.1):
        """
        This function turns rodrigues vector field into X-ray diffraction spots.
        Notice that
        1) Strain scales the same as the rodrigues vector.
        2) Everything is based on small deformation approximation.
        
        We assume the simple cubic lattice here.
       
        Returns tuple of 3 arrays containing as indices
            * 0:  x screen position of a spot
            * 1:  y screen position of a spot
            * 2:  intensity of the spot at the given x,y position
        """
        
        #lattice spacing.
        a = 1     
        hkls = self.ListPlanes(a,k0)
        if strain is not None:
            if scale is not None:
                strain *= scale
        xs = {} 
        ys = {} 
        intensities = {} 
        for h,k,l in hkls:
            # reciprocal vector
            b1,b2,b3 = self.ReciprocalPrimitiveVectors_CubicLattice(strain=strain)
            
            #calculate G
            if strain is not None: 
                G = b1*h + b2*k + b3*l
                Gx = G[0]
                Gy = G[1]
                Gz = G[2]
            else:
                Gx = h*b1
                Gy = k*b2
                Gz = l*b3
            
            #rotate G to lab direction
            Gx_lab,Gy_lab,Gz_lab = self.VectorRotatingUnderRodriguesVector((Gx,Gy,Gz),rodrigues,scale)
            
            #calculate k' and check condition
            #k' = k0 + G_lab
            kx_p = Gx_lab 
            ky_p = Gy_lab
            kz_p = Gz_lab 
            if self.axis == 'x':
                kx_p += k0
                k1 = ky_p
                k2 = kz_p
                k_p = sqrt(kx_p*kx_p+k1*k1+k2*k2)
            elif self.axis == 'y':
                ky_p += k0
                k1 = kz_p
                k2 = kx_p
                k_p = sqrt(ky_p*ky_p+k1*k1+k2*k2)
            elif self.axis == 'z':
                kz_p += k0
                k1 = kx_p
                k2 = ky_p
                k_p = sqrt(kz_p*kz_p+k1*k1+k2*k2)
            else:
                pass 
            intensity = exp(-(k_p-k0)*(k_p-k0)/2./(k0*beamwidth)**2)
            
            #plot a point
            xpos = k1/k_p
            ypos = k2/k_p
            xs[(h,k,l)] = xpos
            ys[(h,k,l)] = ypos
            intensities[(h,k,l)] = intensity
        return xs, ys, intensities

    def ListPlanes(self,latticeConstant,k):
        hkls = self.ListPlanes_Cubic(latticeConstant,k)
        if self.symmetry == 'scc':
            return hkls
        elif self.symmetry == 'fcc':
            result0 = [(t[0],t[1],t[2]) for t in hkls if ((t[0]%2==0)and(t[1]%2==0)and(t[2]%2==0))]
            result1 = [(t[0],t[1],t[2]) for t in hkls if ((t[0]%2!=0)and(t[1]%2!=0)and(t[2]%2!=0))]
            return result0+result1 
        elif self.symmetry == 'bcc':
            result = [(t[0],t[1],t[2]) for t in hkls if ((t[0]+t[1]+t[2])%2==0)]
            return result 
        else:
            pass

    def ListPlanes_Cubic(self,latticeConstant,k):
        """
        According to the apparatus geometry, make a list of avaible
        reflection planes.
        """
        
        #wave length
        lamb = 2*pi/k
        maxtheta = arctan(self.L/sqrt(2)/self.D)/2
        
        #Bragg's law
        d = latticeConstant
        nmax = int(2*d*sin(maxtheta)/lamb)
        assert nmax>0
        N = nmax*2+1
        nsize = [N, N, N]
        h = fromfunction(lambda a,b,c: (a-nmax), nsize).astype(int)
        k = fromfunction(lambda a,b,c: (b-nmax), nsize).astype(int)
        l = fromfunction(lambda a,b,c: (c-nmax), nsize).astype(int)
        
        #Interplanar spacing calculated
        ds = d/sqrt(h**2+k**2+l**2)
        hkl = {} 
        thetas = arcsin(lamb/2/ds)
        
        #Check for allowed hkls and put in a list
        for i in range(N):
            for j in range(N):
                for m in range(N):
                    if thetas[i,j,m] < maxtheta and (i!=0 or j!=0 or m!=0):
                        ijms = [h[i,j,m],k[i,j,m],l[i,j,m]]
                        hkl[tuple(ijms)] = 1.
        hkls = hkl.keys()
        hkls.remove((0,0,0))
        """
        if self.axis == 'x':
            hkls.remove((1,0,0))
            hkls.remove((-1,0,0))
        elif self.axis == 'y':
            hkls.remove((0,1,0))
            hkls.remove((0,-1,0))
        elif self.axis == 'z':
            hkls.remove((0,0,1))
            hkls.remove((0,0,-1))
        else:
            pass
        """
        return hkls

    def ReciprocalPrimitiveVectors_CubicLattice(self,latticeConstant=1,strain=None):
        """
        Calculate the reciprocal primitive vectors for simple cubic lattice in
        the presence of elastic strain. 
        """
        a = latticeConstant
        if strain is not None:
            b1 = 2*pi/a * array([1+strain[y,y]+strain[z,z], -strain[y,x], -strain[z,x]])
            b2 = 2*pi/a * array([-strain[x,y], 1+strain[x,x]+strain[z,z], -strain[z,y]])
            b3 = 2*pi/a * array([-strain[x,z], -strain[y,z], 1+strain[x,x]+strain[y,y]])
            volchange = 1.+strain[x,x]+strain[y,y]+strain[z,z]
            b1 /= volchange
            b2 /= volchange
            b3 /= volchange
        else:
            b1 = 2*pi/a
            b2 = 2*pi/a
            b3 = 2*pi/a
        return b1,b2,b3

    def VectorRotatingUnderRodriguesVector(self,vector,rodrigues,scale=None):
        """
        The direction of Rodrigues vector defines the rotation axis;
        the magnitude of Rodrigues vector defines the rotation angle. 
        """ 
        rodx = rodrigues[x]
        rody = rodrigues[y]
        rodz = rodrigues[z]
        
        #Rotation angle 
        theta = sqrt(rodx**2+rody**2+rodz**2)+2.2e-16
        
        #Rotation axis
        ux = rodx/theta
        uy = rody/theta
        uz = rodz/theta
        if scale is not None:
            theta *= scale
        Gx,Gy,Gz = vector
        
        #Calculate u*G
        u_dot_G = (ux*Gx+uy*Gy+uz*Gz)
        Gx_rotated = Gx*cos(theta)+(uy*Gz-uz*Gy)*sin(theta)+u_dot_G*ux*(1-cos(theta))
        Gy_rotated = Gy*cos(theta)+(uz*Gx-ux*Gz)*sin(theta)+u_dot_G*uy*(1-cos(theta))
        Gz_rotated = Gz*cos(theta)+(ux*Gy-uy*Gx)*sin(theta)+u_dot_G*uz*(1-cos(theta))
        return Gx_rotated,Gy_rotated,Gz_rotated

    def XYPlotToImage(self,xs,ys,intensities,screenlatticesize):
        N = screenlatticesize
        xs = array(xs).flatten()
        ys = array(ys).flatten()
        intensities = array(intensities).flatten()+ME
        data = zeros((2*N,2*N),float)
        xlist = (N+xs*N).astype(int)
        ylist = (N+ys*N).astype(int)
        
        #slow part
        """
        newdata = zeros((2*N,2*N),float)
        c = 0
        for i,j in zip (xlist,ylist):
            newdata[i,j] += intensities[c]
            c += 1
        """
        np = len(xlist)
        s = 2*N
        code = """
        for (int i=0; i<np; i++) 
            *(data+(*(xlist+i))*s+*(ylist+i)) += *(intensities+i);
        """
        variables = ['np','s','xlist', 'ylist','intensities','data']
        W.inline(code, variables, extra_compile_args=["-w"])
        return data 

    def SimulateXrayDiffractionImage(self, rodrigues, strain=None, scale=None, k0=12, beamwidth=0.01, hkl=None, screenlatticesize=128):
        """
        Perform the entire Incoherent X-ray simulation given a set of rodrigues vectors
        and produce the real-space image of the diffraction on the experimental screen.

        Required parameters:
            * rodrigues: a 3 vector of rotation values

        Options available:
            * strain: strain values for the same rodriques rotation values
            * k0: center of the beam's x-ray wavevector
            * beamwidth: spread of the beam as :math:`\\frac{\Delta k}{k}`
            * hkl: a set of lattice vectors to use.  If Null then use all.

        """
        
        xs,ys,ints = self.RodriguesToSpots(rodrigues,strain,scale,k0,beamwidth)
        if hkl is None:
            imagedata = self.XYPlotToImage(xs.values(),ys.values(),ints.values(),screenlatticesize)
        else:
            imagedata = self.XYPlotToImage(xs[hkl],ys[hkl],ints[hkl],screenlatticesize)
        return imagedata 

    def RetrieveSpatialInformation(self, hkl, region, rodrigues, strain=None, scale=None, k0=12, beamwidth=0.01, systemshape=None, screenlatticesize=128):
        """
        Return a boolean array of the size hkl.shape which corresponds to the real space
        information that in turn corresponds to a certain region of the diffraction pattern.

        Region in terms of screen coordinates in the form of a boolean array of size (screenlatticesize, screenlatticesize)

        Rodriques is the typical rodrigues vector provided to the other x-ray functions.
        """

        if systemshape is None: 
            systemshape = rodrigues.gridShape
        xs,ys,ints = self.RodriguesToSpots(rodrigues,strain,scale,k0,beamwidth)
        cutoff = exp(-0.5*4**2)
        print xs.keys()
        print ints.keys()
        N = screenlatticesize
        xlist = (N+xs[hkl]*N).astype(int)
        ylist = (N+ys[hkl]*N).astype(int)
        print ints>cutoff
        mask = 0
        for point in region:
            x,y = point
            base = ones(systemshape,float)
            mask += base*(xlist==x)*(ylist==y)*(ints>cutoff)
        mask = (mask>0).astype(float)
        return mask 


def debug():
    pass

if __name__=="__main__":
    import numpy
    rod1 = {}
    N = 128 
    import sys
    import FieldInitializer
    t,s = FieldInitializer.LoadState(
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
    region = (data1>0.).astype(int).nonzero()
    pos = [(region[0][i],region[1][i]) for i in range(len(region[0])) if region[1][i]<128]
    mask = icx.RetrieveSpatialInformation((1,1,1), pos, rod1, strain=strain, k0=18, beamwidth=0.01, systemshape=gridShape)
    rod2 = rod1.copy()
    rod2[x] *= mask
    rod2[y] *= mask
    rod2[z] *= mask
    strain2 = strain.copy() 
    for i in [x,y,z]:
        for j in [x,y,z]:
            strain2[i,j] *= mask
    data2 = icx.SimulateXrayDiffractionImage(rod2, strain=strain2, k0=24, beamwidth=0.01)
    pylab.figure(0)
    pylab.subplot(121)
    pylab.imshow(data1,interpolation='nearest',cmap=pylab.cm.gist_gray_r)
    pylab.colorbar()
    pylab.title("Random with small strain with window")
    pylab.subplot(122)
    pylab.imshow(data2,interpolation='nearest',cmap=pylab.cm.gist_gray_r,vmax=vMax)
    pylab.title("Debug")
    pylab.figure(1)
    pylab.imshow(mask,interpolation='nearest')
    pylab.title("Mask")
    pylab.show()
