from Plasticity.GridArray.CUDAGridArray import GridArray
from Plasticity.GridArray import FourierSpaceTools
from numpy import *

def identity(a):
    greater = (a>pi).astype(float)
    return a-2*pi*greater

class CUDAFourierSpaceTools:
    """
    This class holds several arrays that can be utilized in calculations in k-space.
    Just note to self:
       fftfreq(N, 1/N) = 
       fromfunction(lambda z: N*func(2*pi*z/float(N))/(2*pi), [N,])
    with the highest frequency's sign shifted
    """
    def __init__(self, gridShape, func=identity):
        self.gridShape = gridShape
        self.dimension = len(gridShape)
        if   self.dimension == 1:
            newgridShape = [gridShape[0],]
            self.gridShape = tuple(newgridShape)
            self.kz  = GridArray.GridArray(fromfunction(lambda z: gridShape[0]*func(2.*pi*(z)/float(gridShape[0])), newgridShape))
            self.kx  = GridArray.GridArray.zeros(tuple(newgridShape))
            self.ky  = GridArray.GridArray.zeros(tuple(newgridShape))
            self.kSq   = self.kz*self.kz
            self.kSqSq = self.kSq*self.kSq

            # FIXME - this is a temporary fix need to be rewritten with proper suppoer in GridArray
            self.kmask = GridArray.GridArray(fromfunction(lambda x: (1-(x==0)),newgridShape))

        elif self.dimension == 2:
            newgridShape = [gridShape[0], gridShape[1]]
            self.gridShape = tuple(newgridShape)
            self.kx  = GridArray.GridArray(fromfunction(lambda x,y: gridShape[1]*func(2.*pi*(x)/float(gridShape[0])), newgridShape))
            self.ky  = GridArray.GridArray(fromfunction(lambda x,y: gridShape[1]*func(2.*pi*(y)/float(gridShape[1])), newgridShape))
            self.kz  = GridArray.GridArray.zeros(tuple(newgridShape))

            self.kxkx, self.kxky, self.kyky  = self.kx*self.kx, self.kx*self.ky, self.ky*self.ky
            self.kSq   = self.kxkx + self.kyky
            self.kSqSq = self.kSq*self.kSq
            # FIXME - this is a temporary fix need to be rewritten with proper suppoer in GridArray
            self.kmask = GridArray.GridArray(fromfunction(lambda x,y: (1-(x==0)*(y==0)),newgridShape))

        elif self.dimension == 3:
            newgridShape = [gridShape[0], gridShape[1], gridShape[2]]
            self.gridShape = tuple(newgridShape)
            self.kx  = GridArray.GridArray(fromfunction(lambda x,y,z: gridShape[0]*func(2.*pi*(x)/float(gridShape[0])), newgridShape))
            self.ky  = GridArray.GridArray(fromfunction(lambda x,y,z: gridShape[1]*func(2.*pi*(y)/float(gridShape[1])), newgridShape))
            self.kz  = GridArray.GridArray(fromfunction(lambda x,y,z: gridShape[2]*func(2.*pi*(z)/float(gridShape[2])), newgridShape))

            self.kxkx, self.kyky, self.kzkz  = self.kx*self.kx, self.ky*self.ky, self.kz*self.kz
            self.kxky, self.kykz, self.kxkz  = self.kx*self.ky, self.ky*self.kz, self.kx*self.kz
            self.kSq   = self.kxkx + self.kyky + self.kzkz
            self.kSqSq = self.kSq*self.kSq

            # FIXME - this is a temporary fix need to be rewritten with proper suppoer in GridArray
            self.kmask = GridArray.GridArray(fromfunction(lambda x,y,z: (1-(x==0)*(y==0)*(z==0)),newgridShape))

        self.k = {}

        x='x'
        y='y'
        z='z'
        xyz = [x,y,z]
        
        self.k[x] = self.kx
        self.k[y] = self.ky
        self.k[z] = self.kz


FourierSpaceTools.FourierSpaceTools = CUDAFourierSpaceTools
