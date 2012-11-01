import numpy
import scipy.misc as misc

import sys
sys.path += ["build/lib.linux-x86_64-2.7", "build/src.linux-x86_64-2.7"]
import boundarypruning as bp

def GetReconstructionWithBoundary(N,orientationmap,filename,indexmap=None):
    if indexmap is None:
        indexmap = self.GetIndexmap()
    newindexmap = numpy.empty(numpy.array(indexmap.shape)*4)
    for i in range(4):
        for j in range(4):
            newindexmap[i::4,j::4] = indexmap
    neworientationmap = numpy.empty([4*N,4*N,3])
    for k in range(3):
        for i in range(4):
            for j in range(4):
                neworientationmap[i::4,j::4,k] = orientationmap[:,:,k]
    boundarymap =  1-(indexmap==numpy.roll(indexmap, 1,0))*(indexmap==numpy.roll(indexmap,-1,0))*\
                   (indexmap==numpy.roll(indexmap, 1,1))*(indexmap==numpy.roll(indexmap,-1,1))
    xs,ys = boundarymap.nonzero()
    for (i,j) in zip(xs,ys):
        temp = [indexmap[i,j]==indexmap[(i-1+N)%N,j],indexmap[i,j]==indexmap[(i+1)%N,j],\
                indexmap[i,j]==indexmap[i,(j-1+N)%N],indexmap[i,j]==indexmap[i,(j+1)%N]]
        pos = [[(4*i,4*j),(4*i+1,4*j+4)],[(4*i+3,4*j),(4*i+4,4*j+4)],\
               [(4*i,4*j),(4*i+4,4*j+1)],[(4*i,4*j+3),(4*i+4,4*j+4)]]
        for n in range(4):
            if not temp[n]:
                newindexmap[pos[n][0][0]:pos[n][1][0],pos[n][0][1]:pos[n][1][1]] = -1
    for k in range(3):
        neworientationmap[:,:,k] *= (newindexmap!=-1)
    """
    Use PIL to plot for larger sizes, since we want to be able to draw pixel by pixel.
    """
    # I don't have working PIL! skip for now
    #from PIL import Image
    import Image
    pilImage = Image.fromarray(neworientationmap.astype('uint8'), 'RGB')
    pilImage.save(filename+'.png')


output = "out"
lena = True

if lena == True:
    N = 128
    rod = misc.imread("lena.jpg")
else:
    N = 1024 
    rod = misc.imread("earth.jpg")

#rod = rod.transpose([1,0,2])
omega    = rod.flatten().astype('float64')
mis      = numpy.zeros(4*N**2, dtype='float64')
grain    = numpy.zeros(4*N**2, dtype='int32')
bdlength = numpy.zeros(4*N**2, dtype='int32')
#FIXME - need to be modified
indexmap = numpy.zeros(N**2, dtype='int32')


status, ngrain, nbd = bp.boundary_pruning(N, 2, omega, mis, grain, bdlength, indexmap, 180, 1.5)
print status, ngrain, nbd
GetReconstructionWithBoundary(N,rod,output,indexmap=indexmap.reshape(N,N))
