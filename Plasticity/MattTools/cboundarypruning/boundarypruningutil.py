import numpy
import scipy.misc as misc

import sys
from Plasticity import PlasticitySystem
from Plasticity import TarFile
from Plasticity.Observers import OrientationField
#sys.path += ["bp/build/lib.linux-x86_64-2.6", "bp/build/src.linux-x86_64-2.6"]
import boundarypruning as bp

def GetReconstructionWithBoundary2D(N,orientationmap,filename,indexmap):
    newindexmap = numpy.empty(numpy.array(indexmap.shape[:2])*4)
    for i in range(4):
        for j in range(4):
            newindexmap[i::4,j::4] = indexmap[:,:]
   
    neworientationmap = numpy.empty([4*N,4*N,3])
    for k in range(3):
        for i in range(4):
            for j in range(4):
                neworientationmap[i::4,j::4,k] = orientationmap[:,:,k]
     
    boundarymap =  1-(indexmap==numpy.roll(indexmap, 1,0))*(indexmap==numpy.roll(indexmap,-1,0))*\
                   (indexmap==numpy.roll(indexmap, 1,1))*(indexmap==numpy.roll(indexmap,-1,1))
    
    ## can we get away with this without all the zips, etc.
    """
    boundarymap =  1-(newindexmap==numpy.roll(newindexmap, 1,0))*(newindexmap==numpy.roll(newindexmap,-1,0))*\
                   (newindexmap==numpy.roll(newindexmap, 1,1))*(newindexmap==numpy.roll(newindexmap,-1,1))
    for k in range(3):
        neworientationmap[:,:,k] *= (newindexmap!=-1)

    """

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


def GetReconstructionWithBoundary3D(N,orientationmap,filename,indexmap,dovtk=False):
    newindexmap = numpy.empty(numpy.array(indexmap.shape)*4)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                newindexmap[i::4,j::4,k::4] = indexmap
    
    neworientationmap = numpy.empty([4*N,4*N,4*N,3])
    for l in range(3):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    neworientationmap[i::4,j::4,k::4,l] = orientationmap[:,:,:,l]
    
    boundarymap =  1-(indexmap==numpy.roll(indexmap, 1,0))*(indexmap==numpy.roll(indexmap,-1,0))*\
                   (indexmap==numpy.roll(indexmap, 1,1))*(indexmap==numpy.roll(indexmap,-1,1))*\
                   (indexmap==numpy.roll(indexmap, 1,2))*(indexmap==numpy.roll(indexmap,-1,2))
    
    xs,ys,zs = boundarymap.nonzero()
    for (i,j,k) in zip(xs,ys,zs):
        temp = [indexmap[i,j,k]==indexmap[(i-1+N)%N,j,k],indexmap[i,j,k]==indexmap[(i+1)%N,j,k],\
                indexmap[i,j,k]==indexmap[i,(j-1+N)%N,k],indexmap[i,j,k]==indexmap[i,(j+1)%N,k],\
                indexmap[i,j,k]==indexmap[i,j,(k-1+N)%N],indexmap[i,j,k]==indexmap[i,j,(k+1)%N]]
        pos = [[(4*i,4*j,4*k),(4*i+1,4*j+4,4*k+4)],[(4*i+3,4*j,4*k),(4*i+4,4*j+4,4*k+4)],\
               [(4*i,4*j,4*k),(4*i+4,4*j+1,4*k+4)],[(4*i,4*j+3,4*k),(4*i+4,4*j+4,4*k+4)],\
               [(4*i,4*j,4*k),(4*i+4,4*j+4,4*k+1)],[(4*i,4*j,4*k+3),(4*i+4,4*j+4,4*k+4)]]
        for n in range(6):
            if not temp[n]:
                newindexmap[pos[n][0][0]:pos[n][1][0],pos[n][0][1]:pos[n][1][1],pos[n][0][2]:pos[n][1][2]] = -1
    
    for k in range(3):
        neworientationmap[:,:,:,k] *= (newindexmap!=-1)
    """
    Use PIL to plot for larger sizes, since we want to be able to draw pixel by pixel.
    """
    if dovtk == True:
        import boundarypruningvtk as vtk
        vtk.plot(N, neworientationmap, prefix="pruningvtk", animate=True, write=True)
    else:
        import Image
        for i in range(4*N):
            pilImage = Image.fromarray(neworientationmap[:,:,i,:].astype('uint8'), 'RGB')
            pilImage.save(filename+'_%04d.png'%i)
    

def Run(N,dim,rodrigues,prefix,J=0.00008,PtoA=1.5,Image=False,Dump=False,dovtk=False,verbose=0):
    output = prefix+str(J)+"_"+str(PtoA)

    rod = numpy.zeros([3]+list(rodrigues['x'].numpy_array().shape), dtype='float64')
    rod[0] = rodrigues['x']
    rod[1] = rodrigues['y']
    rod[2] = rodrigues['z']
    orientationmap = OrientationField.RodriguesToUnambiguousColor(rod[0],rod[1],rod[2])
    rod = rod.transpose(range(1,len(rod.shape))+[0]).copy()
 
    omega    = rod.flatten().astype('float64')
    mis      = numpy.zeros((2*dim)*N**dim, dtype='float64')
    grain    = numpy.zeros((2*dim)*N**dim, dtype='int32')
    bdlength = numpy.zeros((2*dim)*N**dim, dtype='int32')
    indexmap = numpy.zeros(N**dim, dtype='int32')
    
    status, ngrain, nbd = bp.boundary_pruning(N, dim, omega, mis, grain, bdlength, indexmap, J, PtoA, verbose)

    if Image == True:
        if dim == 2 :
            GetReconstructionWithBoundary2D(N,orientationmap,output,indexmap=indexmap.reshape(N,N))
        if dim == 3:
            GetReconstructionWithBoundary3D(N,orientationmap,output,indexmap=indexmap.reshape(N,N,N),dovtk=dovtk)
                     
    
    if Dump == True:
        pickle.dump(mis[:nbd], open("%s.mis.pickle" % output, "w"))
        pickle.dump(grain[:ngrain], open("%s.grain.pickle" % output, "w"))
        pickle.dump(bdlength[:nbd], open("%s.bdlength.pickle" % output, "w"))

    return mis[:nbd], grain[:ngrain], bdlength[:nbd]

"""
import cPickle as pickle
rod = pickle.load(open("rodrigues3d.pickle","r"))
Run(128,3,rod, "testing", Image=True, J=8e-8, PtoA=16)
"""