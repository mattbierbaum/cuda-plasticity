import sys
sys.path += [".."]
import PlasticitySystem
import FieldInitializer

from Constants import *

from numpy import fromfunction, zeros, array, sin, cos, pi
from scipy.cluster.vq import *

rodrigues_weight = 100.

def GrainMapFromRodrigues(rodrigues, n=50, scale=2.*pi, get_index_map=False, resmap=None, rhosum=None, rhoMax=None):
    """
    only works for 2-D for now

    This basically returns the map of grains, by 7-D clustering of spatial
    and rodrigues vector space, with more weight on rodrigues vectors.

    You may supply an initial seed(k centroids) in the resmap parameter.
    Setting get_index_map=True returns the index map and resmap for plotting
    the index map and using the centroids for other purposes.

    It is important to note that k-means algorithm does not find a global
    optimum and uses a random seed. Therefore, results may vary for each
    run even on the same rodrigues vector field.

    example usage:

    import GrainObserver
    grainsize, index_map, resmap = GrainObserver.GrainMapFromRodrigues(rodrigues, get_index_map=True)
    maxsize = grainsize.max()
    threshold = maxsize/5
    grainsize *= (grainsize>threshold)
    cells = len(grainsize[grainsize.nonzero()])
    resmap = resmap[grainsize.nonzero()]
    print "Approximately %d cells identified as significantly large(20%% of largest size)" % cells
    grainsize, index_map, resmap = GrainObserver.GrainMapFromRodrigues(rodrigues, n=cells, get_index_map=True, resmap=resmap)


    This example, runs the function with large(default) number k first, and cuts
    off the centroids with less than 20% of the largest cluster. Then it re runs
    to reassign leftover clusters to the larger cells.
    """
    cells = n
    gridShape = rodrigues[x].shape

    rodx = rodrigues[x].flatten()
    rody = rodrigues[y].flatten()
    rodz = rodrigues[z].flatten()

    nx, ny = gridShape
    """
    We embed the positions in 4-dimensional space to account for
    PBC which can not be applied in kmeans routine directly.
    This embedding still keeps the metric intact
    """
    optional = 0
    if rhosum is not None:
        optional = 1
    rx1 = fromfunction(lambda x,y: cos(x*2.*pi/nx), gridShape)
    rx2 = fromfunction(lambda x,y: sin(x*2.*pi/nx), gridShape)
    ry1 = fromfunction(lambda x,y: cos(y*2.*pi/ny), gridShape)
    ry2 = fromfunction(lambda x,y: sin(y*2.*pi/ny), gridShape)
    data = zeros((len(rodx), 7))
    data[:,0] = rx1.flatten() 
    data[:,1] = rx2.flatten() 
    data[:,2] = ry1.flatten() 
    data[:,3] = ry2.flatten() 

    data = whiten(data)
    """
    put rodrigues vectors in after whitening,
    since these must not be separately treated.
    and these also have danger of being variance free,
    giving NaN
    """
    data[:,4] = rodx
    data[:,5] = rody
    data[:,6] = rodz
    #totvar = rodx.var()+rody.var()+rodz.var()
    # angle is much more important then spatial distance?
    #data[:,4:7] *= rodrigues_weight/sqrt(totvar)
    data[:,4:7] *= rodrigues_weight/scale
    if optional:
        if rhoMax is None:
            rhocond = (rhosum<rhosum.mean()).flatten()
        else:
            rhocond = (rhosum<rhoMax).flatten()
        newdata = zeros((len(rhocond[rhocond.nonzero()]),7))
        newdata[:,0] = data[rhocond.nonzero(),0]
        newdata[:,1] = data[rhocond.nonzero(),1]
        newdata[:,2] = data[rhocond.nonzero(),2]
        newdata[:,3] = data[rhocond.nonzero(),3]
        newdata[:,4] = data[rhocond.nonzero(),4]
        newdata[:,5] = data[rhocond.nonzero(),5]
        newdata[:,6] = data[rhocond.nonzero(),6]
        data = newdata
    if resmap is None:
        res2, idx = kmeans2(data, cells, minit='random')
    else:
        res2, idx = kmeans2(data, resmap, minit='matrix')

    if optional:
        idx2 = zeros(gridShape, dtype=int32).flatten()-1
        idx2[rhocond.nonzero()] = idx
    else:
        idx2 = idx
    """
    Now that we have the clustering, we count and get sizes
    """
    #import pylab
    #grainsize, index, pl = pylab.hist(idx2, bins=cells)
    maxindex = idx2.max()
    grainsize = array([(idx2==index).sum() for index in range(maxindex+1)])
    #grainsize.sort()
    if get_index_map:
        return grainsize, idx2.reshape(gridShape), res2
    return grainsize


def GetEBSDColoredIndex(index_map, resmap, scale=2.*pi, rhosum=None, threshold=None):
    """
    Assign EBSD color indice to every centroid.
    """
    rodrigues = {}
    rodrigues[x] = resmap[:,4] / rodrigues_weight * scale 
    rodrigues[y] = resmap[:,5] / rodrigues_weight * scale 
    rodrigues[z] = resmap[:,6] / rodrigues_weight * scale 

    import OrientationField
    rgb = OrientationField.RodriguesTo100RGB(rodrigues)

    colored_map = array([(rgb[idx,:]) for idx in index_map])
    if rhosum is not None:
        """
        apply a cutoff on rhosum, to plot wall areas black
        """
        if threshold is not None:
            mean = threshold
        else:
            mean = rhosum.mean()
        grain_interior = (rhosum<mean)
        colored_map[:,:,0] *= grain_interior
        colored_map[:,:,1] *= grain_interior
        colored_map[:,:,2] *= grain_interior
    return colored_map
 

def GetUnambiguousColorIndex(index_map, resmap, maxRange, centerR, scale=2.*pi):
    """
    Assign unambiguous color indice to every centroid.
    """
    rodrigues = {}
    rodrigues[x] = resmap[:,4] / rodrigues_weight * scale 
    rodrigues[y] = resmap[:,5] / rodrigues_weight * scale 
    rodrigues[z] = resmap[:,6] / rodrigues_weight * scale 

    rodrigues[x] = 255.*(0.5+(rodrigues[x]-centerR[0])/maxRange)+1.
    rodrigues[y] = 255.*(0.5+(rodrigues[y]-centerR[1])/maxRange)+1.
    rodrigues[z] = 255.*(0.5+(rodrigues[z]-centerR[2])/maxRange)+1.

    colormap = zeros((index_map.shape[0],index_map.shape[1],3),float)
    for i in range(index_map.shape[0]):
        for j in range(index_map.shape[1]):
            colormap[i,j,0] = rodrigues[x][index_map[i,j]] 
            colormap[i,j,1] = rodrigues[y][index_map[i,j]] 
            colormap[i,j,2] = rodrigues[z][index_map[i,j]] 
    return colormap.astype(int)
    

def LoadingTimeSerialRodriguesFromNumpyFiles(gridShape,filename,Ti,Tf,dt):
    """
    Load a time-serial of numpy data files, and return rodrigues vectors,
    time list and RhoModulus list, which are necessary elements to build
    up (8D) time-dependent Kmeans clustering method. 
    """
    rodrigues = {} 
    rodxs = []
    rodys = []
    rodzs = []
    rhos  = []
    time  = []
    t = Ti
    while t<Tf:
        t += dt
        time.append(t)
        state = FieldInitializer.NumpyTensorInitializer(gridShape,filename+str(t).replace('.','_')+'.dat',amplitude=1.)
        rod = state.CalculateRotationRodrigues()
        rho = state.CalculateRhoFourier()
        rhos.append(rho.modulus())
        rodxs.append(rod[x])
        rodys.append(rod[y])
        rodzs.append(rod[z])
    rodrigues[x] = array(rodxs)
    rodrigues[y] = array(rodys)
    rodrigues[z] = array(rodzs)
    rhos = array(rhos)
    return rodrigues,time,rhos


def GrainMapFromRodriguesTimeDependent(N, rodrigues, times, n=5000, scale=2.*pi, get_index_map=False, resmap=None, rhos=None, rhoMax=None):
    """
    This basically returns the map of grains, by 7-D clustering of spatial
    and rodrigues vector space + 1-D time space.
    """
    cells = n
    Nt = len(times)

    result = zeros((Nt,N,N),float)
    rodx = rodrigues[x].flatten()
    rody = rodrigues[y].flatten()
    rodz = rodrigues[z].flatten()

    gridShape = (N,N)
    nx, ny = gridShape
    """
    We embed the positions in 4-dimensional space to account for
    PBC which can not be applied in kmeans routine directly.
    This embedding still keeps the metric intact
    """
    optional = 0
    if rhos is not None:
        optional = 1
    rx1 = fromfunction(lambda x,y: cos(x*2.*pi/nx), gridShape)
    rx2 = fromfunction(lambda x,y: sin(x*2.*pi/nx), gridShape)
    ry1 = fromfunction(lambda x,y: cos(y*2.*pi/ny), gridShape)
    ry2 = fromfunction(lambda x,y: sin(y*2.*pi/ny), gridShape)
    rx1s= array([rx1]*Nt) 
    rx2s= array([rx2]*Nt) 
    ry1s= array([ry1]*Nt) 
    ry2s= array([ry2]*Nt) 
    data = zeros((len(rodx), 8))
    data[:,0] = rx1s.flatten() 
    data[:,1] = rx2s.flatten() 
    data[:,2] = ry1s.flatten() 
    data[:,3] = ry2s.flatten() 
    data[:,4] = array([ones(N*N)*i for i in times]).flatten()
    data = whiten(data)
    """
    put rodrigues vectors in after whitening,
    since these must not be separately treated.
    and these also have danger of being variance free,
    giving NaN
    """
    data[:,5] = rodx
    data[:,6] = rody
    data[:,7] = rodz
    data[:,5:8] *= rodrigues_weight/scale
    if optional:
        if rhoMax is None:
            rhocond = (rhos<rhos.mean()).flatten()
        else:
            rhocond = (rhos<rhoMax).flatten()
        newdata = zeros((len(rhocond[rhocond.nonzero()]),8))
        newdata[:,0] = data[rhocond.nonzero(),0]
        newdata[:,1] = data[rhocond.nonzero(),1]
        newdata[:,2] = data[rhocond.nonzero(),2]
        newdata[:,3] = data[rhocond.nonzero(),3]
        newdata[:,4] = data[rhocond.nonzero(),4]
        newdata[:,5] = data[rhocond.nonzero(),5]
        newdata[:,6] = data[rhocond.nonzero(),6]
        newdata[:,7] = data[rhocond.nonzero(),7]
        data = newdata
    if resmap is None:
        res2, idx = kmeans2(data, cells, minit='random')
    else:
        res2, idx = kmeans2(data, resmap, minit='matrix')

    if optional:
        idx2 = zeros((Nt,N,N), dtype=int32).flatten()-1
        idx2[rhocond.nonzero()] = idx
    else:
        idx2 = idx
    """
    Now that we have the clustering, we count and get sizes
    """
    #import pylab
    #grainsize, index, pl = pylab.hist(idx2, bins=cells)
    maxindex = idx2.max()
    idx2 = idx2.reshape(Nt,N,N)
    grainsize = []
    for i in range(Nt):
        grainsize.append(array([(idx2[i]==index).sum() for index in range(maxindex+1)]))
    if get_index_map:
        return array(grainsize), idx2, res2
    return grainsize


def GetEBSDColoredIndexTimeDependent(index_map, resmap, times, scale=2.*pi, rhosum=None, threshold=None):
    """
    Assign EBSD color indice to every centroid at each time snapshot.
    """
    Nt = len(times)
    rodrigues = {}
    rodrigues[x] = resmap[:,5] / rodrigues_weight * scale 
    rodrigues[y] = resmap[:,6] / rodrigues_weight * scale 
    rodrigues[z] = resmap[:,7] / rodrigues_weight * scale 

    import OrientationField
    rgb = OrientationField.RodriguesTo100RGB(rodrigues)
    
    colored_maps = []
    for i in range(Nt):
        colored_map = array([(rgb[idx,:]) for idx in index_map[i]])
        if rhosum is not None:
            """
            apply a cutoff on rhosum, to plot wall areas black
            """
            if threshold is not None:
                mean = threshold
            else:
                mean = rhosum.mean()
            grain_interior = (rhosum[i]<mean)
            colored_map[:,:,0] *= grain_interior
            colored_map[:,:,1] *= grain_interior
            colored_map[:,:,2] *= grain_interior
        colored_maps.append(colored_map)
    return array(colored_maps)
 

def GetUnambiguousColorIndexTimeDependent(index_map, resmap, times, maxRange, centerR, scale=2.*pi):
    """
    Assign unambiguous color indice to every centroid at each time snapshot.
    """
    Nt = len(times)
    rodrigues = {}
    rodrigues[x] = resmap[:,5] / rodrigues_weight * scale 
    rodrigues[y] = resmap[:,6] / rodrigues_weight * scale 
    rodrigues[z] = resmap[:,7] / rodrigues_weight * scale 

    rodrigues[x] = 255.*(0.5+(rodrigues[x]-centerR[0])/maxRange)+1.
    rodrigues[y] = 255.*(0.5+(rodrigues[y]-centerR[1])/maxRange)+1.
    rodrigues[z] = 255.*(0.5+(rodrigues[z]-centerR[2])/maxRange)+1.

    colormaps = zeros((Nt,index_map.shape[1],index_map.shape[2],3),float)
    for t in range(Nt):
        for i in range(index_map[t].shape[0]):
            for j in range(index_map[t].shape[1]):
                colormaps[t,i,j,0] = rodrigues[x][index_map[t,i,j]] 
                colormaps[t,i,j,1] = rodrigues[y][index_map[t,i,j]] 
                colormaps[t,i,j,2] = rodrigues[z][index_map[t,i,j]] 
    return colormaps.astype(int)
    

if __name__=="__main__":
    """
    N = 256
    filename = '2Dsimulation/Project/External/'+str(N)+'/uniaxial/VG2DNewGaussianRandom'+str(N)+'uniaxialzz_D_50_0linear1_0_betaP'
    dt = 0.08
    T  = 1.6 
    rhomax = 6.
    rodrigues, times, rhos = LoadingTimeSerialRodrigues((N,N),filename,0.,T,dt)
    grains,index_map,resmap = GrainMapFromRodriguesTimeDependent(N, rodrigues, times, 5000, pi/4., True, resmap=None, rhos=rhos, rhoMax=rhomax)
    colored_maps = GetEBSDColoredIndexTimeDependent(index_map, resmap, times, scale=2.*pi, rhosum=rhos, threshold=rhomax)
    colored_maps.tofile("test.dat")
    """
    N = 10
    gridShape = tuple([N,N])
    rodrigues = {}
    rodrigues[x] = zeros(gridShape)
    rodrigues[y] = zeros(gridShape)
    rodrigues[z] = zeros(gridShape)

    rodrigues[z] += 1.0 #+ random.random(gridShape)*0.0
    rodrigues[z][:N/2,:N/2] -= 1.0
    rodrigues[x][:,:N/2] += -1.0 #+ random.random([N,N/2])*0.0
    rodrigues[y][N/2:,:] += 0.5 #+ random.random([N/2,N])*0.0

    sizes = GrainMapFromRodrigues(rodrigues)
    print sizes
