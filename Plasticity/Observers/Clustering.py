import sys
sys.path += [".."]
import PlasticitySystem
import FieldInitializer
import scipy.weave as W

import numpy
from SortedList import *
####################################################################
### Kmeans clustering method from 3D vectors 
####################################################################
class KmeansClustering3D:
    def RunClustering(self,N,vector,K0):
        data = vector.reshape(N**2,3) 
        import scipy.cluster.vq as vq
        resmap,indexmap = vq.kmeans2(data,K0,iter=50,minit='random') 
        newresmap,indexmap = vq.kmeans2(data,resmap,iter=50,minit='matrix')
        self.indexmap = indexmap.reshape(N,N)
        self.CheckTopology(N)

    def CheckTopology(self,N):   
        oldTotalNumOfClusters = numpy.max(self.indexmap)+1
        i = 0
        while i in self.indexmap:
            xs,ys = (self.indexmap==i).nonzero()
            sites = [(xs[i],ys[i]) for i in range(len(xs))]
            newsites = [sites[0]]
            temp = [sites[0]]
            while temp!=[]:
                for (x,y) in temp:
                    neighbors = [((x+1)%N,y),((x-1+N)%N,y),(x,(y+1)%N),(x,(y-1+N)%N)]
                    temp = [n for n in neighbors if n in sites and n not in newsites]
                    newsites += temp
            if len(newsites) != len(sites):
                newcluster = [n for n in sites if n not in newsites]
                maxindex = int(numpy.max(self.indexmap))
                for n in newcluster:
                    self.indexmap[n] = maxindex+1
            i += 1
        newTotalNumOfClusters = numpy.max(self.indexmap)+1
        if oldTotalNumOfClusters == newTotalNumOfClusters:
            print "There is no topological error."
        else:
            print "We redefine "+str(int(newTotalNumOfClusters-oldTotalNumOfClusters))+" new clusters."
                
    def GetGrainsize(self):
        maxIndex = int(numpy.max(self.indexmap))
        return [numpy.sqrt((self.indexmap==i).sum()) for i in range(maxIndex+1)]

    def GetMisorientation(self):
        pass




####################################################################
### Steepest decent clustering method 
####################################################################
CORESIZE = 2
"""
Make a list of neighbor sites within a certain range.

core_neighbors contain only half of the sites to remove redundancy and
kernel function do take that into account.

core_neighbors_redundant retains the redundancy for other purposes.
"""
core_neighbors = [(i,j) for i in range(-CORESIZE-2,CORESIZE+3) \
                   for j in range(-CORESIZE-2,CORESIZE+3) \
                   if (i**2+j**2)<=(CORESIZE+2)**2 and (i != 0 or j != 0)
                       and ((i>0 and j>=0) or (i<=0 and j>0))]
core_neighbors_redundant = [(i,j) for i in range(-CORESIZE-2,CORESIZE+3) \
                   for j in range(-CORESIZE-2,CORESIZE+3) \
                   if (i**2+j**2)<=(CORESIZE+2)**2 and (i != 0 or j != 0)]
# removed redundancy in the last line condition
# 2 times for removing redundancy
kernel = lambda dx,dy: 2.*numpy.exp(-((dx)**2+(dy)**2)/2./CORESIZE**2)
kernel2 = lambda dxy2: 2.*numpy.exp(-(dxy2)/2./CORESIZE**2)*(dxy2<=(CORESIZE+2)**2)

class SiteList(numpy.ndarray):
    """
    Basically works as a list of sites, but allow for operating on them
    like a numpy array. Note that if site coordinates are not numbers
    this will likely not work.
    """
    def __new__(subtype, data, dtype=None, copy=False):
        subarr = numpy.array(data, dtype=dtype, copy=copy)
        subarr = subarr.view(subtype)
        return subarr

    """
    While it is a list, concatenate instead of adding as an array.
    """
    def __add__(self, other):
        return SiteList(numpy.concatenate((self, other)))
    def __radd__(self, other):
        return SiteList(numpy.concatenate((self, other)))
    def __iadd__(self, other):
        return SiteList(numpy.concatenate((self, other)))

    """
    However for operating on the site indices, make it into a numpy
    array
    """
    def get_indices(self):
        return numpy.array(self)

class Box:
    """
    Boundary box class for checking "collision" of two boxes
    """
    def __init__(self, arr, N):
        self.arr = numpy.array(arr)
        self.N = N

    def Inside(self, point):
        x,y = point
        xmM = self.arr[0]
        ymM = self.arr[1]
        return (x<=xmM[1]) and (x>=xmM[0]) and (y<=ymM[1]) and (y>=ymM[0])

    def Inside_x(self, x):
        xmM = self.arr[0]
        return (x<=xmM[1]) and (x>=xmM[0])

    def Inside_y(self, y):
        ymM = self.arr[0]
        return (y<=ymM[1]) and (y>=ymM[0])

    def GetSize(self):
        x_size = self.arr[0,1] - self.arr[0,0] + 1
        y_size = self.arr[1,1] - self.arr[1,0] + 1
        return x_size * y_size

    def MergeWith(self, other):
        xmM = [min(self.arr[0,0],other.arr[0,0]), max(self.arr[0,1],other.arr[0,1])]
        ymM = [min(self.arr[1,0],other.arr[1,0]), max(self.arr[1,1],other.arr[1,1])]
        return Box([xmM, ymM],self.N)

    def GetCenter(self):
        """
        Because collision checking is relatively messy, and also to keep
        the center/radius structure we use these two functions
        """
        return numpy.average(self.arr, axis=1)

    def GetRadius(self):
        return numpy.sqrt(((self.arr[:,1]-self.arr[:,0])**2).sum())*0.5

class BoxWithPBC(Box):
    """
    Boundary box class with periodic boundary conditions.
    """
    def Inside(self, point):
        x,y = point
        xmM = self.arr[0]
        ymM = self.arr[1]
        if x < xmM[0]:
            # see wrapped
            x += self.N
        if y < ymM[0]:
            # see wrapped
            y += self.N
        return (x<=xmM[1]) and (x>=xmM[0]) and (y<=ymM[1]) and (y>=ymM[0])

    def MergeWith(self, other):
        self_over_x = self.arr[0,1] >= self.N
        other_over_x = other.arr[0,1] >= self.N
        if self_over_x and other_over_x:
            # Both wrapping - obvious
            xmM = [min(self.arr[0,0],other.arr[0,0]), max(self.arr[0,1],other.arr[0,1])]
        elif self_over_x and not other_over_x: 
            # See if it's better to wrap other over
            # i.e., check if left of other is in self
            check = self.Inside_x(other.arr[0,0]+self.N-1)
            if check:
                # left of other inside self, wrap over
                xmM = [min(self.arr[0,0],other.arr[0,0]+self.N), max(self.arr[0,1],other.arr[0,1]+self.N)]
            else:
                # not the case 
                xmM = [min(self.arr[0,0],other.arr[0,0]), max(self.arr[0,1],other.arr[0,1])]
        elif not self_over_x and other_over_x: 
            # See if it's better to wrap self over
            # i.e., check if left of other is in other
            check = other.Inside_x(self.arr[0,0]+self.N-1)
            if check:
                # left of self inside other, wrap over
                xmM = [min(other.arr[0,0],self.arr[0,0]+self.N), max(other.arr[0,1],self.arr[0,1]+self.N)]
            else:
                # not the case 
                xmM = [min(other.arr[0,0],self.arr[0,0]), max(other.arr[0,1],self.arr[0,1])]
        elif not self_over_x and not other_over_x: 
            # See if they connect at the boundary
            check1 = (other.Inside_x(0) and self.Inside_x(self.N-1)) 
            check2 = (self.Inside_x(0) and other.Inside_x(self.N-1)) 
            if check1:
                # wrap other over
                xmM = [min(self.arr[0,0],other.arr[0,0]+self.N), max(self.arr[0,1],other.arr[0,1]+self.N)]
            elif check2:
                xmM = [min(other.arr[0,0],self.arr[0,0]+self.N), max(other.arr[0,1],self.arr[0,1]+self.N)]
            else:
                xmM = [min(other.arr[0,0],self.arr[0,0]), max(other.arr[0,1],self.arr[0,1])]

        self_over_y = self.arr[1,1] >= self.N
        other_over_y = other.arr[1,1] >= self.N
        if self_over_y and other_over_y:
            # Both wrapping - obvious
            ymM = [min(self.arr[1,0],other.arr[1,0]), max(self.arr[1,1],other.arr[1,1])]
        elif self_over_y and not other_over_y: 
            # See if it's better to wrap other over
            # i.e., check if left of other is in self
            check = self.Inside_y(other.arr[1,0]+self.N-1)
            if check:
                # left of other inside self, wrap over
                ymM = [min(self.arr[1,0],other.arr[1,0]+self.N), max(self.arr[1,1],other.arr[1,1]+self.N)]
            else:
                # not the case 
                ymM = [min(self.arr[1,0],other.arr[1,0]), max(self.arr[1,1],other.arr[1,1])]
        elif not self_over_y and other_over_y: 
            # See if it's better to wrap self over
            # i.e., check if left of other is in other
            check = other.Inside_y(self.arr[1,0]+self.N-1)
            if check:
                # left of self inside other, wrap over
                ymM = [min(other.arr[1,0],self.arr[1,0]+self.N), max(other.arr[1,1],self.arr[1,1]+self.N)]
            else:
                # not the case 
                ymM = [min(other.arr[1,0],self.arr[1,0]), max(other.arr[1,1],self.arr[1,1])]
        elif not self_over_y and not other_over_y: 
            # See if they connect at the boundary
            check1 = (other.Inside_y(0) and self.Inside_y(self.N-1)) 
            check2 = (self.Inside_y(0) and other.Inside_y(self.N-1)) 
            if check1:
                # wrap other over
                ymM = [min(self.arr[1,0],other.arr[1,0]+self.N), max(self.arr[1,1],other.arr[1,1]+self.N)]
            elif check2:
                ymM = [min(other.arr[1,0],self.arr[1,0]+self.N), max(other.arr[1,1],self.arr[1,1]+self.N)]
            else:
                ymM = [min(other.arr[1,0],self.arr[1,0]), max(other.arr[1,1],self.arr[1,1])]
        return Box([xmM, ymM],self.N)

     
class Cluster:
    def __init__(self,sites,boundaries,N,inhomogeneity=0,box=None,center=None,radius=None):
        self.index = GetNewIndexCluster()
        self.sites = SiteList(sites)
        """
        boundarySites only keep the record of sites near the boundaries.
        Initially they are the same as the sites, but RemoveInternalSites
        method will get rid of some of them to speed up computation.
        """
        self.boundarySites = self.sites.copy()
        """
        last_checked_size for keeping record of how often you remove internal
        sites.
        """
        self.last_checked_size = len(sites)
        self.boundaries = SortedList(boundaries)
        self.inhomogeneity = inhomogeneity
        """
        center and averages for dealing with boundary boxes and overlapping
        clusters
        """
        if center is None:
            self.center = numpy.average(self.sites,axis=0)
        else:
            self.center = center
        if radius is None:
            if len(sites) < 3:
                self.radius = len(sites)/2.
            else:
                self.radius = len(sites)
                #self.UpdateRadius(self.N)
        else:
            self.radius = radius

        if box is None:
            if len(sites) == 1:
                self.box = BoxWithPBC([[self.sites[0,0], self.sites[0,0]], \
                                   [self.sites[0,1], self.sites[0,1]]], N)
            else:
                self.box = BoxWithPBC([[self.sites[:,0].min(), self.sites[:,0].max()], \
                                   [self.sites[:,1].min(), self.sites[:,1].max()]], N)
        else:
            self.box = box

        self.N = N

    def UpdateRadius(self,N):
        """
        Simple version with boxes
        """
        self.center = self.box.GetCenter()
        self.radius = self.box.GetRadius()

    def GetInhomogeneity(self,N,vector,sites=None):
        """
        The cost energy of boundary is determined by inhomogeneities of 
        two associated clusters. As an important property, the inhomogeneity
        of the new cluster needs to be recalculated whenever an boundary gets
        removed.

        Usually, it will calculate the inhomogeneity using self.sites. 
        On the other hand, it returns the corresponding inhomogeneity for
        any given sites.
        """
        if sites is None:
            sites = self.sites
        """
        Calcualte the gaussian factor for site (x1,y1) away from the center (x0,y0). 
        """
        s = 0
        """
        If a small size, do it explicitly
        """
        if len(sites) == 2:
            x,y = sites[0]
            i,j = sites[1]
            dx = min(abs(x-i), abs(x+N-i), abs(x-i-N))
            dy = min(abs(y-j), abs(y+N-j), abs(y-j-N))
            s = numpy.sum((vector[x,y]-vector[i,j])**2)*kernel(dx,dy) 
            return s
        indices = sites.get_indices()
        mask = numpy.zeros((N,N))
        mask[indices[:,0],indices[:,1]] = 1.
        for (di,dj) in core_neighbors:
            newind = (indices+(di,dj))%N
            check = mask[newind[:,0],newind[:,1]]
            if check.sum() > 0:
                nz = check.nonzero()
                vec = vector[indices[:,0],indices[:,1]][nz]
                dvec = vector[newind[:,0],newind[:,1]][nz]
                s += numpy.sum((vec-dvec)**2)*kernel(di,dj)
        return s

    def GetBoundaryInhomogeneity(self,N,vector,sites1,sites2):
        """
        Boundary inhomogeneity computation function only works on the
        inhomogeneity between two clusters, effective for calculating

        cluster0.inhomogeneity + cluster1.inhomogeneity - (cluster0+cluster1).
        inhomogeneity

        used in boundary cost calculation.
        """
        #if len(sites1)*len(sites2)<N*N*N:
        return self.GetBoundaryInhomogeneityDirect(N,vector,sites1,sites2)
        """
        Direct alternative for very large sizes may be too slow. i.e.
        Direct one takes N_1 * N_2, whereas this one takes
        N_neighbors * N_1 or N_2.

        But due to python loop inefficiency direct calculation seems to
        be useful for most practical sizes.
        """
        s = 0
        
        indices1 = sites1.get_indices()
        indices2 = sites2.get_indices()
        mask = numpy.zeros((N,N))
        mask[indices2[:,0],indices2[:,1]] = 1.

        for (di,dj) in core_neighbors_redundant:
            newind = (indices1+(di,dj))%N
            check = mask[newind[:,0],newind[:,1]]
            if check.sum() > 0:
                nz = check.nonzero()
                vec = vector[indices1[:,0],indices1[:,1]][nz]
                dvec = vector[newind[:,0],newind[:,1]][nz]
                s += numpy.sum((vec-dvec)**2)*kernel(di,dj)
        return s
        
    def GetBoundaryInhomogeneityDirect(self,N,vector,sites1,sites2):
        """
        Direct calculation of boundary inhomogeneity.
        Fast for smaller sizes but takes N_1*N_2 space and time thus
        may need to avoid if possible for larger sizes
        """
        indices1 = sites1.get_indices()
        indices2 = sites2.get_indices()
     
        """
        Non-weave version for debugging or checking
        """ 
        """ 
        vec = vector[indices1[:,0],indices1[:,1]]
        dvec = vector[indices2[:,0],indices2[:,1]]
        sz1 = len(vec)
        sz2 = len(dvec)

        vec = vec.reshape((1,sz1,3))   
        dvec = dvec.reshape((1,sz2,3)).transpose(1,0,2) 
        vmdv = numpy.repeat(vec, sz2, axis=0)-numpy.repeat(dvec, sz1, axis=1)

        ind1 = indices1.reshape(tuple([1]+list(indices1.shape)))
        ind2 = indices2.reshape(tuple([1]+list(indices2.shape))).transpose(1,0,2)
        diff = numpy.repeat(ind1, sz2, axis=0)-numpy.repeat(ind2, sz1, axis=1)
        diff = diff%N
        diff = (diff>N/2)*(N-diff*2)+diff
        dist = diff[:,:,0]**2+diff[:,:,1]**2 

        s = numpy.sum(numpy.sum(vmdv**2,axis=2)*kernel2(dist))
        return s
        """ 

        sz1 = len(indices1)
        sz2 = len(indices2)
        code = """
#define clip(x) (((x)+%(N)d)%%%(N)d)
#define pclip(x) ((clip(x)>%(N)d/2) ? %(N)d-clip(x) : clip(x))
#define square(x) ((x)*(x))
#define v(x,y,k) *(vector+(x*%(N)d+y)*3+k)
            double s = 0.;
            for(int i=0; i<sz1; i++) {
                int x1 = indices1[i*2];
                int y1 = indices1[i*2+1];
                double vi0 = v(x1,y1,0);
                double vi1 = v(x1,y1,1);
                double vi2 = v(x1,y1,2);
                for(int j=0; j<sz2; j++) {
                    int x2 = indices2[j*2];
                    int y2 = indices2[j*2+1];
                    int dxy2 = square(pclip(x1-x2))+square(pclip(y1-y2));
                    double dvsq = square(v(x2,y2,0)-vi0)+square(v(x2,y2,1)-vi1)+square(v(x2,y2,2)-vi2);
                    if (dxy2 <= square(%(coresize)d+2)) {
                        s += 2.*dvsq*exp(-(double)dxy2/2./square(%(coresize)d));
                    }
                }
            }
            return_val = PyFloat_FromDouble(s);
        """ % {'coresize' : CORESIZE, 'N' : N}
        variables = ['sz1', 'sz2', 'vector', 'indices1', 'indices2']
        s = W.inline(code, variables, extra_compile_args=["-w"])
        return s

    def GetSize(self):
        """
        Calcualte the average size of cluster.
        """
        return numpy.sqrt(len(self.sites))

    def RemoveInternalSites(self, N):
        """
        Remove sites for which all neighbors in core_neighbors_redundant
        are included in the cluster it belongs to from boundarySites.
        """
        sites = self.sites
        b_sites = self.boundarySites
        indices = sites.get_indices()
        mask = numpy.zeros((N,N))
        mask[indices[:,0],indices[:,1]] = 1.
        b_indices = b_sites.get_indices()
        check = numpy.zeros(len(b_indices))
        max_c = 0 
        for (di,dj) in core_neighbors_redundant:
            newind = (b_indices+(di,dj))%N
            check += mask[newind[:,0],newind[:,1]]
            max_c += 1
        # Points that have at least one not inside nearby
        nz = (check<max_c).nonzero()
        print "Removed ", len(check) - len(nz[0]), " From ", len(check), '/', len(sites)
        self.boundarySites = b_sites[nz]
        return

"""
Give unique indices
"""
bd_index = 0
def GetNewIndex():
    global bd_index
    bd_index += 1
    return bd_index
cl_index = 0
def GetNewIndexCluster():
    global cl_index
    cl_index += 1
    return cl_index

class Boundary:
    def __init__(self,clusterPair,boundarysites,cost,length=1.):
        self.clusterPair = clusterPair 
        self.boundarysites = boundarysites
        self.cost = cost
        self.length = length
        self.index = GetNewIndex()

    def GetCost(self):
        """
        This is the place where the actual cost for the update procedure is
        calculated. Dividing by self.length gives the average cost per length
        of the boundary.
        """
        return self.cost/self.length

    def GetBoundaryLength(self):
        """
        Basically, for any site on the boundary, if it makes a bond only
        once, the length should be one; if it is used twice, the length should
        be sqrt(2); if it is used three times, the length should be 1+sqrt(2).
        Certainly, if it is used four times, it is a cluster with only one site, 
        enclosed by a boundary of length 2*sqrt(2).

        Noted that, this operation needs to repeat for boundary sites of both 
        pair clusters. The mutual length should be determined by their minimum. 
        """
        c1sites = [i[0] for i in self.boundarysites]
        c2sites = [i[1] for i in self.boundarysites]
        sites = [c1sites,c2sites]
        Ls = [] 
        for item in sites:
            L = 0.
            while (item!=[]):
                site = item[0]
                occurrences = item.count(site)
                num = len([item.pop(item.index(site)) for n in range(occurrences)])
                if num == 1:
                    L += 1.
                elif num == 2:
                    L += numpy.sqrt(2)
                elif num == 3:
                    L += 1.+numpy.sqrt(2)
                else:
                    L += 2*numpy.sqrt(2)
            Ls.append(L)
        return numpy.min(Ls)


class SteepestDecentClustering:
    def __init__(self,N,vector,boundaries,clusters,preJ=None):
        self.N = N
        self.vector = vector
        self.boundaries = {}
        for bd in boundaries:
            self.boundaries[bd.index] = bd
        """
        For larger sizes, use binned sorted dictionary since the sorted
        array become excessively large
        """
        if N >= 512:
            cost_dict_class = LargeValueSortedDict
        else:
            cost_dict_class = ValueSortedDict
        """
        Use a value sort augmented dictionary to easily find maximum cost
        """
        self.costs = cost_dict_class([(bd.index,bd.cost) for bd in boundaries])
        self.clusters = {}
        for cl in clusters:
            self.clusters[cl.index] = cl
        """
        If not given preJ, J should be assigned by the average initial values
        of boundaries.
        """
        if preJ is None:
            self.J = self.InitializeBoundaries()
        else:
            self.J = preJ
        """
        Cost is given by both inhomogeneities of clusters and lenght of boundary.
        """
        for b in self.boundaries:
            self.boundaries[b].cost += self.J*self.boundaries[b].length   
            self.costs[b] = self.boundaries[b].GetCost()

    def InitializeBoundaries(self):
        numberOfbonds = sum([self.boundaries[b].length for b in self.boundaries])
        costOfbonds = sum([self.boundaries[b].cost for b in self.boundaries])
        return costOfbonds/numberOfbonds

    def AddNewBoundary(self,clusterPair,length,boundarysites):
        """
        We need a new object of boundary class and fill its properties
        with all from the removed boundary. And the cost of this new
        boundary needs to be recalculated. 
        The new boundary need to be added to the system.
        """
        newb = Boundary(clusterPair,boundarysites,0.,length)
        cc0,cc1 = clusterPair
        """
        newb.cost = self.J*newb.length + cc0.inhomogeneity + cc1.inhomogeneity -\
                    cc0.GetInhomogeneity(self.N,self.vector,sites=cc0.sites+cc1.sites)
        """
        newb.cost = self.J*newb.length -\
                     cc1.GetBoundaryInhomogeneity(self.N,self.vector,cc0.boundarySites,cc1.boundarySites)  
        self.costs[newb.index] = newb.GetCost()
        self.boundaries[newb.index] = newb
        clusterPair[0].boundaries.append(newb)
        clusterPair[1].boundaries.append(newb)
    
    def CombineBoundaries(self,oldBD,removedBD):
        """
        Transfer the properties of the removed boudnary to the
        existed boundary. 
        Noted that the inhomogeneity part of the cost doesn't
        need to be changed. 
        """
        oldBD.boundarysites += removedBD.boundarysites
        oldBD.length += removedBD.length
        oldBD.cost += removedBD.length*self.J
        self.costs[oldBD.index] = oldBD.GetCost()
 
    def TransferBoundaries(self,c0,c1):
        """
        Loop over every boundary of c0.
        Since every bounday has a certain cluster pair,
        we replace c0 with c1 for every boundary of c0 and
        check whether the boundary has already existed.
        """
        bs0 = c0.boundaries
        pairsOfc1 = dict([(bb.clusterPair,i) for i,bb in enumerate(c1.boundaries)])
        for bd in bs0:
            self.boundaries.pop(bd.index)
            self.costs.pop(bd.index)
            sc0,sc1 = bd.clusterPair
            if c0==sc0:
                c = sc1
            else:
                c = sc0
            c.boundaries.remove(bd)
            pair = (c1,c) 
            if  (pair not in pairsOfc1) and ((pair[1],pair[0]) not in pairsOfc1): 
                """
                If the boundary is not in the list of boundaries of c1, we need 
                to add this new boundary to the whole lists of boundarysites and
                boundary lists of both clusters in a pair.
                """
                self.AddNewBoundary(pair,bd.length,bd.boundarysites)
            else:
                """
                Otherwise, the boundary will be added to the existed boundary of c1.
                """
                if pair in pairsOfc1:
                    #nb = c1.boundaries[pairsOfc1.index(pair)]
                    nb = c1.boundaries[pairsOfc1[pair]]
                if (pair[1],pair[0]) in pairsOfc1:
                    #nb = c1.boundaries[pairsOfc1.index((pair[1],pair[0]))]
                    nb = c1.boundaries[pairsOfc1[(pair[1],pair[0])]]
                self.CombineBoundaries(nb,bd)

    def GetDists(self, c0, c1):
        dist = numpy.zeros((len(c1.boundaries),2))
        for i,b in enumerate(c1.boundaries):
            cc0,cc1 = b.clusterPair
            if (cc0 == c1):
                cc1,cc0 = b.clusterPair
            dist[i] = c0.center-cc0.center
        dist = dist%self.N
        dist = (dist>self.N/2)*(self.N-2.*dist)+dist
        dist = numpy.sqrt(dist[:,0]**2+dist[:,1]**2)
        return list(dist)

    def UpdateClusters(self,c0,c1,b):
        """
        Choose c0 to be the smaller cluster
        """
        if len(c0.sites) > len(c1.sites):
            c_tmp = c1
            c1 = c0
            c0 = c_tmp
        """
        Remove c0 from clusters list.
        """
        self.clusters.pop(c0.index)
        """
        sites of c0 are added to c1.
        """
        c0_site_num = len(c0.sites)
        c1_site_num = len(c1.sites)
        c1.sites += c0.sites
        c1.boundarySites += c0.boundarySites
        c1.box = c1.box.MergeWith(c0.box)
        c1.UpdateRadius(self.N)

        if len(c1.sites) > CORESIZE*CORESIZE*16 and\
            len(c0.sites) > CORESIZE*CORESIZE*4 and\
            c1_site_num > CORESIZE*CORESIZE*4 or\
            c1.last_checked_size < c1_site_num+c0_site_num - CORESIZE*CORESIZE*8:
            # Remove internal sites to be no longer evaluated for boundary
            # inhomogeneity
            c1.RemoveInternalSites(self.N)
            c1.last_checked_size = len(c1.sites)
        """
        Inhomogeneity of c1 is evaluated after absorbing c0.
        """
        """
        You don't have to recalculate it since you know this info
        from the boundary cost.
        """
        c1.inhomogeneity = self.J*b.length+c0.inhomogeneity+c1.inhomogeneity-b.cost
        #c1.inhomogeneity = c1.GetInhomogeneity(self.N,self.vector)
        """
        All existing boundaries of c1 are updated with recalculated costs. 
        """
        dists = self.GetDists(c0, c1)
        for b in c1.boundaries:
            cc0,cc1 = b.clusterPair
            if (cc0 == c1):
                cc1,cc0 = b.clusterPair
            dist = dists.pop(0)
            if dist < c0.radius + cc0.radius + CORESIZE * 2:
                b.cost = self.J*b.length -\
                         c1.GetBoundaryInhomogeneity(self.N,self.vector,cc0.boundarySites,cc1.boundarySites)  
                self.costs[b.index] = b.GetCost()
            """
            b.cost = self.J*b.length + (cc0.inhomogeneity+cc1.inhomogeneity-\
                     c1.GetInhomogeneity(self.N,self.vector,sites=cc0.sites+cc1.sites))  
            """
        return c0,c1
 
    def RemoveBoundary(self,index):
        """
        Remove the boundary from the whole boundaries list.
        """
        b = self.boundaries.pop(index)
        self.costs.pop(index)
        """
        Remove the boundary from both boundaries lists of two
        clusters associated with this certain boundary.
        """
        c0,c1 = b.clusterPair
        c0.boundaries.remove(b)
        c1.boundaries.remove(b)
        """
        Transfer all properties but boundaries of cluster c0 to cluster c1 .
        """
        c0,c1=self.UpdateClusters(c0,c1,b)
        """
        Transfer boundaries of cluster c0 to cluster c1.
        """
        self.TransferBoundaries(c0,c1)

    def Run(self):
        """
        On every step, the most expensive boundary is found out and
        removed until the associated cost gets negative.
        """
        print "Start to run!"
        count = 1
        while True:
            count += 1
            if count % self.N == 0:
                print count
            #if count > 5000:
            #    break
            """
            indices = self.boundaries.keys()
            costs = numpy.array([self.boundaries[b].cost for b in indices])
            index = costs.argmax()
            maxcost = costs[index]
            """
            index = self.costs.argmax()
            maxcost = self.costs[index]
            if maxcost <= 0.:
                break
            self.RemoveBoundary(index)
            #self.RemoveBoundary(indices[index])
        print self.J, " The total number of clusters is ",len(self.clusters)


class Initialization:
    def __init__(self,N,vector,Nt=None):
        self.N = N
        self.vector = vector
        self.Nt = Nt

    def GenerateClustersFromEverySite(self):
        """
        Initially, every cluster has zero inhomogeneity, so the initial
        boundary cost value is assigned by different inhomogeneity from 
        the only nearest neighbor cluster. 
        It is simplified by multiplying the costFactor to the deltaTheta.
        """
        N = self.N
        siteList = [[(i,j)] for i in range(N) for j in range(N)]
        clusters = [Cluster(item,[],N) for item in siteList]
        #costFactor = -numpy.exp(-1./2./CORESIZE**2)*2.0
        costFactor = -kernel(1,0)
        horizontalBD = [Boundary((clusters[i*N+j],clusters[i*N+(j+1)%N]),[((i,j),(i,(j+1)%N))],\
                                 costFactor*self.CalculateEuclideanDistanceOfTwoVectors(self.vector[i,j],\
                                 self.vector[i,(j+1)%N]),1.) for i in range(N) for j in range(N)]
        verticalBD = [Boundary((clusters[i*N+j],clusters[((i+1)%N)*N+j]),[((i,j),((i+1)%N,j))],\
                               costFactor*self.CalculateEuclideanDistanceOfTwoVectors(self.vector[i,j],\
                               self.vector[(i+1)%N,j]),1.) for i in range(N) for j in range(N)]
        boundaries = horizontalBD+verticalBD
        for i in range(N):
            for j in range(N):
                clusters[i*N+j].boundaries = [horizontalBD[i*N+j],horizontalBD[i*N+(j-1)*(j>0)+(N-1)*(j==0)],\
                                              verticalBD[i*N+j],verticalBD[((i-1)*(i>0)+(N-1)*(i==0))*N+j]]
        return boundaries,clusters

    def GenerateDynamicalClustersFromEverySite(self,temporalFactor=1.):
        N = self.N
        Nt = self.Nt
        siteList = [[(n,i,j)] for n in range(Nt) for i in range(N) for j in range(N)]
        clusters = [Cluster(item,[],N) for item in siteList]
        costFactor = -numpy.exp(-1./2./CORESIZE**2) 
        horizontalBD = [Boundary((clusters[n*N**2+i*N+j],clusters[n*N**2+i*N+(j+1)%N]),[((n,i,j),(n,i,(j+1)%N))],\
                                 costFactor*self.CalculateEuclideanDistanceOfTwoVectors(self.vector[n,i,j],self.vector[n,i,(j+1)%N])\
                                 ,1.) for n in range(Nt) for i in range(N) for j in range(N)]
        verticalBD = [Boundary((clusters[n*N**2+i*N+j],clusters[n*N**2+((i+1)%N)*N+j]),[((n,i,j),(n,(i+1)%N,j))],\
                               costFactor*self.CalculateEuclideanDistanceOfTwoVectors(self.vector[n,i,j],self.vector[n,(i+1)%N,j])\
                               ,1.) for n in range(Nt) for i in range(N) for j in range(N)]
        temporalBD = [Boundary((clusters[n*N**2+i*N+j],clusters[(n+1)*N**2+i*N+j]),[((n,i,j),(n+1,i,j))],\
                               costFactor*temporalFactor*self.CalculateEuclideanDistanceOfTwoVectors(self.vector[n,i,j],self.vector[n+1,i,j])\
                               ,1.) for n in range(Nt-1) for i in range(N) for j in range(N)]
        boundaries = horizontalBD + verticalBD + temporalBD 
        for n in range(1,Nt-1):
            for i in range(N):
                for j in range(N):
                    clusters[n*N**2+i*N+j].boundaries = [horizontalBD[n*N**2+i*N+j],horizontalBD[n*N**2+i*N+(j-1)*(j>0)+(N-1)*(j==0)],\
                                                         verticalBD[n*N**2+i*N+j],verticalBD[n*N**2+((i-1)*(i>0)+(N-1)*(i==0))*N+j],\
                                                         temporalBD[n*N**2+i*N+j],temporalBD[(n-1)*N**2+i*N+j]]
        for i in range(N):
            for j in range(N):
                clusters[i*N+j].boundaries = [horizontalBD[i*N+j],horizontalBD[i*N+(j-1)*(j>0)+(N-1)*(j==0)],\
                                              verticalBD[i*N+j],verticalBD[((i-1)*(i>0)+(N-1)*(i==0))*N+j],\
                                              temporalBD[i*N+j]]
                clusters[(Nt-1)*N**2+i*N+j].boundaries = [horizontalBD[(Nt-1)*N**2+i*N+j],\
                                                          horizontalBD[(Nt-1)*N**2+i*N+(j-1)*(j>0)+(N-1)*(j==0)],\
                                                          verticalBD[(Nt-1)*N**2+i*N+j],\
                                                          verticalBD[(Nt-1)*N**2+((i-1)*(i>0)+(N-1)*(i==0))*N+j],\
                                                          temporalBD[(Nt-2)*N**2+i*N+j]]
        return boundaries,clusters

    def CalculateEuclideanDistanceOfTwoVectors(self,vec1,vec2):
        return (numpy.sum((vec1-vec2)**2))



####################################################################
### Tools of visualization 
####################################################################
def ReconstructionWithBoundary(indexmap,orientationmap,outputDir):
    N = indexmap.shape[0]
    newindexmap = numpy.empty(numpy.array(indexmap.shape)*4)
    for i in range(4):
        for j in range(4):
            newindexmap[i::4,j::4] = indexmap
    neworientationmap = numpy.empty([4*N,4*N,3])
    for k in range(3):
        for i in range(4):
            for j in range(4):
                neworientationmap[i::4,j::4,k] = orientationmap[:,:,k]
    xs,ys = (FindPointsOnBoundaries(indexmap)).nonzero()
    for (i,j) in zip(xs,ys):
        temp = [indexmap[i,j]==indexmap[(i-1+N)%N,j],indexmap[i,j]==indexmap[(i+1)%N,j],\
                indexmap[i,j]==indexmap[i,(j-1+N)%N],indexmap[i,j]==indexmap[i,(j+1)%N]]
        pos = [[(4*i,4*j),(4*i+1,4*j+4)],[(4*i+3,4*j),(4*i+4,4*j+4)],\
               [(4*i,4*j),(4*i+4,4*j+1)],[(4*i,4*j+4),(4*i+4,4*j+4)]]
        for n in range(4):
            if not temp[n]:
                newindexmap[pos[n][0][0]:pos[n][1][0],pos[n][0][1]:pos[n][1][1]] = -1
    for k in range(3):
        neworientationmap[:,:,k] *= (newindexmap!=-1)
    """
    import pylab
    pylab.figure()
    pylab.imshow(neworientationmap,interpolation='nearest')
    pylab.savefig(outputDir+'withboundary.png')
    pylab.figure()
    pylab.imshow(orientationmap,interpolation='nearest')
    pylab.savefig(outputDir+'orientation.png')
    """
    """
    Use PIL to plot for larger sizes, since we want to be able to draw
    pixel by pixel
    """
    from PIL import Image
    pilImage = Image.fromarray(neworientationmap.astype('uint8'), 'RGB')
    pilImage.save(outputDir+'withboundary.png')
    pilImage = Image.fromarray(orientationmap.astype('uint8'), 'RGB')
    pilImage.save(outputDir+'orientation.png')

    indexmap.tofile(outputDir+'indexmap.dat')
    orientationmap.tofile(outputDir+'orientationmap.dat')
    neworientationmap.tofile(outputDir+'withboundary.dat') 

def GetIndexmapFromClusters(N,clusters):
    indexmap = numpy.zeros((N,N),int)
    i = 0
    for c in clusters:
        for pos in c.sites:
            indexmap[tuple(pos)] = i
        i+=1
    return indexmap

def GetDynamicalIndexmapFromClusters(Nt,N,clusters):
    indexmap = numpy.zeros((Nt,N,N),int)
    i = 0
    for c in clusters:
        for pos in c.sites:
            indexmap[pos] = i
        i += 1 
    return indexmap

def FindPointsOnBoundaries(indexmap):
    return  1-(indexmap==numpy.roll(indexmap, 1,0))*(indexmap==numpy.roll(indexmap,-1,0))*\
              (indexmap==numpy.roll(indexmap, 1,1))*(indexmap==numpy.roll(indexmap,-1,1))

def ShowDynamicalResults(indexmap,output):
    import pylab
    pylab.figure()
    for i in range(4):
        pylab.subplot('22'+str(i+1))
        pylab.imshow(indexmap[i],interpolation='nearest')
    pylab.savefig(output+'.png')

def SaveResults(indexmap,rodriguesmap,boundary,output):
    import pylab
    pylab.figure()
    pylab.imshow(indexmap,interpolation='nearest')
    pylab.savefig(output+'_indexmap.png')
    for i in range(3):
        rodriguesmap[:,:,i] *= (1-boundary)
    rodriguesmap.tofile(output+'.dat')
    pylab.figure()
    pylab.imshow(rodriguesmap,interpolation='nearest')
    pylab.savefig(output+'.png')

def Test():
    Nt = 4
    N = 8 
    rod = numpy.fromfile("No Backup/test/ClusteringDynamicalTest.dat").reshape(Nt,N,N,3) 
    orientationmap = rod + 1.
    system = Initialization(N,rod,Nt)
    scheme = 'average'
    preJ = 1.
    boundaries,clusters = system.GenerateDynamicalClustersFromEverySite(10.) 
    clustering = SteepestDecentClustering(boundaries,clusters,scheme,preJ)
    clustering.Run()
    outputDir = "No Backup/test/SteepestDecent_Dynamical10_0_fromIndividualPts_Test_"+scheme+'_preJ='+str(preJ).replace('.','_')
    indexmap = GetDynamicalIndexmapFromClusters(Nt,N,clustering.clusters.values())
    ShowDynamicalResults(indexmap,outputDir)

def RotateVector_45degree(newN,vec):
    from scipy import ndimage as ndi
    from scipy import signal
    N = vec.shape[0]
    newvec = numpy.zeros((newN,newN,3),float)
    for i in range(3):
        temp = signal.resample(signal.resample(vec[:,:,i],N*4,axis=0),N*4,axis=1) 
        temp = ndi.rotate(temp,45,mode='wrap')
        newvec[:,:,i] = signal.resample(signal.resample(temp,newN,axis=0),newN,axis=1) 
    return newvec

def Run():
    import PlasticitySystem
    import FieldInitializer
    import OrientationField
    N=128
    #statefile = "../../CleanOutSystem/Plasticity/No Backup/data/CU/"+str(N)+"/CU_S_0_2D"+str(N)+"_GlideOnly_betaP_t=9_0.dat"
    #state = FieldInitializer.NumpyTensorInitializer((N,N),statefile,bin=False)
    t, state = FieldInitializer.LoadState("BetaP2D_LLF_Init_%d.save" % N, 10.0)
    state = FieldInitializer.ReformatState(state)
    #statefile = "No Backup/result/BetaP2D_CU_Init_1024_Seed_2.save"
    #t, state = FieldInitializer.LoadState(statefile) 
    rodrigues = state.CalculateRotationRodrigues()
    rod = numpy.zeros((N,N,3),float)
    rod[:,:,0] = rodrigues['x']
    rod[:,:,1] = rodrigues['y']
    rod[:,:,2] = rodrigues['z']
    rotate = False
    if rotate :
        N = int(1.414*N)
        # Make it even
        N -= N%2
        rod = RotateVector_45degree(N,rod)
    orientationmap = OrientationField.RodriguesToUnambiguousColor(rod[:,:,0],rod[:,:,1],rod[:,:,2])
    opt = 1
    if opt == 0:
        K=50
        system = KmeansClustering3D()
        system.RunClustering(N,rod,K)
        outputDir = "No Backup/test/KmeansClustering3D_"+str(N)+"K_"+str(K)
        ReconstructionWithBoundary(system.indexmap,orientationmap,outputDir)
    elif opt == 1:
        preJ = 0.5
        system = Initialization(N,rod)
        boundaries,clusters = system.GenerateClustersFromEverySite() 
        clustering = SteepestDecentClustering(N,rod,boundaries,clusters,preJ)
        clustering.Run()
        outputDir = "NewClustering_Core_"+str(CORESIZE)+"_N_"+str(N)+"_fromIndividualPts_preJ="+str(preJ).replace('.','_')
        indexmap = GetIndexmapFromClusters(N,clustering.clusters.values())
        ReconstructionWithBoundary(indexmap,orientationmap,outputDir)
    else:
        pass

if __name__=='__main__':
    """
    import hotshot
    import hotshot.stats
    prof = hotshot.Profile("profile.prof")
    result = prof.runcall(Run)
    prof.close()
    stats = hotshot.stats.load("profile.prof")
    stats.strip_dirs()
    stats.sort_stats("time","calls")
    stats.print_stats(50)
    import sys
    sys.exit(0)
    """
    Run()
    """
    """
    #Test()
