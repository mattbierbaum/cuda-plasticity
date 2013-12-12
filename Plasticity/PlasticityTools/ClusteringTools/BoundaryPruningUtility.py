import scipy.weave as W
import numpy

from SortedList import *

core_neighbors = [(0,1),(1,0)]
core_neighbors_redundant = [(-1,0),(0,-1),(0,1),(1,0)]


"""
Give unique indices for boundaries and clusters.
"""
bd_index = 0
def GetNewIndexBoundary():
    global bd_index
    bd_index += 1
    return bd_index

cl_index = 0
def GetNewIndexCluster():
    global cl_index
    cl_index += 1
    return cl_index

    
##################################################################
### Class of cluster 
##################################################################
class Cluster:
    def __init__(self,sites,boundaries,N,inhomogeneity=0,box=None,center=None,radius=None,perimeter=None,perimeterToArea=None):
        """
        Unique index for a new cluster object. 
        """
        self.index = GetNewIndexCluster()
        """
        Positions of every site in the cluster.
        """
        self.sites = SiteList(sites)
        """
        All boundaries of the cluster.
        """
        self.boundaries = SortedList(boundaries)
        """
        Size of the whole system consisting all clusters.
        """
        self.N = N
        """
        Inhomogeneity of the cluster.
        """
        self.inhomogeneity = inhomogeneity
        """
        Box of the cluster with periodic boundary conditions.
        """
        if box is None:
            if len(sites) == 1:
                self.box = BoxWithPBC([[self.sites[0,0], self.sites[0,0]], \
                                   [self.sites[0,1], self.sites[0,1]]], N)
            else:
                self.box = BoxWithPBC([[self.sites[:,0].min(), self.sites[:,0].max()], \
                                   [self.sites[:,1].min(), self.sites[:,1].max()]], N)
        else:
            self.box = box
        """
        center and radius for dealing with boundary boxes and overlapping clusters.

        Position of the center of the cluster.
        """
        if center is None:
            self.center = numpy.average(self.sites,axis=0)
        else:
            self.center = center
        """
        Maximum size of the cluster.
        """
        if radius is None:
            if len(sites) < 3:
                self.radius = len(sites)/2.
            else:
                self.radius = len(sites)
        else:
            self.radius = radius
        """
        The perimeter of the cluster.
        """
        self.perimeter = perimeter
        """
        The ratio of perimeter and area of the cluster.
        """
        self.perimeterToArea = perimeterToArea 
        """
        boundarySites only keep the record of sites near the boundaries.
        Initially they are the same as the sites, but RemoveInternalSites
        method will get rid of some of them to speed up computation.
        """
        self.boundarySites = self.sites.copy()
        """
        last_checked_size for keeping record of how often you remove internal sites.
        """
        self.last_checked_size = len(sites)
                
    def UpdateRadius(self):
        """
        Simple version with boxes
        """
        self.center = self.box.GetCenter()
        self.radius = self.box.GetRadius()

    def GetPerimeter(self,sites=None):
        """
        Find out the associated perimeter of this cluster.
        """
        if sites is None:
            indices = self.sites.get_indices()
        else:
            indices = sites.get_indices()
        area = len(indices)
        if area <= 3:
            return 2.*(area+1.)
        else:
            cluster = numpy.zeros((self.N,self.N))
            cluster[indices[:,0],indices[:,1]] = 1.
            mask = (cluster>0)
            newcluster = numpy.copy(cluster)
            newcluster += 1.*(cluster==numpy.roll(cluster,1,0))*mask
            newcluster += 1.*(cluster==numpy.roll(cluster,1,1))*mask
            newcluster += 1.*(cluster==numpy.roll(cluster,-1,0))*mask
            newcluster += 1.*(cluster==numpy.roll(cluster,-1,1))*mask
            """
            Any site with n (1<=n<=4) neighbors will contribute (4-n) to the perimeter;
            """
            perimeter = numpy.sum(newcluster==2.)*3.+numpy.sum(newcluster==3.)*2.+numpy.sum(newcluster==4.)
            return perimeter 

    def GetPerimeterToArea(self,sites=None):
        """
        Find out the associated perimeter/area of this cluster.
        """
        if sites is None:
            indices = self.sites.get_indices()
        else:
            indices = sites.get_indices()
        area = len(indices)
        return self.GetPerimeter(sites)/float(area)

    def GetInhomogeneity(self,vector,sites=None):
        """
        The inhomogeneity of boundary is determined by inhomogeneities of 
        two associated clusters. As an important property, the inhomogeneity
        of the new cluster needs to be recalculated whenever an boundary gets
        removed.

        Usually, it will calculate the inhomogeneity using self.sites. 
        On the other hand, it returns the corresponding inhomogeneity for
        any given sites.
        """
        if sites is None:
            sites = self.sites
        N = self.N
        """
        Calcualte the gaussian factor for site (x1,y1) away from the center (x0,y0). 
        """
        s = 0
        factor = 0
        """
        If a small size, do it explicitly
        """
        if len(sites) == 2:
            x,y = sites[0]
            i,j = sites[1]
            dx = min(abs(x-i), abs(x+N-i), abs(x-i-N))
            dy = min(abs(y-j), abs(y+N-j), abs(y-j-N))
            s = numpy.sum((vector[x,y]-vector[i,j])**2) 
            return s
        indices = sites.get_indices()
        mask = numpy.zeros((N,N))
        mask[indices[:,0],indices[:,1]] = 1.
        for (di,dj) in core_neighbors:
            newind = (indices+(di,dj))%N
            check = mask[newind[:,0],newind[:,1]]
            if check.sum() > 0:
                factor += check.sum()
                nz = check.nonzero()
                vec = vector[indices[:,0],indices[:,1]][nz]
                dvec = vector[newind[:,0],newind[:,1]][nz]
                s += numpy.sum((vec-dvec)**2)
        return s/factor

    def GetBoundaryInhomogeneity(self,vector,sites1,sites2,fast=True):
        """
        Boundary inhomogeneity computation function only works on the
        inhomogeneity between two clusters, effective for calculating

        (cluster0+cluster1).inhomogeneity - cluster0.inhomogeneity - cluster1.inhomogeneity 
 
        used in boundary inhomogeneity calculation.
        """
        if fast:
            return self.GetBoundaryInhomogeneityDirect(vector,sites1,sites2)
        """
        Direct alternative for very large sizes may be too slow. i.e.
        Direct one takes N_1 * N_2, whereas this one takes
        N_neighbors * N_1 or N_2.

        But due to python loop inefficiency direct calculation seems to
        be useful for most practical sizes.
        """
        s = 0
        factor = 0
        
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
                s += numpy.sum((vec-dvec)**2)
                factor += check.sum() 
        return s/factor
        
    def GetBoundaryInhomogeneityDirect(self,vector,sites1,sites2,usingweave=True):
        """
        Direct calculation of boundary inhomogeneity.
        Fast for smaller sizes but takes N_1*N_2 space and time thus
        may need to avoid if possible for larger sizes
        """
        N = self.N

        indices1 = sites1.get_indices()
        indices2 = sites2.get_indices()
     
        if usingweave:
            sz1 = len(indices1)
            sz2 = len(indices2)
            code = """
                #define clip(x) (((x)+%(N)d)%%%(N)d)
                #define pclip(x) ((clip(x)>%(N)d/2) ? %(N)d-clip(x) : clip(x))
                #define square(x) ((x)*(x))
                #define v(x,y,k) *(vector+(x*%(N)d+y)*3+k)
                double s = 0.;
                double factor = 0.;
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
                        if (dxy2 <= 1.) {
                            factor +=2.;
                            s += 2.*dvsq;
                        }
                    }
                }
                return_val = PyFloat_FromDouble(s/factor);
            """ % { 'N' : N}
            variables = ['sz1', 'sz2', 'vector', 'indices1', 'indices2']
            s = W.inline(code, variables, extra_compile_args=["-w"])
            return s
        else:
            """
            Non-weave version for debugging or checking
            """ 
            vec = vector[indices1[:,0],indices1[:,1]]
            dvec = vector[indices2[:,0],indices2[:,1]]
            sz1 = len(vec)
            sz2 = len(dvec)
            vec = vec.reshape((1,sz1,3))   
            dvec = dvec.reshape((1,sz2,3)).transpose(1,0,2) 
            vmdv = numpy.repeat(vec, sz2, axis=0)-numpy.repeat(dvec, sz1, axis=1)
            s = numpy.sum(numpy.sum(vmdv**2,axis=2))
            return s

    def RemoveInternalSites(self):
        """
        Remove sites for which all neighbors in core_neighbors_redundant
        are included in the cluster it belongs to from boundarySites.
        """
        N = self.N
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
        #print "Removed ", len(check) - len(nz[0]), " From ", len(check), '/', len(sites)
        self.boundarySites = b_sites[nz]

    def GetSize(self):
        """
        Calcualte the average size of cluster.
        """
        return numpy.sqrt(len(self.sites))


##################################################################
### Class of boundary 
##################################################################
class Boundary:
    def __init__(self,clusterPair,deltaInhomogeneity=None,length=None,areaToPerimeterOfCombinedCluster=None,\
                 deltaPerimeter=None, id=0):
        """
        Unique index for a new boundary object. 
        """
        self.index = GetNewIndexBoundary()
        """
        Boundary is determined by a pair of clusters.
        """
        self.clusterPair = clusterPair 
        """
        The changing inhomogeneity when removing a boundary.
        """
        self.deltaInhomogeneity = deltaInhomogeneity 
        """
        Length of a boundary.
        """
        self.length = length
        """
        The ratio of area and perimeter of the combined cluster when removing a boundary.
        """
        self.areaToPerimeterOfCombinedCluster = areaToPerimeterOfCombinedCluster 
        """
        The changing perimeter of the combined cluster when removing a boundary.
        """
        self.deltaPerimeter = deltaPerimeter 

        self.id = id

    def GetAverageCost(self,cost):
        """
        This is the place where the actual cost for the update procedure is
        calculated. Dividing by self.length gives the average cost per length
        of the boundary.
        """
        return cost/self.length

    def GetDeltaInhomogeneity(self,vectorField):
        """
        Calcualte the changing inhomogeneity of the combined cluster when removing the boundary.
        """ 
        c0,c1 = self.clusterPair
        return c0.GetBoundaryInhomogeneity(vectorField,c0.boundarySites,c1.boundarySites)

    def GetAreaToPerimeterOfCombinedCluster(self):
        """
        Calcualte the area/perimeter of the combined cluster when removing the boundary.
        """ 
        c0,c1 = self.clusterPair
        return 1./c0.GetPerimeterToArea(sites=c0.sites+c1.sites) 

    def GetDeltaPerimeterOfCombinedCluster(self):
        """
        Calcualte the changing perimeter of the combined cluster when removing the boundary.
        """ 
        c0,c1 = self.clusterPair
        return -(c0.GetPerimeter(sites=c0.sites+c1.sites)-c0.GetPerimeter()-c1.GetPerimeter()) 

    def GetBoundaryLength(self):
        """
        For the pair of clusters, roll one of them along fully four directions.
        Count the total number of overlaped sites of the other cluster. This number
        gives the length of the boundary approximately.
        """
        c0,c1 = self.clusterPair
        c0_indices = c0.sites.get_indices()
        c1_indices = c1.sites.get_indices()
        N = c0.N
        system0 = numpy.zeros((N,N))
        system0[c0_indices[:,0],c0_indices[:,1]] = 1.
        system1 = numpy.copy(system0)
        system1[c1_indices[:,0],c1_indices[:,1]] = 1.
        mask = (system1==1.)
        newsystem = numpy.roll(system0,1,0)*mask
        newsystem += numpy.roll(system0,1,1)*mask
        newsystem += numpy.roll(system0,-1,0)*mask
        newsystem += numpy.roll(system0,-1,1)*mask
        newsystem = (newsystem>0).astype(float) 
        return numpy.sum(newsystem) 


##################################################################
###  Utilities for computational efficiency.
##################################################################
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
