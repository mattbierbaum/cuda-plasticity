from SteepestDecentBoundaryPruningMethod import *
from BoundaryPruningUtility import *

CORESIZE = 1 # For checking whether to remove internal sites.

import sys
sys.path += [".."]
from Plasticity import PlasticitySystem
from Plasticity.FieldInitializers import FieldInitializer

import numpy
from SortedList import *

class ClusteringMixed(SteepestDecentBoundaryPruningMethod):
    def __init__(self,N,vector,boundaries,clusters,preJ=None):
        self.N = N
        self.vector = vector
        self.boundaries = {}
        for bd in boundaries:
            bd.deltaPerimeter = 2.*bd.length
            self.boundaries[bd.index] = bd
        self.clusters = {}
        for cl in clusters:
            cl.perimeter = numpy.sum([bd.length for bd in c.boundaries]) 
            cl.perimeterToArea = cl.perimeter/float(len(cl.sites)) 
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
        self.costs = cost_dict_class([(bd.index,None) for bd in boundaries])
        """
        Cost is given by both inhomogeneities of clusters and length of boundary.
        """
        for b in self.boundaries:
            self.costs[b] = (self.boundaries[b].deltaInhomogeneity+self.J,max([bd.clusterPair[0].perimeterToArea,\
                             bd.clusterPair[1].perimeterToArea]),bd.deltaPerimeter)
        SteepestDecentBoundaryPruningMethod.__init__(self,self.clusters,self.boundaries,self.costs,self.J)

    def InitializeBoundaries(self):
        numberOfbonds = sum([self.boundaries[b].length for b in self.boundaries])
        costOfbonds = sum([self.boundaries[b].deltaInhomogeneity for b in self.boundaries])
        return costOfbonds/numberOfbonds

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
        c1.perimeter += c0.perimeter - 2.*b.length
        c1.perimeterToArea = c1.perimeter/float(len(c1.sites))
        c1.boundarySites += c0.boundarySites
        c1.box = c1.box.MergeWith(c0.box)
        c1.UpdateRadius()
        if len(c1.sites) > CORESIZE*CORESIZE*16 and\
            len(c0.sites) > CORESIZE*CORESIZE*4 and\
            c1_site_num > CORESIZE*CORESIZE*4 or\
            c1.last_checked_size < c1_site_num+c0_site_num - CORESIZE*CORESIZE*8:
            c1.RemoveInternalSites()
            c1.last_checked_size = len(c1.sites)
        """
        Inhomogeneity of c1 is evaluated after absorbing c0.

        You don't have to recalculate it since you know this info
        from the boundary cost.
        """
        c1.inhomogeneity = c0.inhomogeneity+c1.inhomogeneity-b.deltaInhomogeneity
        """
        All existing boundaries of c1 are updated with recalculated costs. 
        """
        dists = self.GetDists(c0, c1)
        for b in c1.boundaries:
            b.deltaPerimeter = 2.*b.length
            cc0,cc1 = b.clusterPair
            if (cc0 == c1):
                cc1,cc0 = b.clusterPair
            dist = dists.pop(0)
            if dist < c0.radius + cc0.radius + CORESIZE * 2:
                b.deltaInhomogeneity = -c1.GetBoundaryInhomogeneity(self.vector,cc0.boundarySites,cc1.boundarySites)  
                self.costs[b.index] = (b.deltaInhomogeneity+self.J,max([cc0.perimeterToArea,cc1.perimeterToArea]),b.deltaPerimeter) 
        return c0,c1

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

    def AddNewBoundary(self,clusterPair,removedBD):
        newb = Boundary(clusterPair,0.,removedBD.length, id=removedBD.id)
        cc0,cc1 = clusterPair
        newb.deltaInhomogeneity = -cc1.GetBoundaryInhomogeneity(self.vector,cc0.boundarySites,cc1.boundarySites)  
        newb.deltaPerimeter = 2.*newb.length
        self.costs[newb.index] = (newb.deltaInhomogeneity+self.J,max([cc0.perimeterToArea,cc1.perimeterToArea]),newb.deltaPerimeter) 
        self.boundaries[newb.index] = newb
        clusterPair[0].boundaries.append(newb)
        clusterPair[1].boundaries.append(newb)
    

class ClusteringOnInhomogeneity(SteepestDecentBoundaryPruningMethod):
    def __init__(self,N,vector,boundaries,clusters,preJ=None):
        self.N = N
        self.vector = vector
        self.boundaries = {}
        for bd in boundaries:
            self.boundaries[bd.index] = bd
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
        self.costs = cost_dict_class([(bd.index,(bd.deltaInhomogeneity,)) for bd in boundaries])
        """
        Cost is given by both inhomogeneities of clusters and length of boundary.
        """
        print "Looping on boundaries"
        for b in self.boundaries:
            self.costs[b] = (self.boundaries[b].deltaInhomogeneity+self.J,)
        SteepestDecentBoundaryPruningMethod.__init__(self,self.clusters,self.boundaries,self.costs,self.J)

    def InitializeBoundaries(self):
        numberOfbonds = sum([self.boundaries[b].length for b in self.boundaries])
        costOfbonds = sum([self.boundaries[b].deltaInhomogeneity for b in self.boundaries])
        return costOfbonds/numberOfbonds

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
        #if abs(b.deltaInhomogeneity + 3.30019381876e-07) < 1e-10:
        #    print "\tinhomo: ", b.deltaInhomogeneity, c0.inhomogeneity, c1.inhomogeneity #c1.inhomogeneity-b.deltaInhomogeneity
        #    print "\tsizes: ",len(c0.sites), len(c1.sites)#c1.inhomogeneity-b.deltaInhomogeneity
        #    print 'yup, bd_index =', b.index
        #    for site in c0.boundaries:
        #        print site.id, site.deltaInhomogeneity
        #    for site in c1.boundaries:
        #        print site.id, site.deltaInhomogeneity

        c0_site_num = len(c0.sites)
        c1_site_num = len(c1.sites)
        c1.sites += c0.sites
        c1.boundarySites += c0.boundarySites
        c1.box = c1.box.MergeWith(c0.box)
        c1.UpdateRadius()
        #if len(c1.sites) > CORESIZE*CORESIZE*16 and\
        #    len(c0.sites) > CORESIZE*CORESIZE*4 and\
        #    c1_site_num > CORESIZE*CORESIZE*4 or\
        #    c1.last_checked_size < c1_site_num+c0_site_num - CORESIZE*CORESIZE*8:
        #    c1.RemoveInternalSites()
        #    c1.last_checked_size = len(c1.sites)
        """
        Inhomogeneity of c1 is evaluated after absorbing c0.

        You don't have to recalculate it since you know this info
        from the boundary cost.
        """
        c1.inhomogeneity = c0.inhomogeneity+c1.inhomogeneity-b.deltaInhomogeneity
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
                b.deltaInhomogeneity =  -c1.GetBoundaryInhomogeneity(self.vector,cc0.boundarySites,cc1.boundarySites)  
                self.costs[b.index] = (b.deltaInhomogeneity+self.J,)
        return c0,c1

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

    def AddNewBoundary(self,clusterPair,removedBD):
        newb = Boundary(clusterPair,0.,removedBD.length, id=removedBD.id)
        cc0,cc1 = clusterPair
        newb.deltaInhomogeneity = -cc1.GetBoundaryInhomogeneity(self.vector,cc0.boundarySites,cc1.boundarySites)  
        self.costs[newb.index] = (newb.deltaInhomogeneity+self.J,)
        self.boundaries[newb.index] = newb
        clusterPair[0].boundaries.append(newb)
        clusterPair[1].boundaries.append(newb)
    

class ClusteringOnPerimeterToArea(SteepestDecentBoundaryPruningMethod):
    def __init__(self,N,boundaries,clusters,preJ=None):
        self.N = N
        self.clusters = clusters
        for c in self.clusters.values():
            c.perimeter = numpy.sum([bd.length for bd in c.boundaries]) 
            c.perimeterToArea = c.perimeter/len(c.sites)
        self.boundaries = boundaries
        for  bd in self.boundaries.values():
            c0,c1 = bd.clusterPair
            bd.deltaPerimeter = 2.*bd.length
            bd.areaToPerimeterOfCombinedCluster = (len(c0.sites)+len(c1.sites))/(c0.perimeter+c1.perimeter-2.*bd.length) 
        self.J = preJ
        self.costs = ValueSortedDict([(bd.index,(max([bd.clusterPair[0].perimeterToArea,\
                     bd.clusterPair[1].perimeterToArea])-self.J,bd.deltaPerimeter,bd.areaToPerimeterOfCombinedCluster)\
                     ) for bd in self.boundaries.values()])
        SteepestDecentBoundaryPruningMethod.__init__(self,self.clusters,self.boundaries,self.costs,self.J)

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
        c1.sites += c0.sites
        """
        The more efficient way to calculate P/A.
        """
        c1.perimeter += c0.perimeter-2.*b.length
        c1.perimeterToArea = c1.perimeter/len(c1.sites)
        """
        All existing boundaries of c1 are updated with recalculated costs. 
        """
        for b in c1.boundaries:
            cc0,cc1  = b.clusterPair
            b.areaToPerimeterOfCombinedCluster = (len(cc0.sites)+len(cc1.sites))/(cc0.perimeter+cc1.perimeter-2.*b.length)
            b.deltaPerimeter = 2.*b.length
            cc0,cc1 = b.clusterPair
            if (cc0 == c1):
                cc1,cc0 = b.clusterPair
            self.costs[b.index] = (max([cc0.perimeterToArea,cc1.perimeterToArea])-self.J,\
                                   b.deltaPerimeter,b.areaToPerimeterOfCombinedCluster)    
        return c0,c1

    def AddNewBoundary(self,clusterPair,removedBD):
        newb = Boundary(clusterPair,0.,removedBD.length, removedBD.id)
        cc0,cc1 = clusterPair
        newb.areaToPerimeterOfCombinedCluster = (len(cc0.sites)+len(cc1.sites))/(cc0.perimeter+cc1.perimeter-2.*newb.length)
        newb.deltaPerimeter = 2.*newb.length
        self.costs[newb.index] = (max([cc0.perimeterToArea,cc1.perimeterToArea])-self.J,\
                                  newb.deltaPerimeter,newb.areaToPerimeterOfCombinedCluster) 
        self.boundaries[newb.index] = newb
        clusterPair[0].boundaries.append(newb)
        clusterPair[1].boundaries.append(newb)
    

class PostClustering(SteepestDecentBoundaryPruningMethod):
    def __init__(self,N,vector,boundaries,clusters,preJ=None):
        self.N = N
        self.vector = vector
        self.clusters = clusters
        for c in self.clusters.values():
            c.perimeter = numpy.sum([bd.length for bd in c.boundaries]) 
            c.perimeterToArea = c.perimeter/float(len(c.sites))
        self.boundaries = boundaries
        for  bd in self.boundaries.values():
            bd.deltaPerimeter = 2.*bd.length
        self.J = preJ
        if N >= 512:
            self.costs = LargeValueSortedDict([(bd.index,(max([bd.clusterPair[0].perimeterToArea,\
                         bd.clusterPair[1].perimeterToArea])-self.J,bd.deltaPerimeter,bd.deltaInhomogeneity)\
                         ) for bd in self.boundaries.values()])
        else:
            self.costs = ValueSortedDict([(bd.index,(max([bd.clusterPair[0].perimeterToArea,\
                         bd.clusterPair[1].perimeterToArea])-self.J,bd.deltaPerimeter,bd.deltaInhomogeneity)\
                         ) for bd in self.boundaries.values()])
        SteepestDecentBoundaryPruningMethod.__init__(self,self.clusters,self.boundaries,self.costs,self.J)

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
        #print len(c0.sites), c0.inhomogeneity, c0.perimeter
        #print len(c1.sites), c1.inhomogeneity, c1.perimeter
        #print "\tinhomo: ", b.deltaInhomogeneity, c0.inhomogeneity, c1.inhomogeneity #c1.inhomogeneity-b.deltaInhomogeneity

        #if abs(b.deltaInhomogeneity + 3.30019381876e-07) < 1e-10:
        #    print "\tinhomo: ", b.deltaInhomogeneity, c0.inhomogeneity, c1.inhomogeneity #c1.inhomogeneity-b.deltaInhomogeneity
        #    print 'yup'

        #if c1.perimeter == 164:
        #    print "================================="
        #    print b.deltaInhomogeneity

        c0_site_num = len(c0.sites)
        c1_site_num = len(c1.sites)
        c1.sites += c0.sites
        c1.perimeter += c0.perimeter - 2.*b.length
        c1.perimeterToArea = c1.perimeter/float(len(c1.sites))
        c1.boundarySites += c0.boundarySites
        c1.box = c1.box.MergeWith(c0.box)
        c1.UpdateRadius()
        if len(c1.sites) > CORESIZE*CORESIZE*16 and\
            len(c0.sites) > CORESIZE*CORESIZE*4 and\
            c1_site_num > CORESIZE*CORESIZE*4 or\
            c1.last_checked_size < c1_site_num+c0_site_num - CORESIZE*CORESIZE*8:
            #print "ON"
            #c1.RemoveInternalSites()
            #c1.last_checked_size = len(c1.sites)
            pass
        """
        Inhomogeneity of c1 is evaluated after absorbing c0.

        You don't have to recalculate it since you know this info
        from the boundary cost.
        """
        c1.inhomogeneity = c0.inhomogeneity+c1.inhomogeneity-b.deltaInhomogeneity
        """
        All existing boundaries of c1 are updated with recalculated costs. 
        """

        dists = self.GetDists(c0, c1)
        for b in c1.boundaries:
            b.deltaPerimeter = 2.*b.length
            cc0,cc1 = b.clusterPair
            if (cc0 == c1):
                cc1,cc0 = b.clusterPair
            dist = dists.pop(0)
            if dist < c0.radius + cc0.radius + CORESIZE * 2:
                b.deltaInhomogeneity = -c1.GetBoundaryInhomogeneity(self.vector,cc0.boundarySites,cc1.boundarySites) 
                b.ptoa = max([cc0.perimeterToArea, cc1.perimeterToArea])
                self.costs[b.index] = (max([cc0.perimeterToArea,cc1.perimeterToArea])-self.J,b.deltaPerimeter,b.deltaInhomogeneity) 
                #self.costs[b.index] = (max([cc0.perimeterToArea,cc1.perimeterToArea])-self.J,b.deltaInhomogeneity,b.deltaPerimeter) 
        return c0,c1

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

    def AddNewBoundary(self,clusterPair,removedBD):
        newb = Boundary(clusterPair,0.,removedBD.length, removedBD.id)
        cc0,cc1 = clusterPair
        newb.deltaInhomogeneity = -cc1.GetBoundaryInhomogeneity(self.vector,cc0.boundarySites,cc1.boundarySites)  
        newb.deltaPerimeter = 2.*newb.length
        self.costs[newb.index] = (max([cc0.perimeterToArea,cc1.perimeterToArea])-self.J,newb.deltaPerimeter,newb.deltaInhomogeneity) 
        self.boundaries[newb.index] = newb
        clusterPair[0].boundaries.append(newb)
        clusterPair[1].boundaries.append(newb)
    

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
        horizontalBD = [Boundary((clusters[i*N+j],clusters[i*N+(j+1)%N]),\
                                 -self.CalculateEuclideanDistanceOfTwoVectors(self.vector[i,j],\
                                 self.vector[i,(j+1)%N]),1.,id=i+(j+1)%N*N) for i in range(N) for j in range(N)]
        verticalBD = [Boundary((clusters[i*N+j],clusters[((i+1)%N)*N+j]),\
                               -self.CalculateEuclideanDistanceOfTwoVectors(self.vector[i,j],\
                               self.vector[(i+1)%N,j]),1.,id=(i+1)%N+j*N) for i in range(N) for j in range(N)]
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
        horizontalBD = [Boundary((clusters[n*N**2+i*N+j],clusters[n*N**2+i*N+(j+1)%N]),\
                                 -self.CalculateEuclideanDistanceOfTwoVectors(self.vector[n,i,j],self.vector[n,i,(j+1)%N])\
                                 ,1.) for n in range(Nt) for i in range(N) for j in range(N)]
        verticalBD = [Boundary((clusters[n*N**2+i*N+j],clusters[n*N**2+((i+1)%N)*N+j]),\
                               -self.CalculateEuclideanDistanceOfTwoVectors(self.vector[n,i,j],self.vector[n,(i+1)%N,j])\
                               ,1.) for n in range(Nt) for i in range(N) for j in range(N)]
        temporalBD = [Boundary((clusters[n*N**2+i*N+j],clusters[(n+1)*N**2+i*N+j]),\
                               -temporalFactor*self.CalculateEuclideanDistanceOfTwoVectors(self.vector[n,i,j],self.vector[n+1,i,j])\
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

def Debug(J=0.1):
    import WallInitializer
    import OrientationField
    dir = "No Backup/DebugCluster/"
    N = 16 
    rodrigues = WallInitializer.RandomRodriguesVectorFieldGenerator((N,N),6,False,seed=1)
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
    preJ = J#0.5
    system = Initialization(N,rod)
    outputDir = dir+"DDDOldClustering_N_"+str(N)+"_preJ="+str(preJ).replace('.','_')
    boundaries,clusters = system.GenerateClustersFromEverySite() 
    clustering = ClusteringOnInhomogeneity(N,rod,boundaries,clusters,preJ)
    clustering.Run(outputDir)
    grain = clustering.GetGrainsizeDist()
    mis = clustering.GetMisorientationDist()
    bdlength = clustering.GetBoundaryLengthDist()
    print 'grain dist', grain
    print 'mis dist', mis
    print 'bdlength dist', bdlength
    clustering.GetReconstructionWithBoundary(N,orientationmap,outputDir)
    return rod,clustering

def Run(N,state,J,PtoA,dir,rod=None,orientationmap=None,snapshot=None,statefile=None,Return=False):
    from Plasticity import PlasticitySystem
    from Plasticity.FieldInitializers import FieldInitializer
    from Plasticity.Observers import OrientationField
    print "Starting it up..."
    if (orientationmap is None) or (rod is None):
        rodrigues = state.CalculateRotationRodrigues()
        rod = numpy.zeros((N,N,3),float)
        rod[:,:,0] = rodrigues['x']
        rod[:,:,2] = rodrigues['y']
        rod[:,:,1] = rodrigues['z']
        rotate = False
        if rotate :
            N = int(1.414*N)
            # Make it even
            N -= N%2
            rod = RotateVector_45degree(N,rod)
        orientationmap = OrientationField.RodriguesToUnambiguousColor(rod[:,:,0],rod[:,:,1],rod[:,:,2])
    if statefile is None:
        preJ = J
        system = Initialization(N,rod)
        outputDir = dir+"_Inhomo_Clustering_preJ="+str(preJ).replace('.','_')
        print "Generating clusters"
        boundaries,clusters = system.GenerateClustersFromEverySite() 
        print "Clustering on inhomogeneity"
        clustering = ClusteringOnInhomogeneity(N,rod,boundaries,clusters,preJ)
        clustering.Run(outputDir)
        clustering.GetReconstructionWithBoundary(N,orientationmap,outputDir)
        outputDir = dir+"_Post_Inhomo_Clustering_preJ="+str(preJ).replace('.','_')+\
                    "_PtoA="+str(PtoA).replace('.','_')
        boundaries,clusters = clustering.boundaries,clustering.clusters
    else:
        file = open(statefile)
        import pickle
        preJ = pickle.load(file)
        clusters = pickle.load(file)
        boundaries = pickle.load(file) 
        for c in clusters.values():
            c.boundaries = [boundaries[index] for index in c.boundaries]
        for bd in boundaries.values():
            bd.clusterPair = (clusters[bd.clusterPair[0]],clusters[bd.clusterPair[1]])
        outputDir = dir+"_Post_Inhomo_Clustering_preJ="+str(preJ).replace('.','_')+\
                    "_PtoA="+str(PtoA).replace('.','_')
    import BoundaryPruningUtility_Local
    BoundaryPruningUtility_Local.bd_index = max(boundaries.keys())+1
    BoundaryPruningUtility_Local.cl_index = max(clusters.keys())+1

    print "Post processing"
    postclustering = PostClustering(N,rod,boundaries,clusters,preJ=PtoA)
    postclustering.post = True
    postclustering.Run(outputDir)
    postclustering.GetReconstructionWithBoundary(N,orientationmap,outputDir)
    if snapshot is not None:
        for i in range(len(snapshot)):
            preJ = snapshot[i]
            outputDir = dir+"_Inhomo_Clustering_preJ="+str(preJ).replace('.','_')
            clustering = ClusteringOnInhomogeneity(N,rod,postclustering.boundaries.values(),postclustering.clusters.values(),preJ)
            clustering.Run(outputDir)
            clustering.GetReconstructionWithBoundary(N,orientationmap,outputDir)
            outputDir = dir+"_Post_Inhomo_Clustering_preJ="+str(preJ).replace('.','_')\
                        +"_PtoA="+str(PtoA).replace('.','_')
            postclustering = PostClustering(N,rod,clustering.boundaries,clustering.clusters,preJ=PtoA)
            postclustering.post = True
            postclustering.Run(outputDir)
            postclustering.GetReconstructionWithBoundary(N,orientationmap,outputDir)
    if Return:
        return postclustering 


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
    #Run()
    #Debug()
