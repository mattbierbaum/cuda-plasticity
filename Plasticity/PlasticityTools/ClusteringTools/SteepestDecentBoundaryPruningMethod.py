import numpy

class SteepestDecentBoundaryPruningMethod:
    def __init__(self,clusters,boundaries,cost,J):
        self.clusters = clusters
        self.boundaries = boundaries
        self.cost = cost
        self.J = J
        self.post = False

    def Run(self,filename):
        """
        On every step, the most expensive boundary is found out and
        removed until the associated cost gets negative.
        """
        print "Start to run!"
        step = 0
        log = open("./logging.txt", 'w')
        while True:
            index = self.costs.argmax()
            maxcost = self.costs[index][0]

            #if len(self.costs[index]) == 3:
            #    print "Removing ", self.costs[index][0], self.costs[index][1], self.costs[index][2]

            #print "step", step, "ptoa", maxcost
            log.write("%i %f\n" % (step, maxcost))
            if (maxcost<=0.):
                break
            self.RemoveBoundary(index)
            step += 1
        self.SaveState(filename)
        self.SaveResults(filename)
        print self.J, " The total number of clusters is ",len(self.clusters)

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

        if self.post:
            print "B" # --", b.index, b.deltaInhomogeneity
            print "%e" % b.deltaInhomogeneity
            for site in c0.boundaries:
                #print "\t0 %i %e" % (site.id, site.deltaInhomogeneity)
                print "\t %i %e %e" % (site.deltaPerimeter, max([c.perimeterToArea for c in site.clusterPair]), site.deltaInhomogeneity)
            for site in c1.boundaries:
                #print "\t1 %i %e" % (site.id, site.deltaInhomogeneity)
                print "\t %i %e %e" % (site.deltaPerimeter, max([c.perimeterToArea for c in site.clusterPair]), site.deltaInhomogeneity)

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

        if self.post:
            print "A" # --", b.index, b.deltaInhomogeneity
            print "%e" % b.deltaInhomogeneity
            #for site in c0.boundaries:
            #    #print "\t0 %i %e" % (site.id, site.deltaInhomogeneity)
            #    print "\t0 %i %e %e" % (site.deltaPerimeter, max([c.perimeterToArea for c in site.clusterPair]), site.deltaInhomogeneity)
            for site in c1.boundaries:
                #print "\t1 %i %e" % (site.id, site.deltaInhomogeneity)
                print "\t %i %e %e" % (site.deltaPerimeter, max([c.perimeterToArea for c in site.clusterPair]), site.deltaInhomogeneity)

    def UpdateClusters(self,c0,c1,b):
        """
        Remove one of the cluster, and transfer all its properties, except
        boundaries, to the other cluster.
        """
        pass

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
                self.AddNewBoundary(pair,bd)
            else:
                """
                Otherwise, the boundary will be added to the existed boundary of c1.
                """
                if pair in pairsOfc1:
                    nb = c1.boundaries[pairsOfc1[pair]]
                if (pair[1],pair[0]) in pairsOfc1:
                    nb = c1.boundaries[pairsOfc1[(pair[1],pair[0])]]
                self.CombineBoundaries(nb,bd)

    def AddNewBoundary(self,clusterPair,removedBoundary):
        """
        We need a new object of boundary class and fill its properties
        with all from the removed boundary. And the cost of this new
        boundary needs to be recalculated. 
        The new boundary need to be added to the system.
        """
        pass
    
    def CombineBoundaries(self,existingBoundary,removedBoundary):
        """
        Transfer the properties of the removed boudnary to the
        existed boundary. 
        Noted that the inhomogeneity part of the cost doesn't
        need to be changed. 
        Usually, the only thing needed is the boundary length.
        """
        #print "merging", existingBoundary.length, removedBoundary.length, existingBoundary.deltaInhomogeneity, removedBoundary.deltaInhomogeneity
        existingBoundary.length += removedBoundary.length
        #print "merged length", existingBoundary.length

    def SaveState(self,filename,mode='w'):
        import sys
        import pickle
        if filename is not None:
            file = open(filename+'_state.save',mode)
        else:
            file = sys.stdout
        pickle.dump(self.J,file,protocol=2) 
        for c in self.clusters.values():
            c.boundaries = [bd.index for bd in c.boundaries] 
        pickle.dump(self.clusters,file,protocol=2) 
        for c in self.clusters.values():
            c.boundaries = [self.boundaries[index] for index in c.boundaries] 
        for bd in self.boundaries.values():
            c0,c1 = bd.clusterPair
            bd.clusterPair = (c0.index,c1.index) 
        pickle.dump(self.boundaries,file,protocol=2) 
        for bd in self.boundaries.values():
            c0index,c1index = bd.clusterPair
            bd.clusterPair = (self.clusters[c0index],self.clusters[c1index])
        file.flush()

    def SaveResults(self,filename,mode='w'):
        import sys
        import pickle
        if filename is not None:
            file = open(filename+'.save',mode)
        else:
            file = sys.stdout
        pickle.dump(self.GetGrainsizeDist(),file,protocol=2) 
        pickle.dump(self.GetMisorientationDist(),file,protocol=2)
        pickle.dump(self.GetBoundaryLengthDist(),file,protocol=2)
        pickle.dump(self.GetIndexmap(),file,protocol=2)
        file.flush()

    def GetGrainsizeDist(self):
        """
        Grain size is equal to the square root of the area of each cluster.
        """
        return numpy.array([c.GetSize()  for c in self.clusters.values()])

    def GetMisorientationDist(self):
        """
        Misorientation angle is equal to the square root of the inhomogeneity of each cluster.
        """
        return numpy.array([numpy.sqrt(numpy.abs(b.deltaInhomogeneity)) for b in self.boundaries.values()])

    def GetBoundaryLengthDist(self):
        """
        Boundary length is measured by the total number of site-pairs along the boundary.
        """
        return numpy.array([b.length for b in self.boundaries.values()])

    def GetIndexmap(self):
        """
        Generate an numpy array of index map.
        """
        indexmap = numpy.zeros((self.N,self.N),int)
        i = 0
        for c in self.clusters.values():
            for pos in c.sites:
                indexmap[tuple(pos)] = i
            i+=1
        return indexmap

    def GetReconstructionWithBoundary(self,N,orientationmap,filename):
        """
        Decorate the orientation map with boundaries.
        """
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
        from PIL import Image
        pilImage = Image.fromarray(neworientationmap.astype('uint8'), 'RGB')
        pilImage.save(filename+'.png')


