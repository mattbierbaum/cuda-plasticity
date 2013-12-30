from collections import defaultdict
import numpy as np

def SegmentationConvert2D(rodx, rody, rodz, segs):
    
    def boundarylength(rodx, rody, rodz, segs):
        keys = set(segs.flatten())
        grainsize = {}
        bdlength = defaultdict(dict)
        misorientation = defaultdict(dict)

        for k in keys:
            grainsize[k] = (segs == k).sum()
            bdlength[k] = {}
            misorientation[k] = {}

        for i in xrange(segs.shape[0]):
            if i % 128 == 0:
                print i

            for j in xrange(segs.shape[1]):
                for ti in [-1, 1]:
                    for tj in [-1, 1]:
                        x = (i + ti) % segs.shape[0]
                        y = (j + tj) % segs.shape[1]
                        if segs[i,j] != segs[x,y]:
                            bdlength[segs[i,j]].setdefault(segs[x,y], 0)
                            misorientation[segs[i,j]].setdefault(segs[x,y], 0)

                            misorientation[segs[i,j]][segs[x,y]] += (rodx[i,j] - rodx[x,y])**2
                            misorientation[segs[i,j]][segs[x,y]] += (rody[i,j] - rody[x,y])**2
                            misorientation[segs[i,j]][segs[x,y]] += (rodz[i,j] - rodz[x,y])**2
                            bdlength[segs[i,j]][segs[x,y]] += 1

        suml = []
        summ = []

        for s in keys:
            for k,v in bdlength[s].iteritems():
                suml.append(v)
                summ.append(np.sqrt(misorientation[s][k]))

        suml = np.array(suml)
        summ = np.array(summ)
        return summ/suml, np.array(grainsize.values()), suml

    return boundarylength(rodx, rody, rodz, segs)


def SegmentationConvert3D(rodx, rody, rodz, segs):
    
    def boundarylength(rodx, rody, rodz, segs):
        keys = set(segs.flatten())
        grainsize = {}
        bdlength = defaultdict(dict)
        misorientation = defaultdict(dict)

        for k in keys:
            grainsize[k] = (segs == k).sum()
            bdlength[k] = {}
            misorientation[k] = {}

        for i in xrange(segs.shape[0]):
            print i

            for j in xrange(segs.shape[1]):
                for k in xrange(segs.shape[2]):
                    for ti in [-1, 1]:
                        for tj in [-1, 1]:
                            for tk in [-1, 1]:
                                x = (i + ti) % segs.shape[0]
                                y = (j + tj) % segs.shape[1]
                                z = (k + tk) % segs.shape[2]
                                if segs[i,j,k] != segs[x,y,z]:
                                    bdlength[segs[i,j,k]].setdefault(segs[x,y,z], 0)
                                    misorientation[segs[i,j,k]].setdefault(segs[x,y,z], 0)

                                    misorientation[segs[i,j,k]][segs[x,y,z]] += (rodx[i,j,k] - rodx[x,y,z])**2
                                    misorientation[segs[i,j,k]][segs[x,y,z]] += (rody[i,j,k] - rody[x,y,z])**2
                                    misorientation[segs[i,j,k]][segs[x,y,z]] += (rodz[i,j,k] - rodz[x,y,z])**2
                                    bdlength[segs[i,j,k]][segs[x,y,z]] += 1

        suml = []
        summ = []

        for s in keys:
            for k,v in bdlength[s].iteritems():
                suml.append(v)
                summ.append(np.sqrt(misorientation[s][k]))

        suml = np.array(suml)
        summ = np.array(summ)
        return summ/suml, np.array(grainsize.values()), suml

    return boundarylength(rodx, rody, rodz, segs)
