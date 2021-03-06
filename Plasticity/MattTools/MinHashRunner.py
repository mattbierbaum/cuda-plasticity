import boundarypruningutil as bp
from Plasticity import TarFile
from os.path import basename, splitext, isfile
import numpy as np
import pylab as pl
import powerlaw
from subprocess import check_call
from Plasticity.FieldInitializers import FieldInitializer

import cPickle as pickle
import MinHash
import SegmentationConversion

def prune_set(time=None, do2d=True, do3d=False):
    run_dir   = "/a/bierbaum/plasticity/keeneland"
    run_dir   = "/media/scratch/plasticity"
    run_types = ["mdp", "lvp", "gcd" ]
    run_dim   = [     3       ]
    run_size  = [  128     ]
    run_seed  = [0,1, 2, 3, 4, 5]
    if time is None:
        run_post  = ["d"]
    else:
        run_post = ["d0"]

    all_mis = []
    all_grain = []
    all_bdlength = []
    all_mis3 = []
    all_grain3 = []
    all_bdlength3 = []

    if time is not None:
        outfolder = run_dir+"/minhash_bp_t_%0.3f/" % time
    else:
        outfolder = run_dir+"/minhash_bp_relax/"

    try:
        check_call(["mkdir", outfolder])
    except Exception as e:
        print "Folder", outfolder, "already exists, probably"

    for type in run_types:
        print "Type:", type
        type_mis = np.array([0.0])
        type_grain = np.array([0.0])
        type_bdlength = np.array([0.0])

        type_mis3 = np.array([0.0])
        type_grain3 = np.array([0.0])
        type_bdlength3 = np.array([0.0])
        for dim, size in zip(run_dim, run_size):
            for post in run_post:
                for seed in run_seed:
                    stub = type+str(dim)+"d"+str(size)
                    folder = run_dir#+"/"+stub
                    file = folder+"/"+stub+"_s"+str(seed)+"_"+post+".tar"

                    #file2 = folder+"/bpoutput/"+stub+"_s"+str(seed)+"_"+post+".tar"

                    ftype = 'tar'
                    if not isfile(file):
                        file = folder+"/"+stub+"_s"+str(seed)+"_"+post+".plas"
                        ftype = 'plas'
                        if not isfile(file):
                            print "Could not find (skipping) ", file
                            continue

                    print "Processing", file
                    if ftype == 'tar':
                        t,s = TarFile.LoadTarState(file, time=time)
                    else:
                        t,s = FieldInitializer.LoadStateRaw(file, size, dim, time=time, hastimes=True)
                    rod = s.CalculateRotationRodrigues()

                    if do2d:
                        print "2D slices:"
                        for i in range(128):
                            prefix = outfolder+splitext(basename(file))[0]+"_2d%03d_" % i
                            print "\t", i
                            trod ={}
                            trod['x'] = rod['x'][:,:,i]
                            trod['y'] = rod['y'][:,:,i]
                            trod['z'] = rod['z'][:,:,i]
                            #mis, grain, bdlength, throwaway = bp.Run(size, 2, trod, outfolder+splitext(basename(file))[0]+"_2d%03d_"%i, J=8e-7, PtoA=1.5, Image=False, Dump=True)
                            segs = MinHash.LSH_run_2d(trod)
                            mis, grain, bdlength = SegmentationConversion.SegmentationConvert2D(trod['x'], trod['y'], trod['z'], segs)

                            pickle.dump(mis, open("%s.mis.pickle" % prefix, "w"), protocol=-1) 
                            pickle.dump(grain, open("%s.grain.pickle" % prefix, "w"), protocol=-1) 
                            pickle.dump(bdlength, open("%s.bdlength.pickle" % prefix, "w"), protocol=-1) 
                            pickle.dump(segs, open("%s.indexmap.pickle" % prefix, "w"), protocol=-1) 

                            type_mis = np.hstack((type_mis, mis))
                            type_grain = np.hstack((type_grain, grain))
                            type_bdlength = np.hstack((type_bdlength, bdlength))

                    if do3d:
                        print "3D volume:"
                        prefix = outfolder+splitext(basename(file))[0]+"_3d_"
                        segs = MinHash.LSH_run_3d(rod)
                        mis3, grain3, bdlength3 = SegmentationConversion.SegmentationConvert3D(rod['x'], rod['y'], rod['z'], segs)

                        pickle.dump(mis3, open("%s.mis.pickle" % prefix, "w"), protocol=-1) 
                        pickle.dump(grain3, open("%s.grain.pickle" % prefix, "w"), protocol=-1) 
                        pickle.dump(bdlength3, open("%s.bdlength.pickle" % prefix, "w"), protocol=-1) 
                        pickle.dump(segs, open("%s.indexmap.pickle" % prefix, "w"), protocol=-1) 

                        type_mis3 = np.hstack((type_mis3, mis3))
                        type_grain3 = np.hstack((type_grain3, grain3))
                        type_bdlength3 = np.hstack((type_bdlength3, bdlength3))

        if do2d:
            all_mis.append(type_mis)
            all_grain.append(type_grain)
            all_bdlength.append(type_bdlength)

        if do3d:
            all_mis3.append(type_mis3)
            all_grain3.append(type_grain3)
            all_bdlength3.append(type_bdlength3)

    if do2d:
        pickle.dump({"mis": all_mis, "grain": all_grain, "length": all_bdlength}, open(outfolder+"prune_all.pickle", "w"))

        for i in range(len(run_types)):
            pl.figure();pl.hist(all_grain[i], bins=np.logspace(0,4,100), histtype='step')

        for i in range(len(run_types)):
            pl.figure();pl.hist(all_mis[i]*all_bdlength[i], bins=np.linspace(1e-3,3e-2,100), histtype='step')

        for i in range(len(run_types)):
            y,x = np.histogram(all_grain[i], bins=np.logspace(0,4,100))
            f = powerlaw.Fit(y)
            print f.alpha

    if do3d:
        pickle.dump({"mis": all_mis3, "grain": all_grain3, "length": all_bdlength3}, open(outfolder+"prune_all_3d.pickle", "w"))

        for i in range(len(run_types)):
            pl.figure();pl.hist(all_grain3[i], bins=np.logspace(0,4,100), histtype='step')

        for i in range(len(run_types)):
            pl.figure();pl.hist(all_mis3[i]*all_bdlength3[i], bins=np.linspace(1e-3,3e-2,100), histtype='step')

        for i in range(len(run_types)):
            y,x = np.histogram(all_grain3[i], bins=np.logspace(0,4,100))
            f = powerlaw.Fit(y)
            print f.alpha

    #return all_mis, all_grain, all_bdlength #, all_mis3, all_grain3, all_bdlength3


def prune_timeseries():
    #run_dir   = "/a/bierbaum/plasticity/keeneland"
    run_dir   = "/media/scratch/plasticity"
    type = "lvp"
    dim   = 3
    size  = 128
    seed  = 0
    post = "d0"

    stub = type+str(dim)+"d"+str(size)
    folder = run_dir#+"/"+stub
    file = folder+"/"+stub+"_s"+str(seed)+"_"+post+".tar"

    outfolder = "./bp_timeseries_"+stub+"/"

    try:
        check_call(["mkdir", outfolder])
    except Exception as e:
        print "Folder", outfolder, "already exists, probably"

    ftype = 'tar'
    if not isfile(file):
        file = folder+"/"+stub+"_s"+str(seed)+"_"+post+".plas"
        ftype = 'plas'
        if not isfile(file):
            print "Could not find (skipping) ", file

    for time in np.arange(0, 400, 1):
        print "Processing", file
        if ftype == 'tar':
            t,s = TarFile.LoadTarState(file, time=time)
        else:
            t,s = FieldInitializer.LoadStateRaw(file, size, dim, time=time, hastimes=True)
        rod = s.CalculateRotationRodrigues()

        print "3D volume:"
        bp.Run(size, 3, rod, outfolder+splitext(basename(file))[0]+"_3d_t=%03d_" % int(time), J=3e-7, PtoA=1.2, Image=False, Dump=True)


def slice_one_3d(file):
    if not isfile(file):
        print "Could not find (skipping) ", file
    else:
        print "Processing", file
        t,s = TarFile.LoadTarState(file)
        cof = TarFile.LoadTarJSON(file)
        rod = s.CalculateRotationRodrigues()
        for i in range(128):
            trod ={}
            trod['x'] = rod['x'][:,:,i]
            trod['y'] = rod['y'][:,:,i]
            trod['z'] = rod['z'][:,:,i]
            mis, grain, bdlength = bp.Run(cof['N'], 2, trod, splitext(basename(file))[0]+"%03d_"%i, J=7e-7, PtoA=1.5, Image=True, Dump=False)

#for t in np.arange(0,400,10):
#    prune_set(t)
prune_set(0, do2d=False, do3d=True)
#prune_timeseries()
