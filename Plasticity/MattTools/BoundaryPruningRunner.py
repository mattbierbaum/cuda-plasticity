import boundarypruningutil as bp
from Plasticity import TarFile
from os.path import basename, splitext, isfile
import numpy as np
import pylab as pl
import powerlaw

def prune_set():
    run_dir   = "/b/bierbaum/plasticity"
    run_types = ["mdp", "lvp", "gcd" ]
    run_dim   = [     3       ]   
    run_size  = [  128     ]
    run_post  = ["r"]
    run_seed  = [0,1, 2, 3, 4, 5, 6, 7]
    
    all_mis = []
    all_grain = []
    all_bdlength = []
    for type in run_types:
        type_mis = np.array([0.0])
        type_grain = np.array([0.0])
        type_bdlength = np.array([0.0])
        for dim, size in zip(run_dim, run_size):
            for post in run_post:
                for seed in run_seed:
                    stub = type+str(dim)+"d"+str(size) 
                    folder = run_dir+"/"+stub
                    file = folder+"/"+stub+"_s"+str(seed)+"_"+post+".tar"
    
                    if not isfile(file):
                        print "Could not find (skipping) ", file
                        continue 
                    else:
                        print "Processing", file
                        t,s = TarFile.LoadTarState(file)
                        rod = s.CalculateRotationRodrigues()
                        #for i in range(128):
                        #    trod ={}  
                        #    trod['x'] = rod['x'][:,:,i]
                        #    trod['y'] = rod['y'][:,:,i]
                        #    trod['z'] = rod['z'][:,:,i]
                        #    mis, grain, bdlength = bp.Run(size, 2, rod, splitext(basename(file))[0], J=7e-7, PtoA=1.5, Image=False, Dump=False)
                        mis, grain, bdlength = bp.Run(size, 3, rod, splitext(basename(file))[0], J=3e-7, PtoA=1.12, Image=False, Dump=False)
                        type_mis = np.hstack((type_mis, mis))
                        type_grain = np.hstack((type_grain, grain))
                        type_bdlength = np.hstack((type_bdlength, bdlength))
        all_mis.append(type_mis)
        all_grain.append(type_grain)
        all_bdlength.append(type_bdlength)         

    import pickle
    pickle.dump({"mis": all_mis, "grain": all_grain, "length": all_bdlength}, open("prune_all.pickle", "w"))
    
    for i in range(3): 
        pl.hist(all_grain[i], bins=np.logspace(0,4,100), histtype='step')
    
    for i in range(3): 
        pl.hist(all_mis[i]*all_bdlength[i], bins=np.linspace(1e-3,3e-2,100), histtype='step')

    for i in range(3):
        y,x = np.histogram(all_grain[i], bins=np.logspace(0,4,100))
        f = powerlaw.Fit(y)
        print f.alpha

    return all_mis, all_grain, all_bdlength

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
 
