import boundarypruningutil as bp
import boxcounting as bc

run_dir   = "./"
run_types = ["mdp", "lvp"   ]
run_dim   = [ 2,    3       ]   
run_size  = [1024,  128     ]
run_post  = ["d", "r", "r0", "r0a", "r0b", "r0c", "r0bx", "r0bz"]
run_seed  = [1, 2, 3, 4, 5, 6]

run_times = {"d": [50.0], 
             "r": [50.0],
             "r0": arange(0,4.0/0.05,1.0),
             "r0a": arange(0,4.0/0.05,1.0),
             "r0b": arange(0,4.0/0.05,1.0),
             "r0c": arange(0,4.0/0.05,1.0),
             "r0bx": arange(0,4.0/0.05,1.0),
             "r0bz": arange(0,4.0/0.05,1.0),
            } 

for type in run_types:
    for dim, size in zip(run_dim, run_size):
        for post in run_post:
            #for time in run_times[post]:
            # average over the seeds
            for seed in run_seed:
                file = run_dir+type+str(dim)+"d"+str(size)+"_s"+str(seed)+"_"+post+".plas"

                if not os.path.isfile(file):
                    print "Could not find (skipping) ", file
                    continue 
                else:
                    print file, seed, time
