from optparse import OptionParser
import Plasticity.RunCUDA as run
import copy, threading, time

#dct = {"cudadir": "/b/bierbaum/cuda-plasticity/", 
#       "homedir": "/a/bierbaum/plasticity/", 
#       "N":       1024,
#       "dim":      2, 
#       "previous": "", 
#       "postfix":  "d", 
#       "method":   "lvp"}

dct = {"cudadir": "/b/bierbaum/cuda-plasticity/", 
       "homedir": "/b/bierbaum/plasticity/", 
       "N":       1024,
       "dim":      2, 
       "previous": "r", 
       "postfix":  "0", 
       "method":   "lvp",
       "load_direction": [[-0.5, 0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,-0.5]],
       "load_rate": 0.05,
       "time_start": 50., 
       "time_end":  100,
       "time_step": 5.0}

def launch(loc, i):
    loc.update({"device": i})
    loc.update({"seed":   i})
   
    sizes = ((128, 3), (1024, 2))
    meths = ("mdp", "gcd", "lvp")
    seeds = (i,)

    import itertools
    grps = itertools.product(sizes, meths, seeds)
    for g in grps:
        N, dim = g[0]
        method = g[1]
        seed   = g[2]

        loc["N"] = N
        loc["dim"] = dim
        loc["method"] = method
        loc["seed"] = seed
        run.simulation(loc)

    
def main():
    threads = []
    for i in range(4):
        loc = copy.copy(dct)
        thread = threading.Thread(target=launch, args=(loc,i))
        thread.start()
        threads.append(thread)
        time.sleep(60)

    for i in range(4):
        threads[i].join()
 
if __name__ == "__main__":
    main()


