from optparse import OptionParser
import Plasticity.RunCUDA as run
import copy, threading, time

dct = {"cudadir": "/b/bierbaum/cuda-plasticity/", 
       "homedir": "/b/bierbaum/plasticity/", 
       "N":       1024,
       "dim":      2, 
       "previous": "", 
       "postfix":  "r", 
       "method":   "lvp"}

#dct = {"cudadir": "/b/bierbaum/cuda-plasticity/", 
#       "homedir": "/b/bierbaum/plasticity/", 
#       "N":       128,
#       "dim":      3, 
#       "previous": "r", 
#       "postfix":  "0", 
#       "method":   "lvp",
#       "load_direction": [[-0.5, 0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,-0.5]],
#       "load_rate": 0.05,
#       "time_start": 50., 
#       "time_end":  500,
#       "time_step": 5.0}

def launch(loc, i):
    loc.update({"device": i})
    loc.update({"seed":   i})
   
    loc["method"] = "lvp"
    loc["seed"] = i 
    run.simulation(loc)
    loc["seed"] = i+4
    run.simulation(loc)

    loc["method"] = "mdp"
    loc["seed"] = i
    run.simulation(loc)
    loc["seed"] = i+4
    run.simulation(loc)

    loc["method"] = "gcd"
    loc["seed"] = i
    run.simulation(loc)
    loc["seed"] = i+4
    run.simulation(loc)

    #loc["postfix"] = 1
    #loc["method"] = "lvp"
    #loc["seed"] = i 
    #run.simulation(loc)
    #loc["seed"] = i+4
    #run.simulation(loc)

    #loc["method"] = "mdp"
    #loc["seed"] = i
    #run.simulation(loc)
    #loc["seed"] = i+4
    #run.simulation(loc)

    #loc["method"] = "gcd"
    #loc["seed"] = i
    #run.simulation(loc)
    #loc["seed"] = i+4
    #run.simulation(loc)
    
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


