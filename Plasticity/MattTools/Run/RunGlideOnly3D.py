import PlasticitySystem
import FieldInitializer
import FieldDynamics
import FieldMover
import Observer
from Constants import *
from CUDAGridArray import GridArray
import NumericalMethods
import CentralUpwindHJBetaPGlideOnlyDynamics

import sys
import os
import getopt

import gc
gc.collect()
#gc.set_debug(gc.DEBUG_LEAK)

class MemoryObserver(Observer.Observer):
    def __init__(self):
        pass

    def Update(self, time, state):
        GridArray.print_mem_usage()


def Relaxation(seed):
    N = 128
    gridShape = (N,N,N)
    
    dynamics = CentralUpwindHJBetaPGlideOnlyDynamics.BetaPDynamics()

    lengthscale = 0.2*(2.**0.5)
    filename = "NewGlideOnly_ls0_28_"+"S_"+str(seed)+"_3D"+str(N)+".save"
    
    t0, state = FieldInitializer.LoadState(filename) #FieldInitializer.GaussianRandomInitializer(gridShape,lengthscale,seed)
    print state.gridShape
    mover = FieldMover.TVDRungeKutta_FieldMover(CFLsafeFactor=0.1)

    obsState = Observer.RecallStateObserver()
    memObs = MemoryObserver()

    startTime = t0. 
    endTime   = 30. 
    dt = 1.

    t = startTime 
    if startTime == 0. :
        recordState = Observer.RecordStateObserver(filename=filename)
        recordState.Update(t, state)
    else:
        T,state = FieldInitializer.LoadState(filename)
        recordState = Observer.RecordStateObserver(filename=filename,mode='a')

    system= PlasticitySystem.PlasticitySystem(gridShape, state, mover, dynamics, [obsState, memObs])

    while t<=(endTime):
        preT = t
        #"""
        if t<=0.1-0.01:
            dt = 0.01
        elif t<=1.:
            dt = 0.05
        elif t<=5.:
            dt = 0.5
        else:
            dt = 2.5
        #"""
        t += dt
        system.Run(startTime=preT, endTime = t)
        system.state = obsState.state
        recordState.Update(t, system.state)

def main(argv):
    try: 
	    opts, args = getopt.getopt(argv, "s:")
    except:
	    sys.exit(2)
    for opt, arg in opts:
	    if opt == '-s':
	        seed = int(arg)
    Relaxation(seed=seed)

if __name__ == "__main__":
    main(sys.argv[1:])
