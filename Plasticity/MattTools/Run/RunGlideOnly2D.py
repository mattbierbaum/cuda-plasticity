from Plasticity import PlasticitySystem
from Plasticity.FieldInitializers import FieldInitializer
from Plasticity.FieldDynamics import FieldDynamics
from Plasticity.FieldMovers import FieldMover
from Plasticity.Observers import Observer
from Plasticity.Constants import *
#from CUDAGridArray import GridArray
#import GridArray
#import NumericalMethods
from Plasticity.FieldDynamics import CentralUpwindHJBetaPGlideOnlyDynamics

import sys
import os
import getopt

def Relaxation(seed):
    N = 32
    gridShape = (N,N)
    
    dynamics = CentralUpwindHJBetaPGlideOnlyDynamics.BetaPDynamics()

    lengthscale = 0.2 #0.2*(2.**0.5)
    #filename = "NewGlideOnly_ls0_28_"+"S_"+str(seed)+"_2D"+str(N)+".save"
    filename = "cudacompare.save"

    state = FieldInitializer.GaussianRandomInitializer(gridShape,lengthscale,seed)
    mover = FieldMover.TVDRungeKutta_FieldMover(CFLsafeFactor=0.1)

    obsState = Observer.RecallStateObserver()

    startTime = 0. 
    endTime   = 20. 
    dt = 1.

    t = startTime 
    if startTime == 0. :
        recordState = Observer.RecordStateObserver(filename=filename)
        recordState.Update(t, state)
    else:
        T,state = FieldInitializer.LoadState(filename)
        recordState = Observer.RecordStateObserver(filename=filename,mode='a')

    system= PlasticitySystem.PlasticitySystem(gridShape, state, mover, dynamics, [obsState,Observer.VerboseTimestepObserver()])

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
	print t

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
