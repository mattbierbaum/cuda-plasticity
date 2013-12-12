from Plasticity import PlasticitySystem
from Plasticity.FieldInitializers import FieldInitializer
from Plasticity.FieldDynamics import SmecticDynamics
from Plasticity.FieldMovers import FieldMover
from Plasticity.PlasticityStates import SmecticState
from Plasticity.Constants import *
from Plasticity.GridArray import GridArray
from Plasticity import NumericalMethods
from Plasticity.Fields import Fields
from Plasticity.Observers import Observer
import pylab
import numpy
import scipy.weave as weave

def go():
    #gridShape = (32,32,32)
    gridShape = (64,64,64)
    #gridShape = (128,128,128)
    sigma = 0.08
    file_output = "testing_L0_08.state"

    state = FieldInitializer.GaussianRandomInitializer(gridShape, sigma=sigma, smectic=True)
    state.enforceBCs()
    state.removeCurl()

    dynamics = SmecticDynamics.SmecticDynamics(K=0.01)
    mover = FieldMover.TVDRungeKutta_FieldMover(CFLsafeFactor=0.1)

    obsState = Observer.RecallStateObserver()

    startTime = 0. 
    endTime   = 30.
    dt = 0.025
    t = 0.

    recordState = Observer.RecordStateObserver(filename=file_output)
    recordState.Update(t, state)

    system=PlasticitySystem.PlasticitySystem(gridShape, state, mover, dynamics, [obsState,Observer.VerboseTimestepObserver()])

    while t<=(endTime):
        preT = t
        if t<=0.1-0.01:
            dt = 0.01
        elif t<=1.:
            dt = 0.05
        elif t<=5.:
            dt = 0.5
        else:
            dt = 2.5
        t += dt
        system.Run(startTime=preT, endTime = t)
        system.state = obsState.state
        recordState.Update(t, system.state)

def main():
    go()

if __name__ == "__main__":
    main()



