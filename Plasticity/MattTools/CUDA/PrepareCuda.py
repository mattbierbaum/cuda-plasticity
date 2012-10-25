import PlasticitySystem
import PlasticityState
import FieldInitializer
import Fields
import numpy
from Constants import *

lengthscale=0.28
N = 1024 

prefix = ""
vacancy = None 
is3d = False 

seed = 0
for seed in [1]: #range(1): 
    if is3d:
        grid = (N,N,N)
        if vacancy is not None:
            grid2 = (10,N,N,N)
            prefix = "3d_vac_"
        else:
            grid2 = (9,N,N,N)
            prefix = "3d_"
    else:
        grid = (N,N)    
        if vacancy is not None:
            grid2 = (10,N,N)
            prefix = "vac_"
        else:
            grid2 = (9,N,N)

    #t,state = FieldInitializer.LoadState("TESTINGVACANCIES_4PT_VacancyWithStress_yy-xx_G1.0_A1.0_C0_STR0.05_L0_S0_2D1024.save",0)
    #t,state = FieldInitializer.LoadState("WALLINTERACTIONS_TwoTiltWalls_Parallel_VacancyWithStress_yy-xx_G10.0_A1000.0_C0_STR0.05_L0_S0_2D1024.save",0)
    state = FieldInitializer.GaussianRandomInitializer(grid,lengthscale,seed,vacancy=vacancy)
    novac = FieldInitializer.GaussianRandomInitializer(grid,lengthscale,seed)

    #state = FieldInitializer.SineWaveInitializer((128,128), randomPhase = True, seed=seed)
    #if N != 128:
    #    state = FieldInitializer.ResizeState(state,N)

    arr = []
    FieldInitializer.RotateState_90degree(state)
 
    if vacancy is not None:
        arr = numpy.zeros(grid2,dtype='float64')
        cnt = 0
        for c in novac.betaP.components:
            arr[cnt] = state.betaP_V[c].transpose()
            cnt += 1

        state.betaP_V['s','s'] *= 0.0
        arr[9] = state.betaP_V['s','s'].transpose()
    else:
        arr = numpy.zeros(grid2,dtype='float64')
        cnt = 0
        for c in novac.betaP.components:
            arr[cnt] = state.betaP[c].transpose()
            cnt += 1

    arr.tofile("%sinitial_%d_%d.mat" % (prefix,N,seed))
