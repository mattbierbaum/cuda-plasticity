import PlasticitySystem
import FieldInitializer
import FieldDynamics
import FieldMover
import CentralUpwindHJBetaPDynamics
import CentralUpwindHJBetaPGlideOnlyDynamics
import Observer

import sys
import os
import getopt
import numpy
from numpy import fft
    
from Constants import *

mu,nu = 0.5,0.3 
lamb = 2.*mu*nu/(1.-2.*nu)
     
def ExternalStrain(sigma,primaryStrain):
    strains = {x:primaryStrain[0],y:primaryStrain[1],z:primaryStrain[2]}
    strain_trace = strains[x]+strains[y]+strains[z]
    for i in [x,y,z]:
        sigma[i,i] += lamb*strain_trace + 2.*mu*strains[i] 
    return sigma

def ExternalStress(sigma,primaryStress):
    stresses = {x:primaryStress[0],y:primaryStress[1],z:primaryStress[2]}
    for i in [x,y,z]:
        sigma[i,i] += stresses[i] 
    return sigma


#class StrainLoadingBetaPDynamics(CentralUpwindHJBetaPDynamics.BetaPDynamics):
class StrainLoadingBetaPDynamics(CentralUpwindHJBetaPGlideOnlyDynamics.BetaPDynamics):
    def __init__(self, Dx=None, Lambda=0, coreEnergy=0, coreEnergyLog=0, rate=None, type='strain', \
                       initial=numpy.array([0.,0.,0.]),initT=0.):
        CentralUpwindHJBetaPGlideOnlyDynamics.BetaPDynamics.__init__(self,Dx,coreEnergy,coreEnergyLog)
        self.rate = rate
        self.type = type
        self.initial = initial
        self.initT = initT

    def GetSigma(self,state,time):
        sigma = state.CalculateSigma()
        if self.type == 'strain':
            return ExternalStrain(sigma,self.initial+self.rate*(time-self.initT)) 
        else:
            return ExternalStress(sigma,self.initial+self.rate*(time-self.initT)) 


def ExternalLoading2D_CU(seed):
    N = 128 
    gridShape = (N,N)

    Lambda = 0
    if Lambda == 0:
        motion = 'GlideClimb'
    else:
        motion = 'GlideOnly'

    strainrate = 0.0
    direction = numpy.array([1.0,-0.5,-0.5]) #numpy.array([-0.5,-0.5,1.0])
    Rate = strainrate*direction 
    Type = 'stress'
    Initial = 0.01*direction

    lengthscale = 0.2 
    relaxfile = "RND_Upwind_L0_S0_2D128.save"
    #relaxfile = "NewGlideOnly_ls0_28_S_"+str(seed)+"_2D"+str(N)+".save"
    #"FixV_CU_S_"+str(seed)+"_2D"+str(N)+"_lengthscale_"+str(lengthscale).replace(".","_")+"_"+motion+".save"
    #os.system("msscmd cd result, get "+relaxfile)
    timef,state = FieldInitializer.LoadState(relaxfile)

    #loadfile = "UNI_zz_S_"+str(seed)+"_rate_"+str(strainrate).replace('.','_')+"_CU_2D"+str(N)+'_betaP.save'
    loadfile = "RND_Loading_xx_S_"+str(seed)+"_constantstress_"+str(0.01)+"_2D"+str(N)+"_betaP.save"

    FinialStrain = 2.
    dynamics = StrainLoadingBetaPDynamics(Lambda=Lambda,rate=Rate,type=Type,initial=Initial)
    mover = FieldMover.TVDRungeKutta_FieldMover(CFLsafeFactor=0.5,dtBound=0.01)
    obsState = Observer.RecallStateObserver()
    filename = loadfile 

    startTime = 0. 
    #endTime   = FinialStrain/strainrate
    endTime   = 200.0

    t = startTime 
    dt = 1.
    if startTime == 0. :
        recordState = Observer.RecordStateObserver(filename=filename)
        recordState.Update(t, state)
    else:
        T,state = FieldInitializer.LoadState(filename,t)
        print "we restart the simulation from T = "+str(T)
        recordState = Observer.RecordStateObserver(filename=filename,mode='a')
    system= PlasticitySystem.PlasticitySystem(gridShape, state, mover, dynamics, [obsState,Observer.VerboseTimestepObserver()])
    while t<=(endTime):
        preT = t
        t += dt
        system.Run(startTime=preT, endTime = t)
        system.state = obsState.state
        recordState.Update(t, system.state)


def CyclicLoading():
    N = 128 
    gridShape = (N,N)

    Lambda = 1
    if Lambda == 0:
        motion = 'GlideClimb'
    else:
        motion = 'GlideOnly'

    strainrate = 0.05
    direction = numpy.array([0.,0.,1.])
    Rate = strainrate*direction
    Type = 'strain'
    strains = numpy.array([0.,2.,-2.,0.,2.])

    dir = "No Backup/initialfiles/"
    lengthscale = 1.0 
    relaxfile = dir+"Uni_zz_CU_lengthscale_"+str(lengthscale).replace('.','_')+'_2D256_'+motion+'_betaP.save'
    timef,state = FieldInitializer.LoadState(relaxfile)
    state = FieldInitializer.ResizeState(state,N)
    state = FieldInitializer.ReformatState(state)

    loadfile = dir+"UNI_zz_S_"+str(strainrate).replace(".","_")+"_CU_2D"+str(N)+'_'+motion+'_betaP.save'

    mover = FieldMover.TVDRungeKutta_FieldMover(CFLsafeFactor=0.2)
    obsState = Observer.RecallStateObserver()

    filename = loadfile 
    startTime = 0. 
    t = startTime 
    
    if startTime == 0. :
        recordState = Observer.RecordStateObserver(filename=filename)
        recordState.Update(t, state)
    else:
        T,state = FieldInitializer.LoadState(filename,t)
        print "we restart the simulation from T = "+str(T)
        recordState = Observer.RecordStateObserver(filename=filename,mode='a')
        t = T

    newt = t
    newstate = state
    for i in range(len(strains)-1):
        sign = (strains[i+1]>strains[i])*1.+(strains[i+1]<strains[i])*(-1.)
        dynamics = StrainLoadingBetaPDynamics(Lambda=Lambda,rate=sign*Rate,initial=strains[i]*direction,initT=newt)
        system= PlasticitySystem.PlasticitySystem(gridShape, newstate, mover, dynamics, [obsState])
        tf = newt + (strains[i+1]-strains[i])/(sign*strainrate)
        while t<=(tf):
            preT = t
            if t-ti<=1.:
                dt = 0.1
            elif t-ti<=5.:
                dt = 0.5
            else:
                dt = 1.
            t += dt
            system.Run(startTime=preT, endTime = t)
            system.state = obsState.state
            recordState.Update(t, system.state)
            newstate = system.state
            newt = t

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"s:")
    except:
        sys.exit(2)
    for opt,arg in opts:
        if opt == '-s':
            seed = int(arg)
    ExternalLoading2D_CU(seed=seed)

if __name__ == "__main__":
    main(sys.argv[1:])

