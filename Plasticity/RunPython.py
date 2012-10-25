#!/usr/bin/python2.4 
import PlasticitySystem
import FieldInitializer
import FieldDynamics
import FieldMover
import VacancyDynamics
import Observer
import pylab
import VacancyState, PlasticityState
import numpy
import scipy.weave as weave
from Constants import *
import GridArray
import CentralUpwindHJBetaPDynamics
import NumericalMethods
import CentralUpwindHJBetaPGlideOnlyDynamics
import WallInitializer
import Fields

# shape and size of sim.
N = 128 
gridShape = (N,N)
 
dir = "./"
lengthscale = 0.2

# load parameters
loadrate = 0.01
loadconst= 0.0
loaddir  = numpy.array([-0.5,-0.5,1.0]) 
loadtype = 'strain'

# vacancy parameters
gamma  = 1e-2
alpha  = 1e0
beta   = 1
c0     = 0

# upwind glide only parameter
Lambda = 0

# old file to use for ics=7
oldfile = "RND_Upwind_L0_S0_2D128.save"


method = 5
"""
0 : Vacancies 
1 : NewGlideOnly
2 : Upwind
3 : Vacancies w/ load 
4 : NewGlideOnly w/ load
5 : Upwind w/ load
"""

ics = 7
"""
0 : random gaussian with lengthscale
1 : 4 point sources
2 : 2 tilt walls - perpendicular point
3 : 2 tilt walls - parallel point
4 : 2 offset parallel point dislocation
5 : Nabarro-Herring hexagons
6 : add walls to a relaxed state
7 : load old file
"""

LoadRate = loadrate*loaddir 
LoadInit = loadconst*loaddir
LoadType = loadtype
loaddstr = '['+str(loaddir[0])+','+str(loaddir[1])+','+str(loaddir[2])+']'


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


class UpwindLoadingBetaPDynamics(CentralUpwindHJBetaPDynamics.BetaPDynamics):
    def __init__(self, Dx=None, Lambda=0, coreEnergy=0, coreEnergyLog=0, rate=None, type='strain', \
                       initial=numpy.array([0.,0.,0.]),initT=0.):
        CentralUpwindHJBetaPDynamics.BetaPDynamics.__init__(self,Dx,coreEnergy,coreEnergyLog)
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


class NewGlideOnlyLoadDynamics(CentralUpwindHJBetaPGlideOnlyDynamics.BetaPDynamics):
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

  
class VacancyDynamicsExternalLoad(VacancyDynamics.BetaP_VacancyDynamics):
    def __init__(self, alpha=1.0, gamma=1.0, beta=1.0, Dx=None, Lambda=0, coreEnergy=0, coreEnergyLog=0, rate=None, type='strain', \
                       initial=numpy.array([0.,0.,0.]),initT=0.):
        VacancyDynamics.BetaP_VacancyDynamics.__init__(self,Dx,coreEnergy,coreEnergyLog,alpha=alpha,gamma=gamma,beta=beta)
        self.rate = rate
        self.type = type
        self.initial = initial
        self.initT = initT

    def GetSigma(self,state,time, cfield):
        sigma = state.CalculateSigma()
        for i in [x,y,z]:
            sigma[i,i] -= self.alpha*cfield 

        if self.type == 'strain':
            return ExternalStrain(sigma,self.initial+self.rate*(time-self.initT)) 
        else:
            return ExternalStress(sigma,self.initial+self.rate*(time-self.initT)) 


def simulation(homedir, cudadir, N, dim, previous, postfix, method, device, seed, header=None, **kwargs):
    prefix = method
    gridShape = (N,)*dim
    lengthscale = 0.2

    directory = prefix+str(len(gridShape))+"d"+str(N)
    unique    = directory+"_s"+str(seed)
    oldstub   = unique+"_"+previous
    oldfile   = unique+"_"+previous+".tar"
    currstub  = unique+"_"+previous+postfix
    currfile  = unique+"_"+previous+postfix+".tar"

    if method == "mdp":
        if "load" in kwargs:
            dynamics = UpwindLoadingBetaPDynamics(Lambda=1,rate=kwargs["load_dir"],\
                        initial=kwargs["load_start"],type=kwargs["load_tye"]))
        else:
            dynamics = CentralUpwindHJBetaPDynamics.BetaPDynamics(Lambda=1)
        mover = FieldMover.TVDRungeKutta_FieldMover(CFLsafeFactor=0.1)

    if method == "gcd":
        if "load" in kwargs:
            dynamics = UpwindLoadingBetaPDynamics(Lambda=0,rate=kwargs["load_dir"],\
                        initial=kwargs["load_start"],type=kwargs["load_type"])
        else: 
            dynamics = CentralUpwindHJBetaPDynamics.BetaPDynamics(Lambda=0)
        mover = FieldMover.TVDRungeKutta_FieldMover(CFLsafeFactor=0.1)

    if method == "lvp":
        if "load" in kwargs:
            dynamics = NewGlideOnlyLoadDynamics(rate=kwargs["load_rate"],\
                        initial=kwargs["load_start"],type=kwargs["load_type"])
        else:
            dynamics = CentralUpwindHJBetaPGlideOnlyDynamics.BetaPDynamics()
        mover = FieldMover.TVDRungeKutta_FieldMover(CFLsafeFactor=0.1)


    filename = dir+info+dlabel+"_L"+str(Lambda)+"_S"+str(seed)+"_"+str(len(gridShape))+"D"+str(N)+".save"
   
    obsState = Observer.RecallStateObserver()
    energychecking = TotalFreeEnergyDownhillObserver()

    startTime = 0. 
    endTime   = 30.
    dt = 0.025

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
        #if t<=0.01-0.001:
        #    dt = 0.001
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

def main():
    Relaxation_BetaPV()

if __name__ == "__main__":
    main()



