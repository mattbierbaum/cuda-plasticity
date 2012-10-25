#!/usr/bin/python2.4 
from Plasticity import PlasticitySystem
from Plasticity.FieldInitializers import FieldInitializer, WallInitializer
from Plasticity.FieldDynamics import FieldDynamics
from Plasticity.FieldMovers import FieldMover
from Plasticity.FieldDynamics import VacancyDynamics
from Plasticity.Observers import Observer
from Plasticity.PlasticityStates import VacancyState, PlasticityState
from Plasticity.Constants import *
from Plasticity.GridArray import GridArray
from Plasticity.FieldDynamics import CentralUpwindHJBetaPDynamics, CentralUpwindHJBetaPGlideOnlyDynamics
from Plasticity import NumericalMethods
from Plasticity.Fields import Fields
import pylab
import numpy
import scipy.weave as weave

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

def local_get(store, dir, file):
    return os.system("cp "+store+"/"+dir+"/"+file+" . ")

def local_put(store, dir, file):
    return os.system("cp "+file+" "+store+"/"+dir)

def convertStateToCUDA(state, filename):
    arr = []
    order = [('x', 'x'), ('x', 'y'), ('x', 'z'),
             ('y', 'x'), ('y', 'y'), ('y', 'z'),
             ('z', 'x'), ('z', 'y'), ('z', 'z'), ('s', 's')] 
    try:  
        for c in order:
            arr.append( state.betaP_V[c].transpose().copy() )
    except:
        for c in state.betaP.components:
            arr.append( state.betaP[c].transpose().copy() )

    arr = numpy.array(arr)
    arr.tofile(filename)

#===========================================================
# start of the simulation function
from Plasticity.Configure import *
from Plasticity.TarFile import * 
import shutil as sh
import os

def simulation(dct, get_file=local_get, put_file=local_put):
    conf = Configuration(dct)
    homedir = conf.homedir
    N       = conf.N
    dim     = conf.dim
    previous= conf.previous
    postfix = conf.postfix
    method  = conf.method
    device  = conf.device
    seed    = conf.seed
    hash    = conf.hash

    prefix = method
    gridShape = (N,)*dim
    lengthscale = 0.2

    directory = prefix+str(len(gridShape))+"d"+str(N)
    unique    = directory+"_s"+str(seed)
    oldstub   = unique+"_"+previous
    oldfile   = unique+"_"+previous+".tar"
    currstub  = unique+"_"+previous+postfix
    currfile  = unique+"_"+previous+postfix+".tar"
   
    file_output= currstub+".plas"
    if os.path.isfile(currfile) == False:
        get_file(homedir, directory, currfile) 
    if os.path.isfile(currfile) == True:
        tar_extract(tar_open(currfile), ".plas")
        file_input = currstub+".plas"
    else:
        if previous == "":
            state = FieldInitializer.GaussianRandomInitializer(gridShape, lengthscale,seed,vacancy=None)
        else:
            get_file(homedir, directory, oldfile)
            tt,state = LoadTarState(oldfile)
        file_input = currstub + ".ics" 
        convertStateToCUDA(state, file_input)

    confname = currstub+".conf"
    write_json(conf, confname) 

    # ====================================================
    # begin non-standard things now
    if method == "mdp":
        if "load" in conf:
            dynamics = UpwindLoadingBetaPDynamics(Lambda=1,rate=conf["load_dir"],\
                        initial=conf["load_start"],type=conf["load_tye"])
        else:
            dynamics = CentralUpwindHJBetaPDynamics.BetaPDynamics(Lambda=1)
        mover = FieldMover.TVDRungeKutta_FieldMover(CFLsafeFactor=0.1)

    if method == "gcd":
        if "load" in conf:
            dynamics = UpwindLoadingBetaPDynamics(Lambda=0,rate=conf["load_dir"],\
                        initial=conf["load_start"],type=conf["load_type"])
        else: 
            dynamics = CentralUpwindHJBetaPDynamics.BetaPDynamics(Lambda=0)
        mover = FieldMover.TVDRungeKutta_FieldMover(CFLsafeFactor=0.1)

    if method == "lvp":
        if "load" in conf:
            dynamics = NewGlideOnlyLoadDynamics(rate=conf["load_rate"],\
                        initial=conf["load_start"],type=conf["load_type"])
        else:
            dynamics = CentralUpwindHJBetaPGlideOnlyDynamics.BetaPDynamics()
        mover = FieldMover.TVDRungeKutta_FieldMover(CFLsafeFactor=0.1)

    obsState = Observer.RecallStateObserver()

    startTime = 0. 
    endTime   = 30.
    dt = 0.025

    t = startTime 
    if startTime == 0. :
        recordState = Observer.RecordStateObserverRaw(filename=file_output)
        recordState.Update(t, state)
    else:
        T,state = FieldInitializer.LoadStateRaw(file_input, N, dim, hastimes=False)
        recordState = Observer.RecordStateObserverRaw(filename=file_output,mode='a')

    system=PlasticitySystem.PlasticitySystem(gridShape, state, mover, dynamics, [obsState,Observer.VerboseTimestepObserver()])

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

    #=======================================================
    # begin standard things again
    if os.path.isfile(confname):
        sh.move(confname, currstub)
    if os.path.isfile(file_input):
        sh.move(file_input, currstub)
    if os.path.isfile(file_output):
        sh.move(file_output, currstub)
    os.system("tar -cvf "+currfile+" "+currstub)
    os.system("rm -rf "+currstub)
    put_file(homedir, directory, currfile)  
 

def main():
    Relaxation_BetaPV()

if __name__ == "__main__":
    main()



