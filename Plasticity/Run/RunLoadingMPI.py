import PlasticitySystem
import FieldInitializer
import FieldDynamics
import FieldMover
import CentralUpwindHJBetaPDynamics
import Observer

import sys
import os
import getopt
import numpy
from numpy import fft
    
from Constants import *


mu,nu = 0.5,0.3 
lamb = 2.*mu*nu/(1.-2.*nu)

class EnergyDownhillChecking(Observer.Observer):
    def __init__(self,initStrain,rate,vol):
        self.preT = 0.
        self.preEnergy = None
        self.presigmazz = None
        self.initStrain = initStrain
        self.vol = vol 
        self.rate = rate
        self.work = 0.
        self.count = 0
    
    def Update(self,time,state):
        betaE_zz = self.rate*time-(numpy.average(state.betaP[z,z])-self.initStrain)
        energy = self.CalculateEnergy(state,betaE_zz) 
        sigmazz = (lamb+2.*mu)*betaE_zz 
        if self.count > 0:
            self.work += (sigmazz + self.presigmazz)*(time-self.preT)*self.rate/2. 
            energy -= self.work*self.vol
        if self.count>0 and self.preEnergy < energy:
            sys.exit(1)
        self.preEnergy = energy
        self.preT = time
        self.presigmazz = sigmazz
        self.count += 1

    def CalculateEnergy(self,state,betaE_zz):
        sigma = state.CalculateSigma()
        for com in sigma.components:
            i,j = com
            sigma[com] += (lamb*(i==j)+2.*mu*(i==z)*(j==z))*betaE_zz 
        energydensity = 0.
        xyz = [x,y,z]
        for i in xyz:
            for j in xyz:
                for k in xyz:
                    for m in xyz:     
                        S = (2.*(i==k)*(j==m)-(i==m)*(j==k)-nu*(i==j)*(k==m)/(1.+nu))/(2.*mu)
                        energydensity += 0.5*S*sigma[i,j]*sigma[k,m]
        return energydensity.sum()
     


class UniaxialLoadingBetaPDynamics(CentralUpwindHJBetaPDynamics.BetaPDynamics):
    def __init__(self, Dx=None, Lambda=0, coreEnergy=0, coreEnergyLog=0, rate=None, initStrain=None):
        CentralUpwindHJBetaPDynamics.BetaPDynamics.__init__(self,Dx,Lambda,coreEnergy,coreEnergyLog)
        self.initStrain = initStrain
        self.rate = rate

    def GetSigma(self,state,time):
        sigma = state.CalculateSigma()
        """
        Stress-rate loading.
        """
        #sigma[z,z] += self.rate*time 
        """
        Strain-rate loading.
        """
        mu,nu = state.mu,state.nu
        lamb = 2.*mu*nu/(1.-2.*nu)
        betaE_zz = self.rate*time-(numpy.average(state.betaP[z,z])-self.initStrain)
        for com in sigma.components:
            i,j = com
            sigma[com] += (lamb*(i==j)+2.*mu*(i==z)*(j==z))*betaE_zz 
        return sigma

def ExternalLoading2D_CU(seed):
    N = 256 
    gridShape = (N,N)

    Lambda = 1
    if Lambda == 0:
        motion = 'GlideClimb'
    else:
        motion = 'GlideOnly'

    Rate = 0.05

    lengthscale = 0.2 
    relaxfile = "CU_S_"+str(seed)+"_2D"+str(N)+"_lengthscale_"+str(lengthscale).replace(".","_")+"_"+motion+".save"
    os.system("msscmd cd result, get "+relaxfile)
    timef,state = FieldInitializer.LoadState(relaxfile)

    loadfile = "ZZLoading_S_"+str(seed)+"_rate_"+str(Rate).replace(".","_")+"_CU_2D"+str(N)+'_betaP.save'

    FinialStrain = 5.
    dynamics = UniaxialLoadingBetaPDynamics(Lambda=Lambda,rate=Rate,initStrain=numpy.average(state.betaP['z','z']))

    mover = FieldMover.TVDRungeKutta_FieldMover(CFLsafeFactor=0.5, dtBound=0.01)

    obsState = Observer.RecallStateObserver()
    energyChecking = EnergyDownhillChecking(numpy.average(state.betaP['z','z']),Rate,N**2)

    startTime = 0. 
    endTime   = FinialStrain/Rate

    filename = loadfile 

    t = startTime 
    dt = 1.
    if startTime == 0. :
        recordState = Observer.RecordStateObserver(filename=filename)
        recordState.Update(t, state)
    else:
        T,state = FieldInitializer.LoadState(filename)
        recordState = Observer.RecordStateObserver(filename=filename,mode='a')

    system= PlasticitySystem.PlasticitySystem(gridShape, state, mover, dynamics, [obsState,energyChecking])

    while t<=(endTime):
        preT = t
        t += dt
        system.Run(startTime=preT, endTime = t)
        system.state = obsState.state
        recordState.Update(t, system.state)


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

