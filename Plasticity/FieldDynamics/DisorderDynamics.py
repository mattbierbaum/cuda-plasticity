from Plasticity.FieldDynamics.CentralUpwindHJ import *
from Plasticity.Constants import *
from Plasticity.FieldDynamics.CentralUpwindHJBetaPDynamics3D import *

from Plasticity.PlasticityStates import PlasticityState
from Plasticity.GridArray import GridArray

from Plasticity import NumericalMethods
from Plasticity.Fields import Fields

import numpy
import itertools

"""
Define H-J type dynamics
"""
class BetaP_DisorderDynamics(BetaPDynamics): 
    def __init__(self, Dx=None, Lambda=0, coreEnergy=0, coreEnergyLog=0, disorder=None):
        BetaPDynamics.__init__(self,Dx,Lambda=Lambda,coreEnergy=coreEnergy,coreEnergyLog=coreEnergyLog) 
        self.disorder = disorder

    def GetSigma(self, state, time):
        sigma = state.CalculateSigma()
        
        rho = state.CalculateRhoFourier()

        rhomod = NumericalMethods.ME
        for m,n in itertools.product([x,y,z], [x,y,z]):
            rhomod += 2*rho[m,n]*rho[m,n]

        rhoprime = Fields.TensorField(state.gridShape, components=rho.components)
        for i,j in itertools.product([x,y,z], [x,y,z]):
            rhoprime[i,j] = self.disorder * rho[i,j] / rhomod
        frhoprime = rhoprime.FFT()

        for i,j,l,t in itertools.product([x,y,z], [x,y,z], [x,y,z], [x,y,z]):
            if perm[l,i,t] != 0 and (state.dimension == 3 or (state.dimension == 2 and l != 'z')):
                sigma[i,j] += perm[l,i,t] * numpy.fft.irfftn( state.ktools.k[l] * frhoprime[t,j] ) #.IFFT()
        return sigma

    def CalculateFlux(self, time, state, CFLCondition=False):
        self.sigma = self.GetSigma(state,time)
        ret = CentralUpwindHJDynamics.CalculateFlux(self,time,state,CFLCondition=CFLCondition) 
        return ret

