from CentralUpwindHJ import *
from Constants import *
from CentralUpwindHJBetaPDynamics3D import *

import PlasticityState
import GridArray

import NumericalMethods
import Fields

import numpy

"""
Define H-J type dynamics
"""
class BetaP_VacancyDynamics(BetaPDynamics): 
    def __init__(self, Dx=None, Lambda=0, coreEnergy=0, coreEnergyLog=0, gamma = 1., alpha = 1., beta=1.):
        BetaPDynamics.__init__(self,Dx,Lambda=Lambda,coreEnergy=coreEnergy,coreEnergyLog=coreEnergyLog) 
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

    def GetSigma(self, state, time, cfield):
        #print "sigma 1: ",; GridArray.print_mem_usage()
        sigma = state.CalculateSigma()
        for i in [x,y,z]:
            sigma[i,i] -= self.alpha*cfield 
        #print "sigma 2: ",; GridArray.print_mem_usage()
        return sigma

    def CalculateFlux(self, time, betaPstate, cfield, CFLCondition=False):
        #print "flux 1: ",; GridArray.print_mem_usage()
        self.sigma = self.GetSigma(betaPstate,time,cfield)
        ret = CentralUpwindHJDynamics.CalculateFlux(self,time,betaPstate,CFLCondition=CFLCondition) 
        #print "flux 2: ",; GridArray.print_mem_usage()
        return ret

    def CalculateVFlux(self,cfield,delta_c,dt,ktools):
        cfield += self.beta*delta_c 
        return self.DiffusionEvolution(cfield,dt,ktools)
        
    def DiffusionEvolution(self, cfield, timeStep, ktools):
        """
        This fourier regularization scheme gives the artificial viscosity solution.
        When the diffusion constant goes to zero, the solution theoretically converges
        to the unique weak one of the original PDE.

        Michael G. Crandall and Pierre-Louis Lions
        Viscosity solutions of Hamilton-Jacobi equations, 1983 
        """
        #print "diff 1:",
        #GridArray.print_mem_usage()

        kc = cfield.rfftn()
        kSq = ktools.kSq
        #kSqSq = ktools.kSqSq
        kt = kSq * -self.gamma * timeStep
        kc = kc * kt.exp()
        newc = kc.irfftn()
        #print "diff 2:",
        #GridArray.print_mem_usage()
        """
        kc = numpy.fft.rfftn(cfield)
        kSq = ktools.kSq
        kSqSq = ktools.kSqSq
        kc *= numpy.exp(-self.gamma * timeStep * kSq) 
        newc = numpy.fft.irfftn(kc)
        """

        return newc #GridArray.GridArray(newc)
        
