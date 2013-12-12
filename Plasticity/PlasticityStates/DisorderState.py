from Plasticity.Fields import Fields
from Plasticity import NumericalMethods
from Plasticity.PlasticityStates.VacancyState import VacancyState

from Plasticity.Constants import *
import numpy
import itertools

ME = NumericalMethods.ME

class DisorderState(VacancyState):
    def __init__(self,gridShape, field=None, nu=0.3, mu=0.5, inherit=None):
        VacancyState.__init__(self, gridShape,field=field,nu=nu,mu=mu,inherit=inherit)

    def CalculateSigma(self,type='k',source='betaE', dodisorder=True):
        sigma = VacancyState.CalculateSigma(self,type=type, source=source)
        rho = self.CalculateRhoFourier()

        rhomod = NumericalMethods.ME
        for m,n in itertools.product([x,y,z], [x,y,z]):
            rhomod += 2*rho[m,n]*rho[m,n]

        rhoprime = Fields.TensorField(self.gridShape, components=rho.components)
        for i,j in itertools.product([x,y,z], [x,y,z]):
            rhoprime[i,j] = self.betaP_V['s','s'] * rho[i,j] / rhomod
        frhoprime = rhoprime.FFT()

        for i,j,l,t in itertools.product([x,y,z], [x,y,z], [x,y,z], [x,y,z]):
            if perm[l,i,t] != 0 and (self.dimension == 3 or (self.dimension == 2 and l != 'z')):
                sigma[i,j] += perm[l,i,t] * numpy.fft.irfftn( self.ktools.k[l] * frhoprime[t,j] ) #.IFFT()
        return sigma

    def CalculateVacancyEnergy(self):
        return  0.5*self.alpha*self.betaP_V['s','s']*self.betaP_V['s','s']
 
