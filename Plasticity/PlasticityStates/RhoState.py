from Plasticity.Fields import Fields
from Plasticity import NumericalMethods
from Plasticity.PlasticityStates import PlasticityState

from Constants import *

ME = NumericalMethods.ME

class RhoState(PlasticityState.PlasticityState):
    def __init__(self, gridShape, field=None, nu=0.3, mu=0.5, inherit=None, betaP=None):
        if betaP is not None:
            PlasticityState.PlasticityState.__init__(self,gridShape,nu=nu,mu=mu,inherit=inherit)
            self.betaP = betaP
            rho = PlasticityState.PlasticityState.CalculateRhoSymmetric(self)
            self.UpdateOrderParameterField(rho)
        elif field is not None:
            PlasticityState.PlasticityState.__init__(self,gridShape,field=field,nu=nu,mu=mu,inherit=inherit)
        else:
            assert False

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['ktools']
        del odict['sinktools']
        odict['field'] = self.GetOrderParameterField().numpy_arrays()
        #FIXME - must be different for subclasses
        del odict['rho']
        return odict

    def GetOrderParameterField(self):
        return self.rho

    def UpdateOrderParameterField(self,rho):
        if rho is None:
            self.rho = Fields.TensorField(self.gridShape)
        else:
            self.rho = rho

    def CalculateRhoSymmetric(self):
        return self.rho
    
    def CalculateRhoFourier(self):
        assert False
        """ Either the fourier or the symmetric rho calculation must be forbidden for consistency """
        return self.rho

    def CalculateBetaP(self):
        K = self.sinktools
        KbetaP = Fields.TensorField(self.gridShape, kspace=True)
        
        Krho = self.rho.FFT()
        for comp in KbetaP.components:
            i,j = comp
            for m in [x,y,z]:
                for n in [x,y,z]:
                    KbetaP[i,j] += perm[i,m,n]*K.k[m]*Krho[n,j]
        KbetaP *= -1.0j/(K.kSq+ME)
        return KbetaP.IFFT()

    def CalculateSigma(self,type='k'):
        mu = self.mu
        nu = self.nu
        if type == 'sinK':
            K = self.sinktools
        elif type == 'k':
            K = self.ktools
        k={}
        k[x]=K.kx
        k[y]=K.ky
        k[z]=K.kz
        kSq  = K.kSq + ME
        rho = self.rho
        Krho = rho.FFT()
        Ksigma = Fields.TensorField(self.gridShape,kspace=True) 
        for compSigma in Ksigma.components:
            i,j = compSigma
            for compRho in Krho.components:
                m,n = compRho
                for s in [x,y,z]:
                    K_ijmn = ((-1.j)*mu*k[s]/kSq)*(perm[i,n,s]*(j==m)+perm[j,n,s]*(i==m)+\
                                                  (2.*perm[m,n,s]/(1.-nu))*(k[i]*k[j]/kSq-(i==j))) 
                    Ksigma[i,j] += K_ijmn*Krho[m,n]
            if self.dimension == 1:
                Ksigma[i,j][0] *= complex(0.)
            elif self.dimension == 2:
                Ksigma[i,j][0,0] *= complex(0.)
            elif self.dimension == 3:
                Ksigma[i,j][0,0,0] *= complex(0.)
        sigma = Ksigma.IFFT()
        return sigma


    def CalculateRotationRodrigues(self,type='k'):
        """
        Calculate the rotation vector field using modified version of
        Yor's formula with elastic strain.
        """
        Krodrigues = Fields.TensorField(self.gridShape, components=[x,y,z], kspace=True)   
        rho = self.rho 
        KRho = rho.FFT() 
        if type == 'sinK':
            K = self.sinktools
        elif type == 'k':
            K = self.ktools
        k = {}
        k[x] = K.kx
        k[y] = K.ky
        k[z] = K.kz 
        kSq = K.kSq+ME
        kSqSq = K.kSqSq+ME
        for i in [x,y,z]:
            for Rhocomp in KRho.components:
                m,n = Rhocomp
                D_imn = ((1.j)/(2.*kSq))*(2.*k[n]*(i==m)-k[m]*(i==n)-k[i]*(m==n)) +\
                        (1.j)*k[i]*k[n]*k[m]/(2.*kSqSq) 
                Krodrigues[i] += D_imn * KRho[m,n]
            if self.dimension == 1:
                Krodrigues[i][0] *= complex(0.)
            elif self.dimension == 2:
                Krodrigues[i][0,0] *= complex(0.)
            elif self.dimension == 3:
                Krodrigues[i][0,0,0] *= complex(0.)
        rodrigues = Krodrigues.IFFT()
        return rodrigues

