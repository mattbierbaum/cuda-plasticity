from Plasticity.Fields import Fields
from Plasticity import NumericalMethods
from Plasticity.PlasticityStates import PlasticityState

from Plasticity.Constants import *

ME = NumericalMethods.ME

class VacancyState(PlasticityState.PlasticityState):
    def __init__(self,gridShape, field=None, nu=0.3, mu=0.5, inherit=None, alpha=1.):
        PlasticityState.PlasticityState.__init__(self,gridShape,field=field,nu=nu,mu=mu,inherit=inherit)
        self.alpha = alpha

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['ktools']
        del odict['sinktools']
        odict['field'] = self.GetOrderParameterField().numpy_arrays()
        #FIXME - must be different for subclasses
        del odict['betaP_V']
        return odict

    def GetOrderParameterField(self):
        return self.betaP_V

    def UpdateOrderParameterField(self, betaP_V):
        if betaP_V is None:
            self.betaP_V = Fields.TensorScalarMixedField(self.gridShape)
        else:
            self.betaP_V = betaP_V 
 
    def ExtractBetaPField(self):
        betaP = Fields.TensorField(self.gridShape)
        for comp in betaP.components:
            betaP[comp] = self.betaP_V[comp]
        return betaP
 
    def DecoupleState(self):
        betaPstate = PlasticityState.PlasticityState(self.gridShape,field=self.ExtractBetaPField())
        return betaPstate, self.betaP_V['s','s']
      
    def RecoupleState(self,betaPstate,cfield):
        for com in betaPstate.betaP.components:
            self.betaP_V[com] = betaPstate.betaP[com]    
        self.betaP_V['s','s'] = cfield

    def CalculateRhoFourier(self,type='k'):
        betaP = self.ExtractBetaPField()
        KbetaP = betaP.FFT()
        if type == 'sinK':
            K = self.sinktools
        elif type == 'k':
            K = self.ktools
        DKbetaP = []
        dimension = len(self.gridShape)
        if dimension == 1:
            DKbetaP.append(KbetaP*1.0j*K.kz)
        elif dimension == 2:
            DKbetaP.append(KbetaP*1.0j*K.kx)
            DKbetaP.append(KbetaP*1.0j*K.ky)
        elif dimension == 3:
            DKbetaP.append(KbetaP*1.0j*K.kx)
            DKbetaP.append(KbetaP*1.0j*K.ky)
            DKbetaP.append(KbetaP*1.0j*K.kz)
        Krho = Fields.TensorField(self.gridShape, kspace=True)
        if   dimension == 1:
            for i in [x,y,z]:
                Krho[x,i] = DKbetaP[0][y,i]
                Krho[y,i] = -DKbetaP[0][x,i]
        elif dimension == 2:
            for i in [x,y,z]:
                Krho[x,i] = -DKbetaP[1][z,i]
                Krho[y,i] = DKbetaP[0][z,i]
                Krho[z,i] = DKbetaP[1][x,i] - DKbetaP[0][y,i]
        elif dimension == 3:
            for i in [x,y,z]:
                Krho[x,i] = DKbetaP[2][y,i] - DKbetaP[1][z,i]
                Krho[y,i] = DKbetaP[0][z,i] - DKbetaP[2][x,i]
                Krho[z,i] = DKbetaP[1][x,i] - DKbetaP[0][y,i]
        rho = Krho.IFFT()
        return rho
 
    def CalculateRhoSymmetric(self):
        betaP = self.ExtractBetaPField()
        DbetaP = []
        for dim in range(len(self.gridShape)):
            DbetaPforSpecificDirection = Fields.TensorField(self.gridShape)
            for component in DbetaPforSpecificDirection.components:
                DbetaPforSpecificDirection[component] = NumericalMethods.SymmetricDerivative(betaP[component], self.gridShape, dim)
            DbetaP.append(DbetaPforSpecificDirection)
        rho = Fields.TensorField(self.gridShape)
        if   self.dimension == 1:
            for i in [x,y,z]:
                rho[x,i] = DbetaP[0][y,i]
                rho[y,i] = -DbetaP[0][x,i]
        elif self.dimension == 2:
            for i in [x,y,z]:
                rho[x,i] = -DbetaP[1][z,i]
                rho[y,i] = DbetaP[0][z,i]
                rho[z,i] = DbetaP[1][x,i] - DbetaP[0][y,i]
        elif self.dimension == 3:
            for i in [x,y,z]:
                rho[x,i] = DbetaP[2][y,i] - DbetaP[1][z,i]
                rho[y,i] = DbetaP[0][z,i] - DbetaP[2][x,i]
                rho[z,i] = DbetaP[1][x,i] - DbetaP[0][y,i]
        return rho

    def CalculateSigma(self,type='k',source='betaE'):
        mu = self.mu
        nu = self.nu
        if source == 'betaE':
            sigma = Fields.TensorField(self.gridShape)
            betaE = self.CalculateBetaElastic()
            lamb = 2.*mu*nu/(1.-2.*nu)
            betaE_tr = betaE[x,x]+betaE[y,y]+betaE[z,z]
            for com in sigma.components:
                i,j = com
                sigma[com] += lamb*betaE_tr*(i==j)+mu*(betaE[i,j]+betaE[j,i])
        elif source == 'rho':
            if type == 'sinK':
                K = self.sinktools
            elif type == 'k':
                K = self.ktools
            k={}
            k[x]=K.kx
            k[y]=K.ky
            k[z]=K.kz
            kSq  = K.kSq + ME
            rho = self.CalculateRhoFourier(type)
            Krho = rho.FFT()
            Ksigma = Fields.SymmetricTensorField(self.gridShape,kspace=True) 
            for compSigma in Ksigma.components:
                i,j = compSigma
                for compRho in Krho.components:
                    m,n = compRho
                    for s in [x,y,z]:
                        K_ijmn = ((-1.j)*mu*k[s]/kSq)*(perm[i,n,s]*(j==m)+perm[j,n,s]*(i==m)+\
                                                      (2.*perm[m,n,s]/(1.-nu))*(k[i]*k[j]/kSq-(i==j))) 
                        Ksigma[i,j] += K_ijmn*Krho[m,n]
                if self.dimension == 1:
                    Ksigma[i,j][0] = complex(0.)
                elif self.dimension == 2:
                    Ksigma[i,j] *= self.ktools.kmask
                elif self.dimension == 3:
                    Ksigma[i,j][0,0,0] = complex(0.)
            sigma = Ksigma.IFFT()
        elif source == 'betaP':
            betaP = self.ExtractBetaPField()
            Ksigma = Fields.SymmetricTensorField(self.gridShape, kspace=True)
            KbetaP = betaP.FFT()
            if type == 'sinK':
                K = self.sinktools
            elif type == 'k':
                K = self.ktools
            k = {}
            k[x] = K.kx
            k[y] = K.ky
            k[z] = K.kz 
            kSq  = K.kSq + ME
            kSqSq= K.kSqSq + ME
            for compSigma in Ksigma.components:
                i,j = compSigma
                for compBetaP in KbetaP.components:
                    m,n = compBetaP
                    M_ijmn = (2.*mu*nu/(1.-nu))*((k[m]*k[n]*(i==j)+k[i]*k[j]*(m==n))/kSq - (i==j)*(m==n)) -\
                             mu*((i==m)*(j==n)+(i==n)*(j==m)) - (2.*mu/(1.-nu))*k[i]*k[j]*k[m]*k[n]/kSqSq +\
                             mu*(k[i]*k[n]*(j==m)+k[i]*k[m]*(j==n)+k[j]*k[n]*(i==m)+k[j]*k[m]*(i==n))/kSq
                    Ksigma[i,j] += M_ijmn*KbetaP[m,n]
                if self.dimension == 1:
                    Ksigma[i,j][0] = complex(0.)
                elif self.dimension == 2:
                    Ksigma[i,j] *= self.ktools.kmask
                elif self.dimension == 3:
                    Ksigma[i,j][0,0,0] = complex(0.)
            sigma = Ksigma.IFFT()
            """
            Doing the following correction will be equivalent to the calculation from betaE.
            Romoving them will return to the same result calculated from rho.
            """
            lamb = 2.*mu*nu/(1.-2.*nu)
            betaP_tr = betaP[x,x].average()+betaP[y,y].average()+betaP[z,z].average()
            for com in sigma.components:
                i,j = com
                sigma[com] -= lamb*betaP_tr*(i==j)+mu*(betaP[i,j].average()+betaP[j,i].average())
        return sigma

    def CalculateSigmawithVacancy(self,type='k',source='betaE',cfield=None):
        sigma = self.CalculateSigma(type=type,source=source)
        
        if cfield is None:
            for i in [x,y,z]:
                sigma[i,i] -= self.alpha*self.betaP_V['s','s'] 
        else:
            for i in [x,y,z]:
                sigma[i,i] -= self.alpha*cfield 
        return sigma
 
    def CalculateVacancyEnergy(self):
        return  0.5*self.alpha*self.betaP_V['s','s']*self.betaP_V['s','s']
 
    def CalculateDisplacementField(self,type='k'):
        KbetaP = self.ExtractBetaPField().FFT()
        Ku = Fields.TensorField(self.gridShape, kspace=True, components=[x,y,z])
        mu = self.mu
        nu = self.nu
        lamb = 2. * mu * nu / (1. - 2*nu)
        if type == 'sinK':
            K = self.sinktools
        else:
            K = self.ktools
        kSq = K.kSq+ME
        for i in Ku.components:
            for Pcomp in KbetaP.components:
                m,n = Pcomp
                L_imn = -(1.0j/kSq) * ( nu/(1-nu) * K.k[i] * (m==n) + \
                         K.k[m]*(i==n) + K.k[n]*(i==m) -\
                         (1./(1.-nu))*(K.k[i]*K.k[m]*K.k[n])/kSq )
                Ku[i] += KbetaP[m,n] * L_imn
            if self.dimension == 1:
                Ku[i][0] *= complex(0.)
            elif self.dimension == 2:
                Ku[i][0,0] *= complex(0.)
            elif self.dimension == 3:
                Ku[i][0,0,0] *= complex(0.)
        u = Ku.IFFT()  
        return u

    def CalculateBetaElastic(self,type='k'):
        betaP = self.ExtractBetaPField()
        KbetaP = betaP.FFT()
        KbetaE = Fields.TensorField(self.gridShape,kspace=True)
        ### Doing this will cause to lose the mean value.
        #KbetaE = KbetaP * (-1.0)
        mu = self.mu
        nu = self.nu
        lamb = 2. * mu * nu / (1. - 2*nu)
        if type == 'sinK':
            K = self.sinktools
        elif type == 'k':
            K = self.ktools
        k = {}
        k[x] = K.kx
        k[y] = K.ky
        k[z] = K.kz 
        kSq  = K.kSq+ME
        kSqSq= K.kSqSq+ME
        for Ecomp in KbetaE.components:
            i,j = Ecomp
            for Pcomp in KbetaP.components:
                m,n = Pcomp
                T_ijmn = lamb/(lamb+2.*mu) * k[i]*k[j]/kSq * (m==n) + \
                    k[i]/kSq*(k[m]*(j==n)+k[n]*(j==m)) -\
                    2.*(lamb+mu)/(lamb+2.*mu)*(k[i]*k[j]*k[m]*k[n])/kSqSq
                KbetaE[i,j] += KbetaP[m,n] * T_ijmn
            if self.dimension == 1:
                KbetaE[i,j][0] = complex(0.)
            elif self.dimension == 2:
                KbetaE[i,j] *= self.ktools.kmask
            elif self.dimension == 3:
                KbetaE[i,j][0,0,0] = complex(0.)
        betaE = KbetaE.IFFT()  
        betaE -= betaP
        return betaE

    def CalculateRotationRodrigues(self,type='k',source='betaP'):
        if source == 'betaE':
            rotE = self.CalculateRotationField(type)
            rodrigues = Fields.TensorField(self.gridShape, components=[x,y,z])   
            rodrigues[x] = rotE[y,z]-rotE[z,y]
            rodrigues[y] = rotE[z,x]-rotE[x,z]
            rodrigues[z] = rotE[x,y]-rotE[y,x]
            rodrigues /= 2.
        elif source == 'betaP':
            Krodrigues = Fields.TensorField(self.gridShape, components=[x,y,z], kspace=True)   
            betaP = self.ExtractBetaPField()
            KbetaP = betaP.FFT()
            if type == 'sinK':
                K = self.sinktools
            elif type == 'k':
                K = self.ktools
            kSq = K.kSq+ME
            for i in [x,y,z]:
                for comp in KbetaP.components:
                    s,t = comp
                    for j in [x,y,z]:
                        Krodrigues[i] += K.k[j]*(perm[i,j,t]*K.k[s]+perm[i,j,s]*K.k[t])*KbetaP[s,t]/kSq/2.
                if self.dimension == 1:
                    Krodrigues[i][0] = complex(0.)
                elif self.dimension == 2:
                    Krodrigues[i] *= self.ktools.kmask
                elif self.dimension == 3:
                    Krodrigues[i][0,0,0] = complex(0.)
            rodrigues = Krodrigues.IFFT()
            rodrigues[x] -= (betaP[y,z]-betaP[z,y])/2.  
            rodrigues[y] -= (betaP[z,x]-betaP[x,z])/2.  
            rodrigues[z] -= (betaP[x,y]-betaP[y,x])/2.  
        elif source == 'rho':
            """
            Using this source will lose the mean values.
            """
            Krodrigues = Fields.TensorField(self.gridShape, components=[x,y,z], kspace=True)   
            rho = self.CalculateRhoFourier(type) 
            KRho = rho.FFT() 
            if type == 'sinK':
                K = self.sinktools
            elif type == 'k':
                K = self.ktools
            kSq = K.kSq+ME
            kSqSq = K.kSqSq+ME
            for i in [x,y,z]:
                for Rhocomp in KRho.components:
                    m,n = Rhocomp
                    D_imn = ((1.j)/(2.*kSq))*(2.*K.k[n]*(i==m)-K.k[m]*(i==n)-K.k[i]*(m==n)) +\
                            (1.j)*K.k[i]*K.k[n]*K.k[m]/(2.*kSqSq) 
                    Krodrigues[i] += D_imn * KRho[m,n]
                if self.dimension == 1:
                    Krodrigues[i][0] = complex(0.)
                elif self.dimension == 2:
                    Krodrigues[i] *= self.ktools.kmask
                elif self.dimension == 3:
                    Krodrigues[i][0,0,0] = complex(0.)
            rodrigues = Krodrigues.IFFT()
        return rodrigues 


