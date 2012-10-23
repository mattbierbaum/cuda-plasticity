import Fields
import NumericalMethods

from scipy import fromfunction, sin, pi

import GridArray
import FourierSpaceTools

#from CUDAGridArray import GridArray
#from CUDAFourierSpaceTools import FourierSpaceTools

from Constants import *

ME = NumericalMethods.ME
fourierTools = {}
fourierSinTools = {}


class PlasticityState:
    """ 
    Base class for representing the plasticity state 
    """
    def __init__(self, gridShape, field=None, nu=0.3, mu=0.5, inherit=None):
        self.gridShape = gridShape
        self.dimension = len(gridShape)
        self.UpdateOrderParameterField(field)
        if inherit is not None:
            self.nu = inherit.nu
            self.mu = inherit.mu
            self.ktools = inherit.ktools
            self.sinktools = inherit.sinktools
        else:
            self.nu = nu
            self.mu = mu
            if gridShape not in fourierTools:
                fourierTools[gridShape] = FourierSpaceTools.FourierSpaceTools(gridShape)
                fourierSinTools[gridShape] = FourierSpaceTools.FourierSpaceTools(gridShape, func=sin)
            self.ktools = fourierTools[gridShape]
            self.sinktools = fourierSinTools[gridShape]

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['ktools']
        del odict['sinktools']
        odict['field'] = self.GetOrderParameterField().numpy_arrays()
        #FIXME - must be different for subclasses
        del odict['betaP']
        return odict

    def __setstate__(self,dict):
        self.__dict__.update(dict)
        if self.gridShape not in fourierTools:
            fourierTools[self.gridShape] = FourierSpaceTools.FourierSpaceTools(self.gridShape)
            fourierSinTools[self.gridShape] = FourierSpaceTools.FourierSpaceTools(self.gridShape, func=sin)
        self.ktools = fourierTools[self.gridShape]
        self.sinktools = fourierSinTools[self.gridShape]
        for component in self.field.components:
            self.field[component] = GridArray.GridArray(self.field[component])
        self.UpdateOrderParameterField(self.field)
        del self.__dict__['field']
         
    def evaluate(self):
        """
        Evaluate postponed evaluations in the field
        """
        field = self.GetOrderParameterField()
        field.evaluate()
        
    def GetOrderParameterField(self):
        """
        returns the order parameter field. By default this is beta^P
        """
        return self.betaP

    def UpdateOrderParameterField(self, orderParameterFieldState):
        """
        Updates the order parameter field
        """
        if orderParameterFieldState is None:
            # FIXME - there maybe a better way?
            self.betaP = Fields.TensorField(self.gridShape)
        else:
            self.betaP = orderParameterFieldState

    def CalculateSigma(self,type='k',source='betaE'):
        """
        Calculate the stress field due to the dislocations.

        Implements the equation (8) of
        S. Limkumnerd and J. Sethna, Mesoscale Theory of Grains and Cells: Crystal Plasticity 
        and Coarsening, PRL 96 095503 (2006).

        Formula (1)
        sigma_ij(k) = K_ijkl(k) rho_kl(k)
        K_ijkl(k) = - i \mu k_m / k^2 [ \epsilon_ilm \delta_jk + \epsilon_jlm \delta_ik
                    + 2 \epsilon_klm / (1 - \nu) ( k_i k_j / k^2 - \delta_ij ) ]

        Formula (2)
        sigma_ij(k) = M_ijmn betaP_mn(k)
        M_ijmn = 2\mu\nu/(1-\nu)[k_m*k_n*\delta_ij/k^2 + k_i*k_j*\delta_mn/k^2 - \delta_ij\delta_mn]
                 - \mu(\delta_im\delta_jn+\delta_in\delta_jm) + \mu/k^2( k_i*k_n\delta_jm + 
                 k_i*k_m\delta_jn + k_j*k_n\delta_im + k_j*k_m\delta_in) - 2\mu/(1-\nu)k_ik_jk_mk_n/k^4 
        """
        """
        When shock forms, N/2 mode cann't be ignored. Using real FFT, we need to use the kernel with
        even order of k vectors. Otherwise, sigma will lose the information of N/2 mode.
        We should mostly use the source of betaP.
        """
        """
        If choosing the source 'betaE' or 'betaP', it corresponds to the fixed volume boundary condition.
        In other words, there are elastic tractions exerted on the boundary to conserve the total volume 
        of the system. During relaxation, those corrections will decay to zero for glide&climb, while they 
        will naturally evolve into isotropic hydrostatic pressure.
        If choosing the source 'rho', it corresponds to the free boundary condition. Hence, all dislocations
        only feel the self-interaction from each other. The system is allowed to change the strain freely at 
        the boundary.
        """
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
                Ksigma[i,j] *= self.ktools.kmask
            sigma = Ksigma.IFFT()
        elif source == 'betaP':
            Ksigma = Fields.SymmetricTensorField(self.gridShape, kspace=True)
            KbetaP = self.betaP.FFT()
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
                Ksigma[i,j] *= self.ktools.kmask
            sigma = Ksigma.IFFT()
            """
            Doing the following correction will be equivalent to the calculation from betaE.
            Romoving them will return to the same result calculated from rho.
            """
            lamb = 2.*mu*nu/(1.-2.*nu)
            betaP_tr = self.betaP[x,x].average()+self.betaP[y,y].average()+self.betaP[z,z].average()
            for com in sigma.components:
                i,j = com
                sigma[com] -= lamb*betaP_tr*(i==j)+mu*(self.betaP[i,j].average()+self.betaP[j,i].average())
        ### in 1D, sine and non-sine give the same answer.
        ### but are Different from the direct (old) method
        """ 
        newsigma = Fields.SymmetricTensorField(self.gridShape)
        if  self.dimension == 1:
            newsigma[x,x] = (-2.*mu/(1.-nu))*(betaP[x,x]+nu*betaP[y,y])
            newsigma[y,y] = (-2.*mu/(1.-nu))*(nu*betaP[x,x]+betaP[y,y])
            newsigma[x,y] = (-1.*mu)*(betaP[x,y]+betaP[y,x])
            for i in [x,y]:
                for j in [x,y]:
                    newsigma[i,j] -= average(newsigma[i,j])
        elif self.dimension == 2:
            KbetaP  = betaP.FFT()
            Kbetaxyyx = KbetaP[x,y] + KbetaP[y,x]
            xyFactor = ((-2./(1.-nu))/(K.kSqSq+ME))*(K.kyky*(KbetaP[x,x]+nu*KbetaP[z,z])+\
                                                        K.kxkx*(KbetaP[y,y]+nu*KbetaP[z,z])-K.kxky*Kbetaxyyx)
            zFactor  = (1./(K.kSq+ME))*(K.kx*(KbetaP[y,z] + KbetaP[z,y])-K.ky*(KbetaP[x,z] + KbetaP[z,x]))
            zzFactor = ((-2./(1.-nu))/(K.kSq+ME))*(K.kxkx*(nu*KbetaP[y,y]+KbetaP[z,z])-\
                                                      K.kxky*nu*Kbetaxyyx + K.kyky*(nu*KbetaP[x,x]+ KbetaP[z,z]))
            #Since the kspace vectors have zero points, we need to avoid those crazy valuse from 
            #(zero / zero). So we set the related grid points with zero values. Notice that, when
            #the lenght of the original real array is even, those vectors have zero points at both
            #ends. 
            xyFactor *= self.ktools.kmask
            zFactor[0,0]  = complex(0.)
            zzFactor *= self.ktools.kmask
            if (self.gridShape[0]%2) == 0:
                xyFactor[self.gridShape[0]/2,0] = complex(0.)
                zFactor[self.gridShape[0]/2,0]  = complex(0.)
                zzFactor[self.gridShape[0]/2,0] = complex(0.)
            if (self.gridShape[1]%2) == 0:
                xyFactor[0,self.gridShape[1]/2] = complex(0.)
                zFactor[0,self.gridShape[1]/2]  = complex(0.)
                zzFactor[0,self.gridShape[1]/2] = complex(0.)
            xyComp = -1.*K.kxky*xyFactor
            xzComp = K.ky*zFactor
            yzComp = -1.*K.kx*zFactor

            Knewsigma = Fields.SymmetricTensorField(self.gridShape)
            Knewsigma[x,x] = mu*K.kyky*xyFactor
            Knewsigma[x,y] = mu*xyComp
            Knewsigma[x,z] = mu*xzComp
            Knewsigma[y,z] = mu*yzComp
            Knewsigma[y,y] = mu*K.kxkx*xyFactor
            Knewsigma[z,z] = mu*zzFactor
            newsigma = Knewsigma.IFFT()
        """
        return sigma

    def CalculateRhoSymmetric(self):
        """ 
        Calculate D betaP first and the appropriate rho from equation:

        rho_ij = - epsilon_ikl d_k beta^P_lj

        This routine uses symmetric derivatives to calculate rho.
        """
        DbetaP = []
        for dim in range(len(self.gridShape)):
            DbetaPforSpecificDirection = Fields.TensorField(self.gridShape)
            for component in DbetaPforSpecificDirection.components:
                DbetaPforSpecificDirection[component] = NumericalMethods.SymmetricDerivative(self.betaP[component], self.gridShape, dim)
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

    def CalculateRhoFourier(self,type='k'):
        """ 
        Calculate D betaP first and the appropriate rho from equation:

        rho_ij = - epsilon_ikl d_k beta^P_lj

        This routine uses Fourier derivative to calculate rho.
        """
        KbetaP = self.betaP.FFT()
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
   
    def CalculateKappa(self,type='k'):
        """
        Kappa_ij = Rho_ij- 1/2Rho_kk delta_ij
        """
        rho = self.CalculateRhoFourier(type)
        traceRho = rho[x,x]+rho[y,y]+rho[z,z]
        kappa = rho
        for i in [x,y,z]:
            kappa[i,i] -= traceRho/2. 
        return kappa

    def CalculateElasticEnergy(self,type='k',source='betaE'):
        """
        E = 1/2 C strain^e * strain^e   or   E = 1/2 S sigma * sigma
        """
        sigma = self.CalculateSigma(type,source)
        energydensity = 0.
        xyz = [x,y,z]
        for i in xyz:
            for j in xyz:
                for k in xyz:
                    for m in xyz:     
                        S = (2.*(i==k)*(j==m)-(i==m)*(j==k)-self.nu*(i==j)*(k==m)/(1.+self.nu))/(2.*self.mu)
                        energydensity += 0.5*S*sigma[i,j]*sigma[k,m]
        return energydensity

    def CalculateDisplacementField(self,type='k'):
        """
        Calculate displacement field based on Yong's derivation
        """
        KbetaP = self.betaP.FFT()
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
                L_imn = -1.0j/kSq * ( nu/(1-nu) * K.k[i] * (m==n) + \
                    K.k[m]*(i==n) + K.k[n]*(i==m) -\
                    1/(1-nu)*(K.k[i]*K.k[m]*K.k[n])/kSq )
                Ku[i] += KbetaP[m,n] * L_imn
            Ku[i] *= self.ktools.kmask
        u = Ku.IFFT()  
        return u

    def CalculateBetaElastic(self,type='k'):
        """
        Calculate elastic deformation gradient(beta^E) based on Yong's
        derivation
        """
        KbetaP = self.betaP.FFT()
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
                    k[i]/kSq*((k[m]*(j==n))+(k[n]*(j==m))) -\
                    2.*(lamb+mu)/(lamb+2.*mu)*(k[i]*k[j]*k[m]*k[n])/kSqSq
                KbetaE[i,j] += KbetaP[m,n] * T_ijmn
            KbetaE[i,j] *= self.ktools.kmask
        betaE = KbetaE.IFFT()  
        betaE -= self.betaP
        return betaE

    def CalculateStrainField(self,type='k',source='betaE'):
        if source == 'betaE':
            betaE = self.CalculateBetaElastic(type)
            strain = Fields.TensorField(self.gridShape)
            for component in strain.components:
                (i,j) = component
                strain[i,j] = (betaE[i,j]+betaE[j,i])/2.
        elif source == 'sigma':
            strain = Fields.TensorField(self.gridShape)
            sigma = self.CalculateSigma(type,'betaE')
            xyz = [x,y,z]
            for i in xyz:
                for j in xyz:
                    for k in xyz:
                        for m in xyz:     
                            S = (2*(i==k)*(j==m)-(i==m)*(j==k)-self.nu*(i==j)*(k==m)/(1.+self.nu))/(2.*self.mu)
                            strain[i,j] += S*sigma[k,m]
        elif source == 'rho':
            """
            Using this source will lose the mean values.
            """
            KStrain = Fields.TensorField(self.gridShape, kspace=True)
            if type == 'sinK':
                K = self.sinktools
            elif type == 'k':
                K = self.ktools
            kSq = K.kSq+ME
            rho = self.CalculateRhoFourier(type) 
            KRho = rho.FFT() 
            nu = self.nu
            for Straincomp in KStrain.components:
                i,j = Straincomp
                for Rhocomp in KRho.components:
                    m,n = Rhocomp
                    N_ijmn = 0.
                    for l in (x,y,z):
                        N_ijmn += -(1.j)* (K.k[l]/(2.*kSq))*(perm[i,n,l]*(j==m) + perm[j,n,l]*(i==m)\
                                   -2.*perm[m,n,l]*(i==j) + 2.*perm[m,n,l]*K.k[i]*K.k[j]/((1.-nu)*kSq))
                    KStrain[i,j] += KRho[m,n] * N_ijmn
                KStrain[i,j] *= self.ktools.kmask
            strain = KStrain.IFFT()  
        return strain 

    def CalculateRotationField(self,type='k'):
        """
        Calculate the rotation field using betaE, by taking the anti-
        symmetric part
        """
        betaE = self.CalculateBetaElastic(type)
        rotE = Fields.TensorField(self.gridShape) 
        for component in rotE.components:
            i,j = component
            rotE[i,j] = (betaE[i,j]-betaE[j,i])/2.
        return rotE

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
            KbetaP = self.betaP.FFT()
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
                Krodrigues[i] *= self.ktools.kmask
            rodrigues = Krodrigues.IFFT()
            rodrigues[x] -= (self.betaP[y,z]-self.betaP[z,y])/2.  
            rodrigues[y] -= (self.betaP[z,x]-self.betaP[x,z])/2.  
            rodrigues[z] -= (self.betaP[x,y]-self.betaP[y,x])/2.  
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
                Krodrigues[i] *= self.ktools.kmask
            rodrigues = Krodrigues.IFFT()
        return rodrigues 

    def CalculateMisorientation(self,type='k',source='betaP'):
        """
        Gamma_ij = d_i Rodrigues_j

        For stress-free state, it is equivalent to Kappa.
        """
        if type == 'sinK':
            K = self.sinktools
        elif type == 'k':
            K = self.ktools
        kSq  = K.kSq + ME
        KGamma = Fields.TensorField(self.gridShape, kspace=True)
        rodrigues = self.CalculateRotationRodrigues(type,source)
        Krodrigues = rodrigues.FFT()
        for component in KGamma.components:        
            i,j = component
            KGamma[i,j] = (1.j)*K.k[j]*Krodrigues[i]
        Gamma = KGamma.IFFT()
        return Gamma


    def __mul__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(self.gridShape, field=self.GetOrderParameterField() * other.GetOrderParameterField(), inherit=self)
        else:
            return self.__class__(self.gridShape, field=self.GetOrderParameterField() * other, inherit=self)
    def __add__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(self.gridShape, field=self.GetOrderParameterField() + other.GetOrderParameterField(), inherit=self)
        else:
            return self.__class__(self.gridShape, field=self.GetOrderParameterField() + other, inherit=self)
    def __div__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(self.gridShape, field=self.GetOrderParameterField() / other.GetOrderParameterField(), inherit=self)
        else:
            return self.__class__(self.gridShape, field=self.GetOrderParameterField() / other, inherit=self)
    def __sub__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(self.gridShape, field=self.GetOrderParameterField() - other.GetOrderParameterField(), inherit=self)
        else:
            return self.__class__(self.gridShape, field=self.GetOrderParameterField() - other, inherit=self)

    def __rmul__(self, other):
        return self.__mul__(other)
    def __radd__(self, other):
        return self.__add__(other)
    def __rdiv__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(self.gridShape, field=other.GetOrderParameterField() / self.GetOrderParameterField(), inherit=self)
        else:
            return self.__class__(self.gridShape, field=other / self.GetOrderParameterField(), inherit=self)
    def __rsub__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(self.gridShape, field=other.GetOrderParameterField() - self.GetOrderParameterField(), inherit=self)
        else:
            return self.__class__(self.gridShape, field=other - self.GetOrderParameterField(), inherit=self)
