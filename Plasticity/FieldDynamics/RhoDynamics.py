import NumericalMethods
import Fields
import FieldDynamics

from Constants import *


class SymmetricDerivativeDynamics(FieldDynamics.FieldDynamics):
    """
    Field dynamics using Symmetric Derivatives for Fourier Regularization algorithms 

    Implments the equation (7) of
    S. Limkumnerd and J. Sethna, Mesoscale Theory of Grains and Cells: Crystal Plasticity 
    and Coarsening, PRL 96 095503 (2006).

    Lambda controls the mobility of climb. For Lambda=0, climb and glide have equal mobility, 
    whereas for Lambda=1 only glide is allowed.

    Initialize with Lambda=1 to subtract the trace off, i.e. prohibit climb.
    """
    def GetSigma(self,state, time=None):
        return state.CalculateSigma()

    def CalculateForce(self,time,state):
        sigma = self.GetSigma(state, time=time)
        rho   = state.CalculateRhoSymmetric()
        rhs   = Fields.TensorField(state.gridShape)
      
        rhobar = FieldDynamics.RhoBar(rho, self.Lambda)
        force = {}
        for i in [x,y,z]:
            force[i] = 0.
            for (j,k) in rhs.components:
                force[i] += rhobar[i,j,k]*sigma[j,k]

        # Normalize with total dislocation density
        rhosum = rho.modulus() 
        for i in [x,y,z]:
            force[i] /= rhosum
        return force

    def CalculateFlux(self,time,state):
        sigma = self.GetSigma(state, time=time)
        rho   = state.CalculateRhoSymmetric()
        rhs   = Fields.TensorField(state.gridShape)
       
        rhobar = FieldDynamics.RhoBar(rho, self.Lambda)
        force = {}
        for i in [x,y,z]:
            force[i] = 0.
            for (j,k) in rhs.components:
                force[i] += rhobar[i,j,k]*sigma[j,k]

        # Normalize with total dislocation density
        rhosum = rho.modulus() 
        for i in [x,y,z]:
            force[i] /= rhosum
 
        rhobar = FieldDynamics.RhoBar(rho, self.Lambda)
        if   state.dimension == 1:
            for (i,j) in rhs.components:
                term = 0.5*(force[i]*rho[z,j]-force[z]*rho[i,j]) 
                rhs[i,j] += NumericalMethods.SymmetricDerivative(term, state.gridShape, 0)
        elif state.dimension == 2:
            for (i,j) in rhs.components:
                if i == x:
                    rhs[i,j] = -0.5*NumericalMethods.SymmetricDerivative(force[x]*rhobar[x,z,j]+force[y]*rhobar[y,z,j]+force[z]*rhobar[z,z,j], state.gridShape, 1)
                elif i == y:
                    rhs[i,j] = 0.5*NumericalMethods.SymmetricDerivative(force[x]*rhobar[x,z,j]+force[y]*rhobar[y,z,j]+force[z]*rhobar[z,z,j], state.gridShape, 0)
                else:
                    rhs[i,j] = 0.5*(NumericalMethods.SymmetricDerivative(force[x]*rhobar[x,x,j]+force[y]*rhobar[y,x,j]+force[z]*rhobar[z,x,j], state.gridShape, 1) \
                                 - NumericalMethods.SymmetricDerivative(force[x]*rhobar[x,y,j]+force[y]*rhobar[y,y,j]+force[z]*rhobar[z,y,j], state.gridShape, 0))          

        elif state.dimension == 3:
            # Not Implemented yet FIXME
            assert False
        return rhs

class SymmetricDerivativeDynamicsWithExternalStress(SymmetricDerivativeDynamics):
    def __init__(self, Lambda=0, Stress_component=None, Stress_value=0.):
        self.stress_component = Stress_component
        self.stress_value = Stress_value
        SymmetricDerivativeDynamics.__init__(self, Lambda=Lambda)

    def GetSigma(self, state, time=0.00):
        sigma = state.CalculateSigma()
        if self.stress_component is not None:
            sigma[self.stress_component] += self.stress_value
        return sigma

class SymmetricDerivativeDynamicsWithLineEnergy(SymmetricDerivativeDynamics):
    def __init__(self, Lambda=0, coreEnergy=0., coreEnergyLog = 0.):
        self.coreEnergy = coreEnergy
        self.coreEnergyLog = coreEnergyLog
        SymmetricDerivativeDynamics.__init__(self, Lambda=Lambda)

    def GetLineEnergySigma(self, state, rho, rhosum):
        """
        This term is added to stress
        \epsilon_lki \partial_k 
            (\lambda_1 - \lambda_2 (1+log\sqrt{Tr rho^T rho}))
            rho_ij / \sqrt{Tr rho^T rho}
        """
        term = rho * (self.coreEnergy - self.coreEnergyLog*(1.0+rhosum.log())) / rhosum
        return term

    def GetSigma(self, state, time=0.00):
        sigma = state.CalculateSigma()
        rho   = state.CalculateRhoSymmetric()

        """
        The sigma returned by this function is not a symmetric tensor
        """
        newsigma = Fields.TensorField(state.gridShape)
        """
        for component in newsigma.components:
            newsigma[component] = sigma[component]
        """

        """
        Calculate \sqrt{ Tr rho^T rho }
        """
        rhosum = rho.modulus()
        """
        term = self.GetLineEnergySigma(state, rho, rhosum)
 
        Kterm = term.FFT()
        deriv_term = {}
        K = state.ktools
        for k in xyz:
            deriv_term[k] = Kterm*1.0j*K.k[k]
       
        Ksigma = newsigma.FFT() 
        for component in Ksigma.components:
            i, j = component
            for l in xyz:
                for k in xyz:
                    Ksigma[component] += perm[i,k,l]*deriv_term[k][l,j]

        # FIXME - this is an adhoc statement restricting the dynamics of
        # these terms to only where there are significant amount of dislocations 
        newsigma = Ksigma.IFFT()

        import scipy.stats   
        mean = rhosum.mean() 
        newsigma *= 0.5*scipy.stats.erfc((mean/rhosum).log())

        for component in newsigma.components:
            newsigma[component] += sigma[component]

        return newsigma
        """
        #term = rho * (self.coreEnergy - self.coreEnergyLog*(1.0+rhosum.log()))
        term = self.GetLineEnergySigma(state, rho, rhosum)*rhosum
 
        dimension = len(state.gridShape)
        if dimension == 1:
            dim = [z]
        elif dimension == 2:
            dim = [x,y]
        elif dimension == 3:
            dim = [x,y,z]

        deriv_rho = {}
        for k in xyz:
            deriv_rho[k] = 0.
            #deriv_rho[k] = Fields.TensorField(state.gridShape)
            if k in dim:
                for m in xyz:
                    for n in xyz:
                        deriv_rho[k] += rho[m,n]*NumericalMethods.SymmetricDerivative(rho[m,n], state.gridShape, dim.index(k))
            deriv_rho[k] /= rhosum**2
        
        deriv_term = {}
        for k in xyz:
            deriv_term[k] = Fields.TensorField(state.gridShape)
            for l in xyz:
                for j in xyz:
                    if k in dim:
                        deriv_term[k][l,j] = NumericalMethods.SymmetricDerivative(term[l,j], state.gridShape, dim.index(k))
        
        for component in newsigma.components:
            i,j = component
            for l in xyz:
                for k in xyz:
                    if k in dim:
                        permutation = perm[i,k,l]
                        #newsigma[component] += perm[i,k,l]*(deriv_term[k][l,j]-term[l,j]*deriv_rho[k])
                        if permutation == 1:
                        
                            newsigma[component] += (deriv_term[k][l,j]-term[l,j]*deriv_rho[k])
                        elif permutation == -1:
                        
                            newsigma[component] += (term[l,j]*deriv_rho[k]-deriv_term[k][l,j])
            newsigma[component] /= rhosum
      
        # FIXME - this is an adhoc statement restricting the dynamics of
        # these terms to only where there are significant amount of dislocations 
        #import scipy.stats   
        #mean = rhosum.mean() 
        #newsigma *= 0.5*scipy.stats.erfc((mean/rhosum).log())

        for component in newsigma.components:
            newsigma[component] += sigma[component]
        return newsigma

class SymmetricDerivativeDynamicsWithLineEnergyWithFR(SymmetricDerivativeDynamicsWithLineEnergy):
    def __init__(self, Lambda=0, coreEnergy=0., coreEnergyLog = 0., diffusion=0.):
        self.diffusion = diffusion
        SymmetricDerivativeDynamicsWithLineEnergy.__init__(self, Lambda=Lambda, coreEnergy=coreEnergy, coreEnergyLog=coreEnergyLog)

    def SetDiffusionConstant(self, gridShape, rhosum, n=5):
        rhomax = rhosum.max()
        self.diffusion = n**2 * (self.coreEnergyLog * rhomax.log())/(rhomax)/gridShape[-1]**2
        
    def GetLineEnergySigma(self, state, rho, rhosum):
        """
        Additional term for proper diffusion
        """
        term = SymmetricDerivativeDynamicsWithLineEnergy.GetLineEnergySigma(self, state, rho, rhosum)

        K = state.sinktools      

        Krho = rho.FFT()
        Kadd_term = Krho * K.kSq * self.diffusion
        add_term = Kadd_term.IFFT()

        term += add_term
        return term
