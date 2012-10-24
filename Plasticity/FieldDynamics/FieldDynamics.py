from Plasticity import NumericalMethods
from Plasticity.Fields import Fields

from Plasticity.Constants import *

class FieldDynamics:
    """
    Base class for field dynamics. Implements the equation of motion right hand side.
    """
    def __init__(self, Lambda=0):
        self.Lambda = Lambda

    def GetSigma(self,state):
        return state.CalculateSigma()

class RhoBar:
    def __init__(self, rho, Lambda=None):
        self.rho = rho
        if Lambda is not None:
            self.LambdaFactor = Lambda/3.
        else:
            self.LambdaFactor = None
        self.data = {}

    def __getitem__(self, index):
        global permutation, perm
        i, j, k = index
        """
        Moved from below

        This is a faithful implementation of the three index rho, rhobar
        """
        """
        See if it's already been calculated once
        """
        if index in self.data:
            return self.data[index]

        rhobar = 0.
        """
        Subtract rho_lkk for lambda != 0
        """
        if self.LambdaFactor is not None and self.LambdaFactor != 0. and j==k:
            for m in [x,y,z]:
                for n in [x,y,z]:
                    permutation = perm[i,m,n]
                    if permutation == 1:
                        rhobar += self.rho[n,m]
                    elif permutation == -1:
                        rhobar -= self.rho[n,m]
            rhobar *= -1.*self.LambdaFactor

        """
        rhobar_ijk = epsilon_ijl rho_lk
        """
        for l in [x,y,z]:
            permutation = perm[i,j,l]
            if permutation == 1:
                rhobar += self.rho[l,k]
            elif permutation == -1:
                rhobar -= self.rho[l,k]
            #rhobar += perm[i,j,l] * self.rho[l,k]
        self.data[index] = rhobar
        return rhobar

        """
        """
        if (i,j,k) in self.data:
            return self.data[(i,j,k)]

        s, l = permutation[i,j]
        if l != None:
            ret = s*self.rho[l,k]
        else:
            ret = 0.
        if j == k and self.LambdaFactor is not None:
            ret *= (1-self.LambdaFactor) 
            """  
            # This is when all three indices are the same
            all = [x,y,z]
            all.remove(i)
            m, n = all
            ns, nl = permutation[m,n]
            ret += self.LambdaFactor*ns*(self.rho[m,n]-self.rho[n,m])
            """
        # Straight forward implementation
        """
        rhobar_ijk = perm_ijl rho_lk
        """
        self.data[(i,j,k)] = ret
        return ret

class SymmetricDerivativeDynamics(FieldDynamics):
    """
    Field dynamics using Symmetric Derivatives for Fourier Regularization algorithms 

    Implments the equation (7) of
    S. Limkumnerd and J. Sethna, Mesoscale Theory of Grains and Cells: Crystal Plasticity 
    and Coarsening, PRL 96 095503 (2006).

    Lambda controls the mobility of climb. For Lambda=0, climb and glide have equal mobility, 
    whereas for Lambda=1 only glide is allowed.

    Initialize with Lambda=1 to subtract the trace off, i.e. prohibit climb.
    """
    def __init__(self, Lambda=0, extLoading=None, loadingRate=None, loading=0.):
        FieldDynamics.__init__(self,Lambda=Lambda)
        self.extLoading = extLoading 
        self.loadingRate = loadingRate
        self.loading = loading

    def GetSigma(self, state, time):
        sigma = state.CalculateSigma(type='sinK')
        if self.extLoading is not None:
            sigma += self.extLoading
        else:
            if self.loadingRate is not None:
                sigma += self.loading + time*self.loadingRate
        return sigma

    def CalculateFlux(self,time,state):
        sigma = self.GetSigma(state,time)
        rho   = state.CalculateRhoFourier(type='sinK')
        rhs   = Fields.TensorField(state.gridShape)
        
        rhobar = RhoBar(rho, self.Lambda)
        rhosum = rho.modulus() 

        if   state.dimension == 1:
            force = {}
            for i in [x,y,z]:
                force[i] = 0.
                for (j,k) in rhs.components:
                    force[i] += rhobar[i,j,k]*sigma[j,k]
                force[i] /= rhosum+NumericalMethods.ME
            for (i,j) in rhs.components:
                rhs[i,j] = 0.5*(force[x]*rhobar[x,i,j]+force[y]*rhobar[y,i,j]+force[z]*rhobar[z,i,j])
            """
            for (i,j) in rhs.components:
                rhs[i,j] = 0.5*((sigma[i,x]*rho[x,x]-sigma[x,x]*rho[i,x])*rho[x,j]+(sigma[i,y]*rho[x,y]-sigma[x,y]*rho[i,y])*rho[x,j]+\
                                    (sigma[i,x]*rho[y,x]-sigma[y,x]*rho[i,x])*rho[y,j]+(sigma[i,y]*rho[y,y]-sigma[y,y]*rho[i,y])*rho[y,j])
            """ 
        elif state.dimension == 2:
            """
            for (i,j) in rhs.components:
                rhs[i,j] = 0.5*((sigma[i,x]*rho[x,x]-sigma[x,x]*rho[i,x])*rho[x,j]+(sigma[i,y]*rho[x,y]-sigma[x,y]*rho[i,y])*rho[x,j]+\
                                (sigma[i,z]*rho[x,z]-sigma[x,z]*rho[i,z])*rho[x,j]+(sigma[i,x]*rho[y,x]-sigma[y,x]*rho[i,x])*rho[y,j]+\
                                (sigma[i,y]*rho[y,y]-sigma[y,y]*rho[i,y])*rho[y,j]+(sigma[i,z]*rho[y,z]-sigma[y,z]*rho[i,z])*rho[y,j]+\
                                (sigma[i,x]*rho[z,x]-sigma[z,x]*rho[i,x])*rho[z,j]+(sigma[i,y]*rho[z,y]-sigma[z,y]*rho[i,y])*rho[z,j]+\
                                (sigma[i,z]*rho[z,z]-sigma[z,z]*rho[i,z])*rho[z,j]) 
            """
            """
            Alternative: Yor's original method
            This runs much faster. (~4 times)
            """
            sigrho = Fields.TensorField(state.gridShape, rank=1)
            for j in [x,y,z]:
                sigrho[x] += sigma[y,j]*rho[z,j] - sigma[z,j]*rho[y,j]
                sigrho[y] += sigma[z,j]*rho[x,j] - sigma[x,j]*rho[z,j]
                sigrho[z] += sigma[x,j]*rho[y,j] - sigma[y,j]*rho[x,j]
            sigrhoAnt = Fields.TensorField(state.gridShape)
            sigrhoAnt[x,y] = sigrho[z]
            sigrhoAnt[x,z] = -1.*sigrho[y]
            sigrhoAnt[y,x] = -1.*sigrho[z]
            sigrhoAnt[y,z] = sigrho[x]
            sigrhoAnt[z,x] = sigrho[y]
            sigrhoAnt[z,y] = -1.*sigrho[x]
            if self.Lambda == 0:
                for component in rho.components:
                    i,j = component
                    for k in [x,y,z]:
                        rhs[i,j] += sigrhoAnt[i,k]*rho[k,j] 
            elif self.Lambda == 1:
                presure = (sigma[x,x]+sigma[y,y]+sigma[z,z])/3. 
                for component in rho.components:
                    i,j = component
                    for k in [x,y,z]:
                        rhs[i,j] += sigrhoAnt[i,k]*rho[k,j]+presure*(rho[i,k]-rho[k,i])*rho[k,j] 
                trace_rhs = (rhs[x,x]+rhs[y,y]+rhs[z,z])/3.
                for i in [x,y,z]:
                    rhs[i,i] -= trace_rhs
            rhs *= 0.5/(rhosum+NumericalMethods.ME)
        elif state.dimension == 3:
            # Not Implemented yet FIXME
            assert False
        return rhs


class GodunovDynamics(FieldDynamics):
    """
    Field dynamics for ENO/WENO Godunov algorithms

    Our dynamics equation is the Hamilton-Jacobi equation:
            u_t(x,t) + H(x,t,u,Du) = 0

    Implements the algorithm 2.1 with Godunov scheme (2.5)  of 
    Stanley Osher and Chi-Wang Shu, High-ordered essentially nonoscillatory schemes for 
    Hamilton-Jacobi equations.
    """
    def __init__(self, Lambda=0, order=5, method='ENO'):
        self.order = order
        FieldDynamics.__init__(self, Lambda=Lambda)
        if method == 'ENO':
            self.derivative = NumericalMethods.ENO_Derivative
        elif method == 'WENO':
            self.derivative = NumericalMethods.WENO_Derivative
 
    def CalculateFlux(self,time,state,CFLCondition=False):
        sigma = self.GetSigma()
        rho   = state.CalculateRhoSymmetric()
        rhs = Fields.TensorField(state.gridShape)
        ###  RHS  = -H(u,Du) = -(A*u**2 + B*u + C)  ###  
        if   state.dimension == 1:
            A = Fields.TensorField(state.gridShape, directionList=[(x), (x,y,z), (x,y,z)])
            B = Fields.TensorField(state.gridShape, directionList=[(x), (x,y,z), (x,y,z)])
            C = Fields.TensorField(state.gridShape, directionList=[(x,y,z), (x,y,z)])
            A[x,x,x] = -sigma[x,x]/2.
            B[x,x,x] = (sigma[x,y]*rho[y,y]-sigma[y,x]*rho[x,x]-sigma[y,y]*rho[x,y])/2.
            A[x,x,y] = -sigma[x,y]/2.
            B[x,x,y] = (sigma[x,x]*rho[y,x]-sigma[y,x]*rho[x,x]-sigma[y,y]*rho[x,y])/2.
            A[x,y,x] = -sigma[y,x]/2.
            B[x,y,x] = (sigma[x,x]*rho[y,x]+sigma[x,y]*rho[y,y]-sigma[y,y]*rho[x,y])/2.
            A[x,y,y] = -sigma[y,y]/2.
            B[x,y,y] = (sigma[x,x]*rho[y,x]+sigma[x,y]*rho[y,y]-sigma[y,x]*rho[x,x])/2.
            B[x,x,z] = B[x,y,z] = (sigma[x,x]*rho[y,x]+sigma[x,y]*rho[y,y]-sigma[y,x]*rho[x,x]-sigma[y,y]*rho[x,y])/2.
        elif state.dimension == 2:
            A = Fields.TensorField(state.gridShape, directionList=[(x,y), (x,y,z), (x,y,z)])
            B = Fields.TensorField(state.gridShape, directionList=[(x,y), (x,y,z), (x,y,z)])
            C = Fields.TensorField(state.gridShape)
            DbetaP = Fields.TensorField(state.gridShape, directionList=[(x,y), (x,y,z), (x,y,z)])
            for component in DbetaP.components:
                dir, i, j = component
                direction = [x,y].index(dir)
                DbetaP[component] = NumericalMethods.SymmetricDerivative(state.betaP[i,j],state.gridShape,direction)
            xSyR  = sigma[x,x]*rho[y,x]+sigma[x,y]*rho[y,y]+sigma[x,z]*rho[y,z]
            xSzR  = sigma[x,x]*rho[z,x]+sigma[x,y]*rho[z,y]+sigma[x,z]*rho[z,z]
            ySxR  = sigma[y,x]*rho[x,x]+sigma[y,y]*rho[x,y]+sigma[y,z]*rho[x,z]
            ySzR  = sigma[y,x]*rho[z,x]+sigma[y,y]*rho[z,y]+sigma[y,z]*rho[z,z]
            zSxR  = sigma[z,x]*rho[x,x]+sigma[z,y]*rho[x,y]+sigma[z,z]*rho[x,z]
            zSyR  = sigma[z,x]*rho[y,x]+sigma[z,y]*rho[y,y]+sigma[z,z]*rho[y,z]
            A[y,x,x] = -0.5*sigma[x,x]
            B[y,x,x] = 0.5*(zSxR-sigma[x,y]*rho[z,y]-sigma[x,z]*rho[z,z]+2.*sigma[x,x]*DbetaP[x,y,x])
            C[x,x]    = 0.5*((ySxR-xSyR)*rho[y,x]+(-zSxR+sigma[x,y]*rho[z,y]+sigma[x,z]*rho[z,z]-\
                                                      sigma[x,x]*DbetaP[x,y,x])*DbetaP[x,y,x])
            A[y,x,y] = -0.5*sigma[x,y]
            B[y,x,y] = 0.5*(zSxR-sigma[x,x]*rho[z,x]-sigma[x,z]*rho[z,z]+2.*sigma[x,y]*DbetaP[x,y,y])
            C[x,y]    = 0.5*((ySxR-xSyR)*rho[y,y]+(-zSxR+sigma[x,x]*rho[z,x]+sigma[x,z]*rho[z,z]-\
                                                      sigma[x,y]*DbetaP[x,y,y])*DbetaP[x,y,y])
            A[y,x,z] = -0.5*sigma[x,z]
            B[y,x,z] = 0.5*(zSxR-sigma[x,y]*rho[z,y]-sigma[x,x]*rho[z,x]+2.*sigma[x,z]*DbetaP[x,y,z])
            C[x,z]    = 0.5*((ySxR-xSyR)*rho[y,z]+(-zSxR+sigma[x,y]*rho[z,y]+sigma[x,x]*rho[z,x]-\
                                                      sigma[x,z]*DbetaP[x,y,z])*DbetaP[x,y,z])
            A[x,y,x] = -0.5*sigma[y,x]
            B[x,y,x] = 0.5*(-zSyR+sigma[y,y]*rho[z,y]+sigma[y,z]*rho[z,z]+2.*sigma[y,x]*DbetaP[y,x,x])
            C[y,x]    = 0.5*((xSyR-ySxR)*rho[x,x]+(zSyR-sigma[y,y]*rho[z,y]-sigma[y,z]*rho[z,z]-\
                                                      sigma[y,x]*DbetaP[y,x,x])*DbetaP[y,x,x])
            A[x,y,y] = -0.5*sigma[y,y]
            B[x,y,y] = 0.5*(-zSyR+sigma[y,x]*rho[z,x]+sigma[y,z]*rho[z,z]+2.*sigma[y,y]*DbetaP[y,x,y])
            C[y,y]    = 0.5*((xSyR-ySxR)*rho[x,y]+(zSyR-sigma[y,x]*rho[z,x]-sigma[y,z]*rho[z,z]-\
                                                      sigma[y,y]*DbetaP[y,x,y])*DbetaP[y,x,y])
            A[x,y,z] = -0.5*sigma[y,z]
            B[x,y,z] = 0.5*(-zSyR+sigma[y,y]*rho[z,y]+sigma[y,x]*rho[z,x]+2.*sigma[y,z]*DbetaP[y,x,z])
            C[y,z]    = 0.5*((xSyR-ySxR)*rho[x,z]+(zSyR-sigma[y,y]*rho[z,y]-sigma[y,x]*rho[z,x]-\
                                                      sigma[y,z]*DbetaP[y,x,z])*DbetaP[y,x,z])
            A[x,z,x] = -0.5*sigma[z,x]
            B[x,z,x] =  0.5*(ySzR-sigma[z,y]*rho[y,y]-sigma[z,z]*rho[y,z])
            A[y,z,x] = -0.5*sigma[z,x]
            B[y,z,x] = -0.5*(xSzR-sigma[z,y]*rho[x,y]-sigma[z,z]*rho[x,z])
            A[x,z,y] = -0.5*sigma[z,y]
            B[x,z,y] =  0.5*(ySzR-sigma[z,x]*rho[y,x]-sigma[z,z]*rho[y,z])
            A[y,z,y] = -0.5*sigma[z,y]
            B[y,z,y] = -0.5*(xSzR-sigma[z,x]*rho[x,x]-sigma[z,z]*rho[x,z])
            A[x,z,z] = -0.5*sigma[z,z]
            B[x,z,z] =  0.5*(ySzR-sigma[z,x]*rho[y,x]-sigma[z,y]*rho[y,y])
            A[y,z,z] = -0.5*sigma[z,z]
            B[y,z,z] = -0.5*(xSzR-sigma[z,x]*rho[x,x]-sigma[z,y]*rho[x,y])
        elif state.dimension == 3:
            pass
       
        forwardFlow, backwardFlow = self.CalculateDirectionalDerivatives(state) 
        rhs = self.Godunov(state,A,B,C,forwardFlow,backwardFlow)

        if self.Lambda == 1:
            Trace = rhs[x,x]+rhs[y,y]+rhs[z,z]
            for i in [x,y,z]:
                rhs[i,i] -= Trace/3.

        if CFLCondition is True:
            return rhs, self.CalculateCFLVelocity(state, A, B, C)
        return rhs

    def CalculateCFLVelocity(self, state, A, B, C):
        dimension = state.dimension
        xyz = [x,y,z]
        velocity = Fields.TensorField(state.gridShape, directionList=[xyz[:dimension], xyz, xyz]) 
        for component in velocity.components:
            dir, i, j = component
            direction = xyz.index(dir)
            DbetaP = NumericalMethods.SymmetricDerivative(state.betaP[i,j],state.gridShape,direction)
            velocity[component] = -(2.*A[component]*DbetaP + B[component])
        return velocity
 
    def CalculateDirectionalDerivatives(self, state):
        xyz = [x,y,z]
        dimension = state.dimension
        forwardFlow = Fields.TensorField(state.gridShape, directionList=[xyz[:dimension], xyz, xyz])       
        backwardFlow = Fields.TensorField(state.gridShape, directionList=[xyz[:dimension], xyz, xyz])       

        for component in forwardFlow.components:
            dir, i, j = component
            direction = xyz.index(dir) 
            forwardFlow[component] = self.derivative(state.betaP[i,j], direction, 1, self.order, dimension)*state.gridShape[direction]
            backwardFlow[component] = self.derivative(state.betaP[i,j], direction, -1, self.order, dimension)*state.gridShape[direction]

        return forwardFlow, backwardFlow 


    def Godunov(self,state,A,B,C,forwardFlow,backwardFlow):   
        dimension = state.dimension 

        xyz = [x,y,z]

        hj = Fields.TensorField(state.gridShape)
        for component in forwardFlow.components:
            dim, i, j = component
            flow_sign = (forwardFlow[component]>backwardFlow[component])
            if ((A[component]).min() == 0. ) and ((A[component]).max() == 0.):
                RangeMin,RangeMax = NumericalMethods.Extreme_linear(B[component],backwardFlow[component],forwardFlow[component])
                HJ_i = flow_sign*RangeMin + (1.-flow_sign)*RangeMax
                if dim == x:
                    hj[i,j] = -HJ_i
                else:
                    hj[i,j] -= HJ_i
            else:
                M = -B[component]/(2.*(A[component]+NumericalMethods.ME))
                InRange = ((M-forwardFlow[component])*(M-backwardFlow[component])<0)
                InRangeMin,InRangeMax,OutRangeMin,OutRangeMax = NumericalMethods.Extreme_quad(A[component],B[component],
                                                                                              backwardFlow[component],forwardFlow[component])  
                HJ_i = InRange*(flow_sign*InRangeMin+(1.-flow_sign)*InRangeMax)+\
                                     (1.-InRange)*(flow_sign*OutRangeMin+(1.-flow_sign)*OutRangeMax)
                if dim == x:
                    hj[i,j] = -HJ_i
                else:
                    hj[i,j] -= HJ_i

        if C is not None:
            hj = hj - C
        return hj
    


class UpwindDynamics(FieldDynamics):
    """
    Field dynamics for Upwind algorithms

    Implments the equation (11) and (12) of
    S. Limkumnerd and J. Sethna, Mesoscale Theory of Grains and Cells: Crystal Plasticity 
    and Coarsening, PRL 96 095503 (2006).

    with Upwind derivatives.
    """
    def __init__(self, Lambda=0, workHardening=False, scheme=None):
        self.scheme = scheme
        self.workHardening = workHardening
        FieldDynamics.__init__(self, Lambda=Lambda)

    def CalculateWorkHardeningCoefficient(self,state):
        """
        The more the dislocation lines, the stronger work hardening.
        One choice is:   D(rho) = D0/(Tr(trans(rho)*rho))^n 
                    or   D(rho) = D0/(1+(Tr(trans(rho)*rho))^n)

        Another choice from engineering community is given by:
                D = g^(-2)*(abs(T')/g)^(1/m)
        """
        if self.scheme == 'rhoDependent':
            rho = state.CalculateRhoSymmetric()
            localLengthRhoSquare = 0.
            for component in state.betaP.components:
                localLengthRhoSquare += rho[component]*rho[component]
            return localLengthRhoSquare.power(-.5) 
            """
            Plug 1.into the denominator to speed up the simulation before
            wall forming.
            """
            #return 1./(1.+localLengthRhoSquare.power(6.)) 
        elif self.scheme == 'yieldStressDependent':
            sigma = self.GetSigma(state)
            Trace = sigma[x,x]+sigma[y,y]+sigma[z,z]
            for i in [x,y,z]:
                sigma[i,i] -= Trace/3.
            strengthsigma = 0.
            for component in state.betaP.components:
                strengthsigma += sigma[component]*sigma[component]
            return strengthsigma.power(1./12)

    def CalculateFlux(self,time,state):
        dimension = state.dimension
        if dimension == 1:
            signFactor = NumericalMethods.SymmetricDerivative(state.mu*(
                            (state.betaP[x,y]+state.betaP[y,x])*(state.betaP[x,y]+state.betaP[y,x])/2.+\
                            (state.betaP[x,x]*state.betaP[x,x]+state.betaP[y,y]*\
                                (state.betaP[y,y]+(2.*state.nu)*state.betaP[x,x]))/(1.-state.nu)),state.gridShape,-1)
            DbetaP = Fields.TensorField(state.betaP.gridShape)
            for component in state.betaP.components:
                DbetaP[component] = NumericalMethods.UpwindDerivative(state.betaP[component], state.gridShape, signFactor, -1)
            """
            Original way of doing Upwind
            """
            velocity = -0.5*state.mu*((state.betaP[x,y]+state.betaP[y,x])*(DbetaP[x,y]+DbetaP[y,x])+\
                                 2.*(state.betaP[x,x]*DbetaP[x,x]+state.betaP[y,y]*DbetaP[y,y])+\
                                 (2.*state.nu/(1.-state.nu))*(state.betaP[x,x]+state.betaP[y,y])*(DbetaP[x,x]+DbetaP[y,y]))
            """
            Jim's proposed way
            velocity = -0.5*signFactor
            """

            rhs = Fields.TensorField(state.betaP.gridShape)
            if self.workHardening == True:
                D = self.CalculateWorkHardeningCoefficient(state)
                for component in state.betaP.components:
                    rhs[component] = D*velocity*DbetaP[component]
            else:
                for component in state.betaP.components:
                    rhs[component] = velocity*DbetaP[component]
            if self.Lambda == 1:
                Trace = rhs[x,x]+rhs[y,y]+rhs[z,z]
                for i in [x,y,z]:
                    rhs[i,i] -= Trace/3.
            return rhs


