from numpy import  fromfunction, power, array

from Plasticity.FieldMovers import FieldMover
from Plasticity import NumericalMethods

ME = NumericalMethods.ME


class OperatorSplittingSpectralFilterRK_FieldMover(FieldMover.OperatorSplittingFourierRegularizationRK_FieldMover):
    def __init__(self, diffusion=2e-4, order=2, eps=1.E-4, minimumTimeStep=0., initialTimeStep=1.E-4, PIStepsizeControl=False, 
                 filter=False, filterParameters=None):
        self.filter = filter
        self.filterParameters = filterParameters
        FieldMover.OperatorSplittingSpectralViscosityRK_FieldMover(self, diffusion=diffusion, order=order, eps=eps, minimumTimeStep=minimumTimeStep,
                                                                initialTimeStep=initialTimeStep, PIStepsizeControl=PIStepsizeControl)

    def FourierRegularize(self, state, timeStep):
        if self.filter == True:
            KorderParameter = self.Filtering(state)
            state.UpdateOrderParameterField(KorderParameter.IFFT())
        else:
            FieldMover.OperatorSplittingFourierRegularizationRK_FieldMover.FourierRegularize(self, state, timeStep)
 
    def Filtering(self,state):
        KorderParameter = state.GetOrderParameterField().FFT()
        alpha, C = self.filterParameters
        dim = state.dimension
        for item in KorderParameter.components:
            if dim == 1:
                M = state.gridShape[0]/2+1
                temp = fromfunction(lambda i: -alpha*((i/float(M)-C)/(1.-C))**2*(i>int(C*M)), [M,])
                KorderParameter[item] *= temp.exp() 
            elif dim == 2:
                pass 
            elif dim == 3:
                pass  
        return KorderParameter

class OperatorSplittingSpectralViscosityRK_FieldMover(FieldMover.RungeKutta_FieldMover):
    """
    2nd/4th-order Runge-Kutta field mover with Operator Splitting Method Spectral Viscosity.
    """
    def __init__(self, diffusion=2e-4, order=2, eps=1.E-4, minimumTimeStep=0., initialTimeStep=1.E-4, PIStepsizeControl=False): 
        self.order = order
        self.diffusion = diffusion
        FieldMover.RungeKutta_FieldMover.__init__(self, eps, minimumTimeStep, initialTimeStep, PIStepsizeControl)

    def CalculateOneAdaptiveTimeStep(self,state,flux,t,dt,scale,dynamics):
        if self.PIStepsizeControl == True:
            performedTimeStep, nextTimeStep, time, newstate, errmax = NumericalMethods.rkqs(state,flux,t,dt,self.eps,scale,dynamics,
										            self.PIStepsizeControl,self.pre_errmax)
            self.pre_errmax = errmax
            self.FourierRegularize(newstate, performedTimeStep)
            return performedTimeStep, nextTimeStep, time, newstate
        else:
            performedTimeStep, nextTimeStep, time, newstate = NumericalMethods.rkqs(state,flux,t,dt,self.eps,scale,dynamics)
            self.FourierRegularize(newstate, performedTimeStep)
            return performedTimeStep, nextTimeStep, time, newstate

    def CalculateOneFixedTimeStep(self,state,flux,t,dt,dynamics):
        newState, error = NumericalMethods.rkck(state,flux,t,dt,dynamics)
        self.FourierRegularize(newState, dt)
        return newState
   
    def FourierRegularize(self, state, timeStep):
        KorderParameter = state.GetOrderParameterField().FFT()
        KorderParameter *= (-self.diffusion * timeStep * self.Filtering(state)).exp() 
        state.UpdateOrderParameterField(KorderParameter.IFFT())
        
    def Filtering(self,state):
        dim = state.dimension
        theta = (self.order-1.)/self.order
        N = array(state.gridShape)
        m = 0.1*power(N,theta)/power(log(N),dim/2)
        if dim == 1:
            Q = fromfunction(lambda i: (1.-power(m[0]/(i+ME),(self.order-1.)/theta))*(i>m[0]), [N[0]/2+1,])            
        elif dim == 2:
            pass
        elif dim == 3:
            pass  
        if self.order == 2:
            return state.ktools.kSq*Q
        elif self.order == 4:
            return state.ktools.kSqSq*Q
