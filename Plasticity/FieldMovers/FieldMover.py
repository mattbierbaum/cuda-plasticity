import sys

import NumericalMethods
from Constants import *
import GridArray
import numpy

class FieldMover:
    """Base class for field mover"""
    def Run(self,state,observers,dynamics,startTime,endTime,dT):
        assert(False)


class TVDRungeKutta_FieldMover(FieldMover):
    """TVD Runge-Kutta field mover"""
    def __init__(self, CFLsafeFactor=0.1,dtBound=0.01):
        self.CFLsafeFactor = CFLsafeFactor
        self.dtBound = dtBound

    def Run(self, state, observers, dynamics=None, startTime=0., endTime=10.,  fixedTimeStep=None):
        """Runs the simulation using the dynamics provided"""
        t = startTime
        numberOfSteps = 0
        while True:
            #state, dt = self.CalculateOneFixedTimeStep(t,state,dynamics,endTime,fixedTimeStep)
            if fixedTimeStep is not None:
                state, dt = self.CalculateOneFixedTimeStep(t,state,dynamics,endTime,fixedTimeStep)
            else:
                state, dt = self.CalculateOneAdaptiveTimeStep(t,state,dynamics,endTime)
            t += dt
            numberOfSteps += 1
            for observer in observers:
                observer.Update(t,state)
            #print "The totle number of timesteps is ",numberOfSteps, t, t>=endTime, dt
            if t >= endTime:
                break

    def CalculateOneFixedTimeStep(self,time,state,dynamics,endTime,timeStep):
        rhs = dynamics.CalculateFlux(time,state)

        alpha = [[1.],[3./4.,1./4.],[1./3.,0.,2./3.]]
        beta  = [[1.],[0.,1./4.],[0.,0.,2./3.]]

        maxset = {}
        dt = timeStep
        if time+dt > endTime:
            dt = endTime-time

        # 3rd order TVD Runge-Kutta
        L_0   = dt*rhs
        F_0   = alpha[0][0]*state + beta[0][0]*L_0
        rhs = dynamics.CalculateFlux(time, F_0)
        L_1   = dt*rhs
        F_1   = alpha[1][0]*state + beta[1][0]*L_0 + alpha[1][1]*F_0 + beta[1][1]*L_1
        rhs = dynamics.CalculateFlux(time, F_1)
        L_2   = dt*rhs
        F_2   = alpha[2][0]*state + beta[2][0]*L_0 + alpha[2][1]*F_0 + beta[2][1]*L_1 + alpha[2][2]*F_1 + beta[2][2]*L_2
        return F_2, dt


    def CalculateOneAdaptiveTimeStep(self,time,state,dynamics,endTime):
        rhs, velocity = dynamics.CalculateFlux(time,state,CFLCondition=True)

        alpha = [[1.],[3./4.,1./4.],[1./3.,0.,2./3.]]
        beta  = [[1.],[0.,1./4.],[0.,0.,2./3.]]

        maxset = {}
        # calculate dt with CFL condition
        """
        for component in velocity.components:
            dir, i, j = component
            componentMax = (velocity[component]).fabs().max()
            if dir in maxset:
                maxset[dir] = max([maxset[dir], componentMax])
            else:
                maxset[dir] = componentMax
        sum = 0.
        for dir in maxset:
            direction = ['x','y','z'].index(dir)
            sum += state.gridShape[direction] * maxset[dir]
        #calculate dt with another method#
        for component in velocity.components:
            dir, i, j = component
            componentMax = (velocity[component]).fabs().max()
            if dir in maxset:
                maxset[dir] = max([maxset[dir], componentMax])
            else:
                maxset[dir] = componentMax
        sum = 0.
        for dir in maxset:
            direction = ['x','y','z'].index(dir)
            sum = (state.gridShape[direction] * maxset[dir]).max()
        """
        sum = max(state.gridShape)*velocity.max()

        dt = self.CFLsafeFactor / sum
        """
        set up maximum bound for time step.
        Otherwise, energy may go uphill sometime.
        This maximum bound is nearly independent of system size.
        For GlideOnly, 0.03 seems OK, while for GlideClimb, 0.01
        seems OK. 
        """
        #if (dt > self.dtBound) and (self.dtBound is not None):
        #    dt = self.dtBound 
        #print "Max time step: ", dt, sum
        if time+dt > endTime:
            dt = endTime-time

        # 3rd order TVD Runge-Kutta
        
        L_0   = dt*rhs
        F_0   = alpha[0][0]*state + beta[0][0]*L_0
        F_2   = alpha[2][0]*state + beta[2][0]*L_0 + alpha[2][1]*F_0

        del rhs
        #print "\tDEL0 ",
        #GridArray.print_mem_usage()

        rhs = dynamics.CalculateFlux(time, F_0)
        L_1   = dt*rhs
        F_1   = alpha[1][0]*state + beta[1][0]*L_0 + alpha[1][1]*F_0 + beta[1][1]*L_1
        F_2  = F_2 + beta[2][1]*L_1 + alpha[2][2]*F_1

        del rhs, L_1, F_0, L_0
        #print "\tDEL1 ",
        #GridArray.print_mem_usage()


        rhs = dynamics.CalculateFlux(time, F_1)
        L_2   = dt*rhs
        F_2   = F_2 + beta[2][2]*L_2

        del rhs, L_2, F_1
        #print "\tDEL2 ",
        #GridArray.print_mem_usage()
       
        return F_2, dt



class RungeKutta_FieldMover(FieldMover):
    """
    4th-order Runge-Kutta field mover

    Run method is derived from odeint of Numerical Recipes Runge-Kutta solver, but is modified in
    several ways.
    """
    def __init__(self, eps=1.E-6, minimumTimeStep=0., initialTimeStep=1.E-4, PIStepsizeControl=False):
        self.eps = eps
        self.minimumTimeStep = minimumTimeStep
        self.initialTimeStep = initialTimeStep
        self.PIStepsizeControl = PIStepsizeControl
        if self.PIStepsizeControl == True:
            self.pre_errmax = 1.

    def Run(self, state, observers, dynamics, startTime=0., endTime=10., fixedTimeStep=None):
        """
        Runs the simulation. 

        By default, this runs adaptive time step integrator. To run fixed time steps,
        set fixedTimeStep to desired time step value.
        """
        t = startTime
        if fixedTimeStep == None:
            """Using adaptive time step."""
            numberOfGoodSteps, numberOfBadSteps = 0 , 0
            dt = self.initialTimeStep
            while True:
                flux = dynamics.CalculateFlux(t, state)
                scale = (state.GetOrderParameterField()).fabs()+(flux*dt).fabs()+NumericalMethods.ME
                #scale = (state.GetOrderParameterField()).fabs().max()+(flux*dt).fabs()+NumericalMethods.ME
                #field = state.GetOrderParameterField()
                #scale = field.fabs()+(flux*dt).fabs()+field.max()*NumericalMethods.ME*1.0e6
                if t+dt > endTime:
                    dt = endTime-t
                performedTimeStep, nextTimeStep, t, state = self.CalculateOneAdaptiveTimeStep(state,flux,t,dt,scale,dynamics.CalculateFlux)
                if performedTimeStep == dt:
                    numberOfGoodSteps += 1
                else:
                    numberOfBadSteps += 1
                if (abs(nextTimeStep) <= self.minimumTimeStep):
                    print "Step size too small in this RungeKuttaFieldMover."
                    print "...now exiting to system..."
                    sys.exit(1)
                dt = nextTimeStep
                for observer in observers:
                    observer.Update(t,state)
                if t >= endTime:
                    return numberOfGoodSteps, numberOfBadSteps
        else:
            """Using fixed time step."""
            dt = fixedTimeStep
            numberOfSteps = 0
            while True:
                if t+dt > endTime:
                    dt = endTime-t
                flux = dynamics.CalculateFlux(None, state)
                state = self.CalculateOneFixedTimeStep(state,flux,t,dt,dynamics.CalculateFlux)
                t += dt
                numberOfSteps += 1
                for observer in observers:
                    observer.Update(t,state)
                if t >= endTime:
                    return numberOfSteps

    def CalculateOneAdaptiveTimeStep(self,state,flux,t,dt,scale,dynamics):
        if self.PIStepsizeControl == True:
            performedTimeStep, nextTimeStep, time, state, errmax = NumericalMethods.rkqs(state,flux,t,dt,self.eps,scale,dynamics,
                                                 self.PIStepsizeControl,self.pre_errmax)
            self.pre_errmax = errmax
            return performedTimeStep, nextTimeStep, time, state
        else:
            return NumericalMethods.rkqs(state,flux,t,dt,self.eps,scale,dynamics)

    def CalculateOneFixedTimeStep(self,state,flux,t,dt,dynamics):
        newState, error = NumericalMethods.rkck(state,flux,t,dt,dynamics)
        return newState

  
class LaxFriedrichs_RK_FieldMover(RungeKutta_FieldMover):
    """
    This works for conservation laws: U_t + f(U)_x = 0
    Lax-Friedrichs mover: U(n+1,j) = 0.5*(U(n,j-1)+U(n,j+1))-dt*(f(U(n,j+1))-f(U(n,j-1)))/(2.*dx)
    For general nonlinear pde, it looks like: U_t + F(U)=0
    So the mvoer is rewritten as: U(n+1,j) = 0.5*(U(n,j-1)+U(n,j+1))-dt*rhs(U)
    It may not be suitable for general non-linear pde.
    """
    def CalculateOneAdaptiveTimeStep(self,state,flux,t,dt,scale,dynamics):
        correction = self.SymmetricCorrection(state)
        if self.PIStepsizeControl == True:
            performedTimeStep, nextTimeStep, time, newstate, errmax = NumericalMethods.rkqs(state,flux,t,dt,self.eps,scale,dynamics,
                                                    self.PIStepsizeControl,self.pre_errmax)
            self.pre_errmax = errmax
            return performedTimeStep, nextTimeStep, time, newstate+correction
        else:
            performedTimeStep, nextTimeStep, time, newstate = NumericalMethods.rkqs(state,flux,t,dt,self.eps,scale,dynamics)
            return performedTimeStep, nextTimeStep, time, newstate+correction

    def CalculateOneFixedTimeStep(self,state,flux,t,dt,dynamics):
        correction = self.SymmetricCorrection(state)
        newState, error = NumericalMethods.rkck(state,flux,t,dt,dynamics)
        return newState+correction

    def SymmetricCorrection(self,state):
        newstate = 1.*state
        field = state.GetOrderParameterField()
        newfield = newstate.GetOrderParameterField()
        for comp in field.components:
            if state.dimension == 1:
                newfield[comp] = field[comp].roll(-1,0)+field[comp].roll(1,0)
            elif field.dimension == 2:
                newfield[comp] = field[comp].roll(-1,0)+field[comp].roll(1,0)+\
                                 field[comp].roll(-1,1)+field[comp].roll(1,1)
            elif field.dimension == 3:
                newfield[comp] = field[comp].roll(-1,0)+field[comp].roll(1,0)+\
                                 field[comp].roll(-1,1)+field[comp].roll(1,1)+\
                                 field[comp].roll(-1,2)+field[comp].roll(1,2)
        newfield *= 0.5
        newstate.UpdateOrderParameterField(newfield)
        return newstate


class OperatorSplittingFourierRegularizationRK_FieldMover(RungeKutta_FieldMover):
    """
    2nd/4th-order Runge-Kutta field mover with Operator Splitting Method Fourier Regularization.
    """
    def __init__(self, diffusion=2e-4, order=2, eps=1.E-4, minimumTimeStep=0., initialTimeStep=1.E-4, PIStepsizeControl=False):
        self.order = order
        self.diffusion = diffusion
        RungeKutta_FieldMover.__init__(self, eps, minimumTimeStep, initialTimeStep, PIStepsizeControl)

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
        """
        This fourier regularization scheme gives the artificial viscosity solution.
        When the diffusion constant goes to zero, the solution theoretically converges
        to the unique weak one of the original PDE.

        Michael G. Crandall and Pierre-Louis Lions
        Viscosity solutions of Hamilton-Jacobi equations, 1983 
        """
        KorderParameter = state.GetOrderParameterField().FFT()
        if state.dimension == 1:
            kSq = state.ktools.kSq
            kSqSq = state.ktools.kSqSq
        elif state.dimension == 2:
            kSq = state.ktools.kSq
            kSqSq = state.ktools.kSqSq
        elif state.dimension == 3:
            pass
        if self.order == 2:
            KorderParameter *= (-self.diffusion * timeStep * kSq).exp() 
        elif self.order == 4: 
            KorderParameter *= (-self.diffusion * timeStep * kSqSq).exp() 
        state.UpdateOrderParameterField(KorderParameter.IFFT())
 

class OperatorSplittingTVDRK_FieldMover(TVDRungeKutta_FieldMover):
    """
    TVD Runge-Kutta field mover with Operator Splitting Method Fourier Regularization.
    
    Gamma is the diffusion constant for the equation of motion of vacancies.
    """
    def CalculateOneFixedTimeStep(self,time,state,dynamics,endTime,timeStep):
        prevTrBetaP = state.betaP_V[x,x]+state.betaP_V[y,y]+state.betaP_V[z,z]
        betaPstate, cfield = state.DecoupleState() 
        rhs = dynamics.CalculateFlux(time,betaPstate,cfield)
        alpha = numpy.array([[1.],[3./4.,1./4.],[1./3.,0.,2./3.]]).astype('float64')
        beta  = numpy.array([[1.],[0.,1./4.],[0.,0.,2./3.]]).astype('float64')
        maxset = {}
        dt = timeStep
        if time+dt > endTime:
            dt = endTime-time
        # 3rd order TVD Runge-Kutta
        L_0   = dt*rhs
        F_0   = alpha[0][0]*betaPstate + beta[0][0]*L_0
        rhs = dynamics.CalculateFlux(time, F_0, cfield)
        L_1   = dt*rhs
        F_1   = alpha[1][0]*betaPstate + beta[1][0]*L_0 + alpha[1][1]*F_0 + beta[1][1]*L_1
        rhs = dynamics.CalculateFlux(time, F_1, cfield)
        L_2   = dt*rhs
        F_2   = alpha[2][0]*betaPstate + beta[2][0]*L_0 + alpha[2][1]*F_0 + beta[2][1]*L_1 + alpha[2][2]*F_1 + beta[2][2]*L_2

        TrBetaP = F_2.betaP[x,x]+F_2.betaP[y,y]+F_2.betaP[z,z]
        newc = dynamics.CalculateVFlux(cfield,TrBetaP-prevTrBetaP,dt,state.ktools)

        state.RecoupleState(betaPstate,newc)#F_2,newc)
        return state, dt

    def CalculateOneAdaptiveTimeStep(self,time,state,dynamics,endTime):
        prevTrBetaP = state.betaP_V[x,x]+state.betaP_V[y,y]+state.betaP_V[z,z]
        betaPstate, cfield = state.DecoupleState() 
        rhs, velocity = dynamics.CalculateFlux(time,betaPstate,cfield,CFLCondition=True)
        alpha = numpy.array([[1.,0,0],[3./4.,1./4.,0],[1./3.,0.,2./3.]]).astype('float64')
        beta  = numpy.array([[1.,0,0],[0.,1./4.,0],[0.,0.,2./3.]]).astype('float64')
        maxset = {}
        sum = max(state.gridShape)*velocity.max()
        dt = self.CFLsafeFactor / sum
        if (dt > self.dtBound) and (self.dtBound is not None):
            dt = self.dtBound 
        #print "Max time step: ", dt, sum
        if time+dt > endTime:
            dt = endTime-time
        # 3rd order TVD Runge-Kutta
        L_0   = dt*rhs
        F_0   = alpha[0][0]*betaPstate+ beta[0][0]*L_0

        rhs = dynamics.CalculateFlux(time, F_0, cfield)
        L_1   = dt*rhs
        F_1   = alpha[1][0]*betaPstate + beta[1][0]*L_0 + alpha[1][1]*F_0 + beta[1][1]*L_1

        rhs = dynamics.CalculateFlux(time, F_1, cfield)
        L_2   = dt*rhs
        F_2   = alpha[2][0]*betaPstate + beta[2][0]*L_0 + alpha[2][1]*F_0 + beta[2][1]*L_1 + alpha[2][2]*F_1 + beta[2][2]*L_2

        TrBetaP = F_2.betaP[x,x]+F_2.betaP[y,y]+F_2.betaP[z,z]
        newc = dynamics.CalculateVFlux(cfield,TrBetaP-prevTrBetaP,dt,state.ktools)

        state.RecoupleState(F_2,newc)
        return state, dt
        
