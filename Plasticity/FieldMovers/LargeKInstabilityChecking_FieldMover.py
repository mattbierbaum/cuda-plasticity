import FieldMover
import NumericalMethods
import FieldInitializer

import sys

class LargeKInstabilityCheckingRK_FieldMover(FieldMover.RungeKutta_FieldMover):
    """
    """
    def __init__(self, initialChecking=None, filename=None, eps=1.E-4, minimumTimeStep=0., initialTimeStep=1.E-4, PIStepsizeControl=False):
        self.initialChecking = initialChecking 
        self.filename = filename
        self.stop = False
        FieldMover.RungeKutta_FieldMover.__init__(self, eps, minimumTimeStep, initialTimeStep, PIStepsizeControl)

    def Run(self, state, observers, dynamics, startTime=0., endTime=10., fixedTimeStep=None):
        t = startTime
        if fixedTimeStep == None:
            """Using adaptive time step."""
            numberOfGoodSteps, numberOfBadSteps = 0 , 0
            dt = self.initialTimeStep
            while True:
                flux = dynamics.CalculateFlux(t, state)
                scale = state.GetOrderParameterField().fabs()+(flux*dt).fabs()+NumericalMethods.ME
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
                if (t >= endTime) or (self.stop==True):
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
                if (t >= endTime) or (self.stop==True):
                    return numberOfSteps
 
    def CalculateOneAdaptiveTimeStep(self,state,flux,t,dt,scale,dynamics):
        if self.PIStepsizeControl == True:
            performedTimeStep, nextTimeStep, time, newstate, errmax = NumericalMethods.rkqs(state,flux,t,dt,self.eps,scale,dynamics,
										            self.PIStepsizeControl,self.pre_errmax)
            self.pre_errmax = errmax
            if self.CheckInstability(newstate) == False:
                FieldInitializer.SaveNumpyFileFromState(newstate,self.filename)  
                self.stop = True 
            return performedTimeStep, nextTimeStep, time, newstate
        else:
            performedTimeStep, nextTimeStep, time, newstate = NumericalMethods.rkqs(state,flux,t,dt,self.eps,scale,dynamics)
            if self.CheckInstability(newstate) == False:
                FieldInitializer.SaveNumpyFileFromState(newstate,self.filename)  
                self.stop = True 
            return performedTimeStep, nextTimeStep, time, newstate

    def CheckInstability(self,state,limit=1.e5):
        """
        This limit is good for smooth initial condtion with large correlation length.
        The smaller correlation length, the smaller limit.
        """
        rhoM = state.CalculateRhoFourier().modulus()
        KrhoM = rhoM.fftn()
        if state.dimension == 1:
            N = state.gridShape[0]
            Instability = (KrhoM[N/2].real**2+KrhoM[N/2].imag**2).sqrt()
        elif state.dimension == 2:
            Nx,Ny = state.gridShape
            Instability = (KrhoM[Nx/2,Ny/2].real**2+KrhoM[Nx/2,Ny/2].imag**2).sqrt()
        elif state.dimension == 3:
            Nx,Ny,Nz = state.gridShape
            Instability = (KrhoM[Nx/2,Ny/2,Nz/2].real**2+KrhoM[Nx/2,Ny/2,Nz/2].imag**2).sqrt()
        return ((Instability/self.initialChecking)<limit)
        


