from CentralUpwind import *
from NumericalMethods import *
from Constants import *

import Fields
import FieldDynamics

class RhoDynamics(CentralUpwindDynamics):
    def __init__(self, Dx=None, Lambda=0, coreEnergy=0, coreEnergyLog=0):
        self.Dx = Dx
        self.Lambda = Lambda
        self.coreEnergy = coreEnergy
        self.coreEnergyLog = coreEnergyLog
        self.counter = 0
        """
        used for previous time step
        """
        self.prevTimeStep = False
        self.time = None
        self.vtime = None
        #self.velocity = {}
        self.velocity = None

    def RhoVelocity(self,rho,dir=None,opt=None,sig=None):
        sigma = self.sigma
        rhobar = FieldDynamics.RhoBar(rho, self.Lambda)
        #force = {}
        force = Fields.TensorField(rho.gridShape, components=xyz)
        if sig is not None:
            sigma = sig
        elif opt is not None and dir is not None:
            if opt == '+':
                pm = -1
            elif opt == '-':
                pm = 1
            else:
                assert False
            sigma = (rollfield(sigma, pm, dir) + sigma)*0.5
            
        for i in [x,y,z]:
            force[i] = 0.
            for (j,k) in rho.components:
                force[i] += rhobar[i,j,k]*sigma[j,k]

        # Normalize with total dislocation density
        rhosum = rho.modulus()

        # FIXME - this has to go with the brute force decimation
        #rhosum = sumoddeven(rhosum)
        for i in [x,y,z]:
            force[i] /= rhosum
        
        return force
        
    def F_1D(self,field):
        velocity = self.RhoVelocity(field)
        ret = field*velocity[z]
        for (i,j) in ret.components:
            ret[i,j] -= field[z,j]*velocity[i]
        return ret

    def F_1Dprime(self,field):
        velocity = self.RhoVelocity(field)
        return velocity[z].fabs().max()

    def F_2D(self,field,dir=0,opt=None,sig=None):
        comp = xyz[dir]
        velocity = self.RhoVelocity(field,dir=dir,opt=opt,sig=sig)
        """
        ret = field*velocity[comp]
        for (i,j) in ret.components:
            ret[i,j] -= field[comp,j]*velocity[i]
        """
        ret = Fields.TensorField(field.gridShape, field.components)
        rhobar = FieldDynamics.RhoBar(field, self.Lambda)
        for (i,j) in ret.components:
            for m in xyz:
                for k in xyz: 
                    ret[i,j] += perm[i,comp,m]*rhobar[k,m,j]*velocity[k]
        return ret

    def F_2Dprime(self,field,dir=0,opt=None,sig=None):
        velocity = self.RhoVelocity(field,dir=dir,opt=opt,sig=sig)
        return velocity[xyz[dir]].fabs()

    def CalculateFlux(self, time, state, CFLCondition=False):
        self.state = state
        #self.sigma = self.GetSigma(state,time)
        self.sigma = state.CalculateSigma()
        return CentralUpwindDynamics.CalculateFlux(self,time,state,CFLCondition=CFLCondition) 

    def CalculateFluxOSD(self, time, state, CFLCondition=False, dim=0):
        self.state = state
        self.sigma = state.CalculateSigma()
        return CentralUpwindDynamics.CalculateFluxOSD(self,time,state,CFLCondition=CFLCondition,dim=dim) 

    def Q_2D(self,field,deriv_x,deriv_y,dir=0):
        comp = xyz[dir]
        ret = Fields.TensorField(field.gridShape, field.components)
        for j in xyz:
            ret[comp,j] = deriv_x[x,j]+deriv_y[y,j] 
        return ret

