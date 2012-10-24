from Plasticity.FieldDynamics.CentralUpwindHJ import *
from Plasticity.Constants import *

from Plasticity import NumericalMethods
from Plasticity.Fields import Fields

"""
Define BetaP H-J type dynamics
"""
class BetaPDynamics(CentralUpwindHJDynamics):
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
        self.velocity = {}

    def BetaP1D_RhoMod(self,u,ux):
        rho2 = NumericalMethods.ME
        for component in ux.components:
            i,j = component
            if i == z:
                continue
            rho2 += ux[component]*ux[component] 
        return rho2.sqrt()

    def BetaP2D_RhoMod(self,u,ux,uy):
        rho2 = NumericalMethods.ME
        for component in ux.components:
            rho2 += ux[component]*ux[component] 
            rho2 += uy[component]*uy[component] 
        for j in [x,y,z]:
            rho2 -= ux[x,j]*ux[x,j]
            rho2 -= uy[y,j]*uy[y,j]
            rho2 -= 2*ux[y,j]*uy[x,j]
        return rho2.sqrt()

    def BetaP1D_Velocity(self,u,ux):
        velocity = Fields.TensorField(u.gridShape, components=[x,y,z])
        sigma = self.sigma
        for n in velocity.components:
            for l in velocity.components:
                velocity[l] += ux[l,n]*sigma[z,n] 
                velocity[z] -= ux[l,n]*sigma[l,n]
        if self.Lambda != 0:
            #glide only
            sigma_tr = (sigma[x,x]+sigma[y,y]+sigma[z,z])/3.*self.Lambda
            velocity[z] += sigma_tr*(ux[x,x]+ux[y,y]+ux[z,z])
            for l in velocity.components:
                velocity[l] -= sigma_tr*(ux[l,z])

        velocity /= self.BetaP1D_RhoMod(u,ux)
        return velocity

    def BetaP2D_Velocity(self,u,ux,uy,opt=None):
        if self.prevTimeStep and self.vtime is not None and self.time == self.vtime and opt is not None and opt in self.velocity:
            #print "Reusing velocity", opt
            return self.velocity[opt]

        #print "Update velocity at t=", self.time, opt
        velocity = Fields.TensorField(u.gridShape, components=[x,y,z])
        sigma = self.sigma
        for n in velocity.components:
            for l in velocity.components:
                velocity[l] += ux[l,n]*sigma[x,n] 
                velocity[l] += uy[l,n]*sigma[y,n] 
                velocity[x] -= ux[l,n]*sigma[l,n]
                velocity[y] -= uy[l,n]*sigma[l,n]

        if self.Lambda != 0:
            #glide only
            sigma_tr = (sigma[x,x]+sigma[y,y]+sigma[z,z])/3.*self.Lambda
            velocity[x] += sigma_tr*(ux[x,x]+ux[y,y]+ux[z,z])
            velocity[y] += sigma_tr*(uy[x,x]+uy[y,y]+uy[z,z])
            for l in velocity.components:
                velocity[l] -= sigma_tr*(ux[l,x]+uy[l,y])

        velocity /= self.BetaP2D_RhoMod(u,ux,uy)

        if self.prevTimeStep and opt is not None:
            if self.vtime != self.time:
                self.velocity = {}
                self.vtime = self.time
            self.velocity[opt] = velocity
        return velocity

    def H_1D(self,field,deriv):
        V = self.BetaP1D_Velocity(field,deriv)
        H = deriv * V[z]
        for j in [x,y,z]:
            for l in [x,y,z]:
                H[z,j] -= V[l] * deriv[l,j]
        # glide only
        if self.Lambda != 0:
            J_trace = -self.Lambda/3.*(V[z]*(deriv[x,x]+deriv[y,y])-V[x]*deriv[x,z]-V[y]*deriv[y,z])
            for i in [x,y,z]:
                H[i,i] += J_trace

        return H
 
    def H_1Dprime(self,field,deriv):
        V = self.BetaP1D_Velocity(field,deriv)
        return V[z]

    def H_2D(self,field,deriv_x,deriv_y,opt=None):
        V = self.BetaP2D_Velocity(field,deriv_x,deriv_y,opt=opt)
        H = deriv_x * V[x] + deriv_y * V[y]
        for j in [x,y,z]:
            for l in [x,y,z]:
                H[x,j] -= V[l] * deriv_x[l,j]
                H[y,j] -= V[l] * deriv_y[l,j]
        # glide only
        if self.Lambda != 0:
            J_trace = -self.Lambda/3.*(V[x]*(deriv_x[y,y]+deriv_x[z,z]-deriv_y[x,y])+V[y]*(deriv_y[x,x]+deriv_y[z,z]-deriv_x[y,x])-V[z]*(deriv_x[z,x]+deriv_y[z,y]))
            for i in [x,y,z]:
                H[i,i] += J_trace

        return H

    def H_2Dprime(self,field,deriv_x,deriv_y,dir=0,opt=None):
        V = self.BetaP2D_Velocity(field,deriv_x,deriv_y,opt=opt)
        if dir==0:
            return V[x]
        elif dir==1:
            return V[y]
        else:
            assert False

    def GetSigma(self, state, time):
        sigma = state.CalculateSigma()
        return sigma

    def CalculateFlux(self, time, state, CFLCondition=False):
        self.state = state
        self.sigma = self.GetSigma(state,time)
        return CentralUpwindHJDynamics.CalculateFlux(self,time,state,CFLCondition=CFLCondition) 

