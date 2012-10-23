from CentralUpwindHJ3D import *
from Constants import *

import NumericalMethods
import Fields

"""
Define BetaP H-J type dynamics
"""
class BetaPDynamics(CentralUpwindHJDynamics):
    def __init__(self, Dx=None, coreEnergy=0, coreEnergyLog=0):
        self.Dx = Dx
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

    def BetaP3D_RhoMod(self,u,ux,uy,uz):
        rho2 = NumericalMethods.ME
        for component in ux.components:
            rho2 += ux[component]*ux[component] 
            rho2 += uy[component]*uy[component] 
            rho2 += uz[component]*uz[component] 
        for j in [x,y,z]:
            rho2 -= ux[x,j]*ux[x,j]
            rho2 -= uy[y,j]*uy[y,j]
            rho2 -= uz[z,j]*uz[z,j]
            rho2 -= 2*ux[y,j]*uy[x,j]
            rho2 -= 2*uy[z,j]*uz[y,j]
            rho2 -= 2*uz[x,j]*ux[z,j]
        return rho2.sqrt()

    def BetaP1D_Velocity(self,u,ux):
        velocity = Fields.TensorField(u.gridShape, components=[x,y,z])
        sigma = self.sigma
        for n in velocity.components:
            for l in velocity.components:
                velocity[l] += ux[l,n]*sigma[z,n] 
                velocity[z] -= ux[l,n]*sigma[l,n]
        velocity /= self.BetaP1D_RhoMod(u,ux)
        return velocity

    def BetaP2D_ClimbVelocity(self,u,ux,uy,opt=None):
        """
        v_l = \sigma_{st} \vrho_{lst}/|\rho|
            = (-\sigma_{st}\partial_l\betaP_{st} + \sigma_{st}\partial_s\betaP_{lt})/|rho| 
        """
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
        if self.prevTimeStep and opt is not None:
            if self.vtime != self.time:
                self.velocity = {}
                self.vtime = self.time
            self.velocity[opt] = velocity
        return velocity

    def BetaP2D_Velocity(self,u,ux,uy,opt=None):
        """
        v_l' = (\simga_{st}-\delta_{st}\sigma_{ab}\vrho_{pab}\vrho_{pkk}/\vrho_{qcc}/\vrho_{qdd})\vrho_{lst}/|\rho|
             = v_l -\vrho_{lss}\vrho_{pkk}v_p/\vrho_{qcc}/\vrho_{qdd}/|\rho|
             = v_l -\vrho_{lss}\vrho_{pkk}v_p / (\rho_{mn}\rho_{mn}-\rho_{mn}\rho_{nm}) / |\rho|
             = v_l - \vrho_{lss}rhov/rhorho
        rhov = v_p\vrho_{pkk}, 
        rhorho = \vrho_{lss}\vrho_{\ltt}= (\d_q\betaP_{kk})^2+(\d_k\betaP_{qk})^2-2\d_q\betaP_{kk}\d_t\betaP_{qt}.
        """
        if self.prevTimeStep and self.vtime is not None and self.time == self.vtime and opt is not None and opt in self.velocity:
            #print "Reusing velocity", opt
            return self.velocity[opt]
        #print "Update velocity at t=", self.time, opt
        rhorho = NumericalMethods.ME
        uxTr = ux[x,x]+ux[y,y]+ux[z,z]
        uyTr = uy[x,x]+uy[y,y]+uy[z,z]
        rhorho += uxTr*uxTr + uyTr*uyTr 
        rhorho -= 2*uxTr*(ux[x,x]+uy[x,y])
        rhorho -= 2*uyTr*(ux[y,x]+uy[y,y])
        for n in [x,y,z]:
            rhorho += (ux[n,x]+uy[n,y])*(ux[n,x]+uy[n,y])
        v = self.BetaP2D_ClimbVelocity(u,ux,uy,opt)
        rhov = NumericalMethods.ME
        rhov += v[x]*(ux[z,z]+ux[y,y]-uy[x,y]) 
        rhov += v[y]*(uy[x,x]+uy[z,z]-ux[y,x]) 
        rhov += -v[z]*(ux[z,x]+uy[z,y]) 
        v[x] += (ux[y,y]+ux[z,z]-uy[x,y])*rhov/rhorho  
        v[y] += (uy[x,x]+uy[z,z]-ux[y,x])*rhov/rhorho  
        v[z] -= (ux[z,x]+uy[z,y])*rhov/rhorho 
        v /= self.BetaP2D_RhoMod(u,ux,uy)
        if self.prevTimeStep and opt is not None:
            if self.vtime != self.time:
                self.velocity = {}
                self.vtime = self.time
            self.velocity[opt] = v
        return v

 
    def BetaP3D_ClimbVelocity(self,u,ux,uy,uz,opt=None):
        """
        v_l = \sigma_{st} \vrho_{lst}/|\rho|
            = (-\sigma_{st}\partial_l\betaP_{st} + \sigma_{st}\partial_s\betaP_{lt})/|rho| 
        """
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
                velocity[l] += uz[l,n]*sigma[z,n] 
                velocity[x] -= ux[l,n]*sigma[l,n]
                velocity[y] -= uy[l,n]*sigma[l,n]
                velocity[z] -= uz[l,n]*sigma[l,n]
        if self.prevTimeStep and opt is not None:
            if self.vtime != self.time:
                self.velocity = {}
                self.vtime = self.time
            self.velocity[opt] = velocity
        return velocity

    def BetaP3D_Velocity(self,u,ux,uy,uz,opt=None):
        """
        v_l' = (\simga_{st}-\delta_{st}\sigma_{ab}\vrho_{pab}\vrho_{pkk}/\vrho_{qcc}/\vrho_{qdd})\vrho_{lst}/|\rho|
             = v_l -\vrho_{lss}\vrho_{pkk}v_p/\vrho_{qcc}/\vrho_{qdd}/|\rho|
             = v_l -\vrho_{lss}\vrho_{pkk}v_p / (\rho_{mn}\rho_{mn}-\rho_{mn}\rho_{nm}) / |\rho|
             = v_l - \vrho_{lss}rhov/rhorho
        rhov = v_p\vrho_{pkk}, 
        rhorho = \vrho_{lss}\vrho_{\ltt}= (\d_q\betaP_{kk})^2+(\d_k\betaP_{qk})^2-2\d_q\betaP_{kk}\d_t\betaP_{qt}.
        """
        if self.prevTimeStep and self.vtime is not None and self.time == self.vtime and opt is not None and opt in self.velocity:
            #print "Reusing velocity", opt
            return self.velocity[opt]
        #print "Update velocity at t=", self.time, opt
        rhorho = NumericalMethods.ME
        uxTr = ux[x,x]+ux[y,y]+ux[z,z]
        uyTr = uy[x,x]+uy[y,y]+uy[z,z]
        uzTr = uz[x,x]+uz[y,y]+uz[z,z]
        rhorho += uxTr*uxTr + uyTr*uyTr +uzTr*uzTr
        rhorho -= 2*uxTr*(ux[x,x]+uy[x,y]+uz[x,z])
        rhorho -= 2*uyTr*(ux[y,x]+uy[y,y]+uz[y,z])
        rhorho -= 2*uzTr*(ux[z,x]+uy[z,y]+uz[z,z])
        for n in [x,y,z]:
            rhorho += (ux[n,x]+uy[n,y]+uz[n,z])*(ux[n,x]+uy[n,y]+uz[n,z])

        del uxTr, uyTr, uzTr

        v = self.BetaP3D_ClimbVelocity(u,ux,uy,uz,opt)
        rhov = NumericalMethods.ME
        rhov += v[x]*(ux[z,z]+ux[y,y]-uy[x,y]-uz[x,z]) 
        rhov += v[y]*(uy[x,x]+uy[z,z]-ux[y,x]-uz[y,z]) 
        rhov += v[z]*(uz[x,x]+uz[y,y]-ux[z,x]-uy[z,y])
 
        factor = rhov/rhorho
        del rhov, rhorho


        v[x] += (ux[y,y]+ux[z,z]-uy[x,y]-uz[x,z])*factor
        v[y] += (uy[x,x]+uy[z,z]-ux[y,x]-uz[y,z])*factor
        v[z] += (uz[x,x]+uz[y,y]-ux[z,x]-uy[z,y])*factor
        v /= self.BetaP3D_RhoMod(u,ux,uy,uz)
        if self.prevTimeStep and opt is not None:
            if self.vtime != self.time:
                self.velocity = {}
                self.vtime = self.time
            self.velocity[opt] = v
        return v

    def H_1D(self,field,deriv):
        V = self.BetaP1D_Velocity(field,deriv)
        H = deriv * V[z]
        for j in [x,y,z]:
            for l in [x,y,z]:
                H[z,j] -= V[l] * deriv[l,j]
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
        return H

    def H_2Dprime(self,field,deriv_x,deriv_y,dir=0,opt=None):
        V = self.BetaP2D_Velocity(field,deriv_x,deriv_y,opt=opt)
        if dir==0:
            return V[x]
        elif dir==1:
            return V[y]
        else:
            assert False

    def H_3D(self,field,deriv_x,deriv_y,deriv_z,opt=None):
        V = self.BetaP3D_Velocity(field,deriv_x,deriv_y,deriv_z,opt=opt)
        H = deriv_x * V[x] + deriv_y * V[y] + deriv_z * V[z]
        for j in [x,y,z]:
            for l in [x,y,z]:
                H[x,j] -= V[l] * deriv_x[l,j]
                H[y,j] -= V[l] * deriv_y[l,j]
                H[z,j] -= V[l] * deriv_z[l,j]
        return H

    def H_3Dprime(self,field,deriv_x,deriv_y,deriv_z,dir=0,opt=None):
        V = self.BetaP3D_Velocity(field,deriv_x,deriv_y,deriv_z,opt=opt)
        if dir==0:
            return V[x]
        elif dir==1:
            return V[y]
        elif dir==2:
            return V[z]
        else:
            assert False

    def GetSigma(self, state, time):
        sigma = state.CalculateSigma()
        return sigma

    def CalculateFlux(self, time, state, CFLCondition=False):
        self.state = state
        self.sigma = self.GetSigma(state,time)
        return CentralUpwindHJDynamics.CalculateFlux(self,time,state,CFLCondition=CFLCondition) 
