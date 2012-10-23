import FieldDynamics
import Fields

from NumericalMethods import *

"""
This controls the theta parameter for MinMod limiters.
Setting this to 0.0 reduces the scheme to a Local Lax-Friedrich like scheme.
"""
theta = 1.0

class CentralUpwindDynamics(FieldDynamics.FieldDynamics):
    """
    This base class for central upwind dynamics implement 2nd order
    central upwind scheme by Kurganov et al for hyperbolic conservation
    laws.

    For writing a different equation you have to overload
    F_1D, F_1Dprime
    F_2D, F_2Dprime
    and Q_1D/Q_2D if necessary
    """
    def __init__(self, Dx=None):
        self.Dx = Dx
        self.Q = None
 
    def F_1D(self,field):
        pass
    def F_1Dprime(self,field):
        """
        This must return a scalar characteristic speed field
        """
        pass
    def F_2D(self,field,dir=0,opt=None):
        pass
    def F_2Dprime(self,field,dir=0,opt=None):
        """
        This must return a scalar characteristic speed field
        """
        pass

    def Q_1D(self,field,deriv):
        pass
    def Q_2D(self,field,deriv_x,deriv_y,dir=0):
        pass

    def CalculateFluxOSD(self, time, state, CFLCondition=False, dim=0):
        """
        Calculate flux by operator splitting dimensions

        evaluate one dimension at a time
        """
        field = state.GetOrderParameterField()
        """
        Perform the interpolation + right and left derivative part
        """
        Dx = 1./state.gridShape[-1]
        deriv_p = (rollfield(field, -1, dim) - field)/Dx
        deriv_m = rollfield(deriv_p, 1, dim)
        #u_x = MinMod2Field(deriv_m, deriv_p)
        u_x = MinMod3Field(theta*deriv_m, theta*deriv_p, (rollfield(field,-1,dim)-rollfield(field,1,dim))/Dx*0.5)

        # u_(x+1/2)^+
        field_p = rollfield(field - 0.5*Dx*u_x, -1, dim)
        # u_(x+1/2)^-
        field_m = field + 0.5*Dx*u_x

        """
        Find out the characteristic velocities
        """
        # a_(x+1/2)
        a = Max2(self.F_2Dprime(field_m,dir=dim), self.F_2Dprime(field_p,dir=dim))

        """
        now calculate the rhs
        """
        
        f_p = self.F_2D(field_p,dir=dim) 
        f_m = self.F_2D(field_m,dir=dim) 
        f_t = f_p+f_m
        
        a_term = (field_p-field_m)*a
        f_plus_a = f_t-a_term
        derivative = -0.5/Dx*(f_plus_a-rollfield(f_plus_a,1,dim))

        #FIXME - add Q for diffusion terms
        if CFLCondition:
            return derivative, a.fabs()
        else:
            return derivative

    def CalculateFlux(self, time, state, CFLCondition=False):
        if state.dimension == 1:
            field = state.GetOrderParameterField()
            """
            Perform the interpolation + right and left derivative part
            """
            Dx = 1./state.gridShape[-1]
            deriv_p = (rollfield(field, -1, 0) - field)/Dx
            deriv_m = rollfield(deriv_p, 1, 0)
            #u_x = MinMod2Field(deriv_m, deriv_p)
            u_x = MinMod3Field(theta*deriv_m, theta*deriv_p, (rollfield(field,-1,0)-rollfield(field,1,0))/Dx*0.5)

            # u_(x+1/2)^+
            field_p = rollfield(field - 0.5*Dx*u_x, -1, 0)
            # u_(x+1/2)^-
            field_m = field + 0.5*Dx*u_x

            """
            Find out the characteristic velocities
            """
            # a_(x+1/2)
            a = Max2(self.F_1Dprime(field_m), self.F_1Dprime(field_p))

            """
            now calculate the rhs
            """
            
            f_p = self.F_1D(field_p) 
            f_m = self.F_1D(field_m) 
            f_t = f_p+f_m
            
            a_term = (field_p-field_m)*a
            f_plus_a = f_t-a_term
            derivative = -0.5/Dx*(f_plus_a-rollfield(f_plus_a,1,0))

            #FIXME - add Q for diffusion terms
            if CFLCondition:
                return derivative, a.fabs()
            else:
                return derivative
        elif state.dimension == 2:
            field = state.GetOrderParameterField()
            """
            Perform the interpolation + right and left derivative part
            """
            Dx = 1./state.gridShape[-1]
            deriv_x_p = (rollfield(field, -1, 0) - field)/Dx
            deriv_x_m = rollfield(deriv_x_p, 1, 0)
            deriv_y_p = (rollfield(field, -1, 1) - field)/Dx
            deriv_y_m = rollfield(deriv_y_p, 1, 1)
            #u_x = MinMod2Field(deriv_x_m, deriv_x_p)
            u_x = MinMod3Field(theta*deriv_x_m, theta*deriv_x_p, (rollfield(field,-1,0)-rollfield(field,1,0))/Dx*0.5)
            u_y = MinMod3Field(theta*deriv_y_m, theta*deriv_y_p, (rollfield(field,-1,1)-rollfield(field,1,1))/Dx*0.5)

            # u_(x+1/2)^+
            field_x_p = rollfield(field - 0.5*Dx*u_x, -1, 0)
            field_y_p = rollfield(field - 0.5*Dx*u_y, -1, 1)
            # u_(x+1/2)^-
            field_x_m = field + 0.5*Dx*u_x
            field_y_m = field + 0.5*Dx*u_y

            """
            Find out the characteristic velocities
            """
            # a_(x+1/2)
            a_x = Max2(self.F_2Dprime(field_x_m,opt='-'), self.F_2Dprime(field_x_p,opt='+'))
            a_y = Max2(self.F_2Dprime(field_y_m,dir=1,opt='-'), self.F_2Dprime(field_y_p,dir=1,opt='+'))

            """
            now calculate the rhs
            """
            
            f_x_p = self.F_2D(field_x_p,opt='+') 
            f_x_m = self.F_2D(field_x_m,opt='-') 
            f_y_p = self.F_2D(field_y_p,dir=1,opt='+') 
            f_y_m = self.F_2D(field_y_m,dir=1,opt='+') 

            f_x_t = f_x_p+f_x_m
            f_y_t = f_y_p+f_y_m
           
            a_x_term = (field_x_p-field_x_m)*a_x
            f_x_plus_a = f_x_t-a_x_term
            a_y_term = (field_y_p-field_y_m)*a_y
            f_y_plus_a = f_y_t-a_y_term

            if self.Q is not None:
                q_x_p = 0.5*(self.Q_2D(field,(rollfield(field,-1,0)-field)/Dx,u_y,dir=0)+self.Q_2D(rollfield(field,-1,0),(rollfield(field,-1,0)-field)/Dx,rollfield(u_y,-1,0)))
                q_y_p = 0.5*(self.Q_2D(field,u_x,(rollfield(field,-1,1)-field)/Dx,dir=1)+self.Q_2D(rollfield(field,-1,1),rollfield(u_x,-1,1),(rollfield(field,-1,1)-field)/Dx))
                f_x_plus_a += self.Q*q_x_p
                f_y_plus_a += self.Q*q_y_p
            derivative = -0.5/Dx*(f_x_plus_a-rollfield(f_x_plus_a,1,0)+f_y_plus_a-rollfield(f_y_plus_a,1,1))

            if CFLCondition:
                return derivative, a_x.fabs()+a_y.fabs()
            else:
                return derivative
        else:
            """
            Not implemented for now
            """
            assert False 

"""
From here on we have specific parts for Burger's equation
"""

def Burgers1D_F(u):
    return 0.5*u*(u['u']+u['v'])

def Burgers1D_Fprime(u):
    return (u['u']+u['v']).fabs()
    #return ux['u']

def Burgers2D_F(u,dir=0):
    tot = 0.5*(u['u']**2+u['v']**2)
    if dir==0:
        ret = Fields.TensorField(u.gridShape, u.components)
        ret['u'] = tot
        return ret
    elif dir==1:
        ret = Fields.TensorField(u.gridShape, u.components)
        ret['v'] = tot
        return ret

def Burgers2D_Fprime(u,dir=0):
    return Max2(u['u'].fabs(), u['v'].fabs())
    
class BurgersDynamics(CentralUpwindDynamics):
    def F_1D(self,field):
        return Burgers1D_F(field)
    def F_1Dprime(self,field):
        return Burgers1D_Fprime(field)
    def F_2D(self,field,dir=0,opt=None):
        return Burgers2D_F(field,dir)
    def F_2Dprime(self,field,dir=0,opt=None):
        return Burgers2D_Fprime(field,dir)

