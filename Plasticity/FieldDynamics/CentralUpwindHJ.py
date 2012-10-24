from Plasticity.FieldDynamics import FieldDynamics

from Plasticity.NumericalMethods import *

"""
This controls the theta parameter for MinMod limiters.
Setting this to 0.0 reduces the scheme to a Local Lax-Friedrich like scheme.
"""
theta = 1.0

def FindDerivatives(field, coord=0, Dx=None):
    """
    one sided difference
    """
    if Dx is None:
        Dx = 1./field.gridShape[-1]
    diff = (rollfield(field, -1, coord)-field)
    """
    second order derivative minmod
    """ 
    #theta = 1.
    #Dx = 1./state.gridShape[-1]
    diff2_1 = theta*(rollfield(diff, -1, coord) - diff)/Dx/Dx
    diff2_2 = (rollfield(diff, -1, coord) - rollfield(diff, 1, coord))/Dx/Dx*0.5
    diff2_3 = rollfield(diff2_1, 1, coord)
    diff2 = MinMod3Field(diff2_1, diff2_2, diff2_3)
    """
    Evaluate right and left derivatives 
    """
    deriv_p = diff/Dx - 0.5*Dx*diff2
    # with N=32, this develops something weird probably a typo in the paper since it breaks the symmetry
    #deriv_m = rollfield(diff, 1, coord)/Dx + 0.5*Dx*diff2
    deriv_m = rollfield(diff, 1, coord)/Dx + 0.5*Dx*rollfield(diff2, 1, coord)
    return deriv_p, deriv_m

class CentralUpwindHJDynamics(FieldDynamics.FieldDynamics):
    def __init__(self, Dx=None):
        self.Dx = Dx
  
    def H_1D(self,field,deriv):
        pass
    def H_1Dprime(self,field,deriv):
        """
        This must return a scalar characteristic speed field
        """
        pass
    def H_2D(self,field,deriv_x,deriv_y,opt=None):
        pass
    def H_2Dprime(self,field,deriv_x,deriv_y,dir=0,opt=None):
        """
        This must return a scalar characteristic speed field
        """
        pass

    def CalculateFlux(self, time, state, CFLCondition=False):
        if state.dimension == 1:
            field = state.GetOrderParameterField()
            """
            Perform the interpolation + right and left derivative part
            """
            deriv_p, deriv_m = FindDerivatives(field, Dx=self.Dx)
            """
            Find out right and left velocities

            Here we are just using two point values assuming that we do not know the analytic form of H.
            For Burgers equation this should work just fine since Hprime is linear in u_x.
            """
            a_1 = self.H_1Dprime(field,deriv_m)
            a_2 = self.H_1Dprime(field,deriv_p)
            a_p = Max3(a_1,a_2,0)
            a_m = Min3(a_1,a_2,0).fabs()
            """
            Calculate time derivative field and return
            """
            # Added ME for correcting 0 velocity points
            a_tot = a_p+a_m+ME
            H_p = self.H_1D(field,deriv_p)
            H_m = self.H_1D(field,deriv_m)
            psi_int = (deriv_p*a_p + deriv_m*a_m)/(a_tot) - (H_p-H_m)/(a_tot)
            """
            Use either the less diffusive method by Kurganov or the old one.
            For 2D, the new method from adaptive Central Upwind scheme does not seem to ork very well because the new term diverges
            """
            #derivative = -(H_p*a_m+H_m*a_p)/(a_tot) + ((deriv_p-deriv_m)/(a_tot)-MinMod2Field((deriv_p-psi_int)/(a_tot), (psi_int-deriv_m)/(a_tot)))*a_m*a_p
            derivative = -(H_p*a_m+H_m*a_p)/(a_tot) + ((deriv_p-deriv_m)/(a_tot))*a_m*a_p

            if CFLCondition:
                return derivative, a_tot 
            else:
                return derivative
        elif state.dimension == 2:
            #print 't = ', time
            field = state.GetOrderParameterField()
            #print 'field', field['u']
            deriv_x_p, deriv_x_m = FindDerivatives(field, 0, Dx=self.Dx)
            #print 'd_x_p/m', deriv_x_p['u'], deriv_x_m['u']
            deriv_y_p, deriv_y_m = FindDerivatives(field, 1, Dx=self.Dx)
            #print 'd_y_p/m', deriv_y_p['u'], deriv_y_m['u']
            #self.SetTime(time)
            self.time = time
            a_1 = self.H_2Dprime(field,deriv_x_m,deriv_y_m,0,opt='--')
            a_2 = self.H_2Dprime(field,deriv_x_m,deriv_y_p,0,opt='-+')
            a_3 = self.H_2Dprime(field,deriv_x_p,deriv_y_m,0,opt='+-')
            a_4 = self.H_2Dprime(field,deriv_x_p,deriv_y_p,0,opt='++')

            a_p = Max5(a_1, a_2, a_3, a_4, 0.)
            a_m = Min5(a_1, a_2, a_3, a_4, 0.).fabs()

            b_1 = self.H_2Dprime(field,deriv_x_m,deriv_y_m,1,opt='--')
            b_2 = self.H_2Dprime(field,deriv_x_m,deriv_y_p,1,opt='-+')
            b_3 = self.H_2Dprime(field,deriv_x_p,deriv_y_m,1,opt='+-')
            b_4 = self.H_2Dprime(field,deriv_x_p,deriv_y_p,1,opt='++')

            b_p = Max5(b_1, b_2, b_3, b_4, 0.)
            b_m = Min5(b_1, b_2, b_3, b_4, 0.).fabs()

            """
            Lax-Friedrich like velocities - seem to cure problems, very similar to adaptive diffusion in theory
            """
            """
            a_p = a_p.max()
            a_m = a_m.max()
            a_p = a_m = max(a_p, a_m)

            b_p = b_p.max()
            b_m = b_m.max()
            b_p = b_m = max(b_p, b_m)
            """
            """
            a_p = Max5(self.H_2Dprime(field,deriv_x_m,deriv_y_m,0),\
                        self.H_2Dprime(field,deriv_x_m,deriv_y_p,0),\
                        self.H_2Dprime(field,deriv_x_p,deriv_y_m,0),\
                        self.H_2Dprime(field,deriv_x_p,deriv_y_p,0),\
                        ME)
            a_m = Min5(self.H_2Dprime(field,deriv_x_m,deriv_y_m,0),\
                        self.H_2Dprime(field,deriv_x_m,deriv_y_p,0),\
                        self.H_2Dprime(field,deriv_x_p,deriv_y_m,0),\
                        self.H_2Dprime(field,deriv_x_p,deriv_y_p,0),\
                        -ME).fabs()
            #print 'a_p/m', a_p['u'], a_m['u']
            b_p = Max5(self.H_2Dprime(field,deriv_x_m,deriv_y_m,1),\
                        self.H_2Dprime(field,deriv_x_m,deriv_y_p,1),\
                        self.H_2Dprime(field,deriv_x_p,deriv_y_m,1),\
                        self.H_2Dprime(field,deriv_x_p,deriv_y_p,1),\
                        ME)
            b_m = Min5(self.H_2Dprime(field,deriv_x_m,deriv_y_m,1),\
                        self.H_2Dprime(field,deriv_x_m,deriv_y_p,1),\
                        self.H_2Dprime(field,deriv_x_p,deriv_y_m,1),\
                        self.H_2Dprime(field,deriv_x_p,deriv_y_p,1),\
                        -ME).fabs()
            """
            a_tot = a_p+a_m+ME
            b_tot = b_p+b_m+ME

            H_p_p = self.H_2D(field,deriv_x_p,deriv_y_p,opt='++')
            H_p_m = self.H_2D(field,deriv_x_p,deriv_y_m,opt='+-')
            H_m_p = self.H_2D(field,deriv_x_m,deriv_y_p,opt='-+')
            H_m_m = self.H_2D(field,deriv_x_m,deriv_y_m,opt='--')

            psi_int_x_p = (deriv_x_p*a_p+deriv_x_m*a_m)/a_tot - (H_p_p-H_m_p)/a_tot
            psi_int_x_m = (deriv_x_p*a_p+deriv_x_m*a_m)/a_tot - (H_p_m-H_m_m)/a_tot
            psi_int_y_p = (deriv_y_p*b_p+deriv_y_m*b_m)/b_tot - (H_p_p-H_p_m)/b_tot
            psi_int_y_m = (deriv_y_p*b_p+deriv_y_m*b_m)/b_tot - (H_m_p-H_m_m)/b_tot
            """ 
            derivative = -(H_p_p*a_m*b_m+H_p_m*a_m*b_p+H_m_p*a_p*b_m+H_m_m*a_p*b_p)/a_tot/b_tot +\
                        ((deriv_x_p-deriv_x_m)*a_m*a_p/a_tot \
                                -MinMod2Field((deriv_x_p-psi_int_x_m)*b_p/b_tot/a_tot, (psi_int_x_m-deriv_x_m)/a_tot) \
                                -MinMod2Field((deriv_x_p-psi_int_x_p)*b_m/b_tot/a_tot, (psi_int_x_p-deriv_x_m)/a_tot)) +\
                        ((deriv_y_p-deriv_y_m)*b_m*b_p/b_tot \
                                -MinMod2Field((deriv_y_p-psi_int_y_m)*a_p/a_tot/b_tot, (psi_int_y_m-deriv_y_m)/b_tot) \
                                -MinMod2Field((deriv_y_p-psi_int_y_p)*a_m/a_tot/b_tot, (psi_int_y_p-deriv_y_m)/b_tot))
                                )
            """ 
            derivative = -(H_p_p*a_m*b_m+H_p_m*a_m*b_p+H_m_p*a_p*b_m+H_m_m*a_p*b_p)/a_tot/b_tot +\
                        ((deriv_x_p-deriv_x_m)*a_m*a_p/a_tot ) + \
                        ((deriv_y_p-deriv_y_m)*b_m*b_p/b_tot )
            if CFLCondition:
                return derivative, a_tot+b_tot
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
import BurgersState

def Burgers1D_H(u,ux):
    #return u*ux
    return ux*(ux['u']+ux['v'])*0.5*2

def Burgers1D_Hprime(u,ux):
    #return u
    return (ux['v']+ux['u'])*2

def Burgers2D_H(u,ux,uy):
    return 0.5*(ux*ux+uy*uy)

def Burgers2D_Hprime(u,ux,uy,dir=0):
    if dir == 0:
        return ux['u']
    elif dir == 1:
        return uy['u']
    else:
        assert False
    
class BurgersHJDynamics(CentralUpwindHJDynamics):
    def H_1D(self,field,deriv):
        return Burgers1D_H(field,deriv)
    def H_1Dprime(self,field,deriv):
        return Burgers1D_Hprime(field,deriv)
    def H_2D(self,field,deriv_x,deriv_y):
        return Burgers2D_H(field,deriv_x,deriv_y)
    def H_2Dprime(self,field,deriv_x,deriv_y,dir=0):
        return Burgers2D_Hprime(field,deriv_x,deriv_y,dir)

