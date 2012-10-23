from numpy import finfo, iterable
import sys

ME = finfo(float).eps

def SymmetricDerivative(In,gridShape,direction,order=1):
    """
    Calculate the symmetric numerical directional derivative of an array.

    direction refers to the axis that you want to take the derivative along.
    """
    if   order == 1:
        return 0.5*gridShape[-1]*(In.roll(-1,direction)-In.roll(1,direction))
        # Use above for rectangular simulations
        #return 0.5*gridShape[direction]*(In.roll(-1,direction)-In.roll(1,direction))
    elif order == 2:
        return gridShape[-1]**2*(In.roll(-1,direction)-2.*In+In.roll(1,direction))
        # Use above for rectangular simulations
        #return gridShape[direction]**2*(In.roll(-1,direction)-2.*In+In.roll(1,direction))
    elif order == 4:
        return gridShape[-1]**4*(In.roll(-2,direction)-4.*In.roll(-1,direction)+6.*In-4.*In.roll(1,direction)+In.roll(2,direction))
        # Use above for rectangular simulations
        #return gridShape[direction]**4*(In.roll(-2,direction)-4.*In.roll(-1,direction)+6.*In-4.*In.roll(1,direction)+In.roll(2,direction))
    else:
        raise ValueError, "We calculate the derivatives with the order of 1, 2 or 4."


def UpwindDerivative(In,gridShape,velocity, direction):
    """
    Calculate the upwind derivative of an array. 

    velocity gives the upwind direction at each grid point.
    """
    right_moving = (velocity>0)
    return gridShape[direction]*((In-In.roll(1,direction))*right_moving + (In.roll(-1,direction)-In)* (1.-right_moving))


def ENO_Derivative(In,spatialDirection,flowDirection,order,dimension):
    """ 
    Implementation based on ENO interpolation algorithm from

    C. Shu, Essential Non-Oscillatory and Weighted Essentially Non-Oscillatory 
    Schemes for Hyperbolic Conservation Laws, ICASE Report No. 97-65

    page 52. Basically it is as follows:

    (1) P^1(x) = f[x_j] + f[x_j, x_j+1](x-x_j), k^1 = j
    (2) If k^(l-1) and P^(l-1) are both defined, then let
       
       a^l = f[x_k^(l-1),...,x_k^(l-1)+1]
       b^l = f[x_k^(l-1)-1,...,x_k^(l-1)]

    (i) If |a^l| >= |b^l|, then c^l=b^l, k^l = k^(l-1) - 1
                    otherwise,  c^l=a^l, k^l = k^(l-1)
    (ii) P^l = P^(l-1) + c^l \Prod^{k^(l-1)+l-1}_{i=k^(l-1)} (x-x_i)

    f[...] are Newton divided differences, inductively defined as
    f[x_1, x_2, ... , x_k+1] = 
             (f[x_2, ..., x_k+1] - f[x_1, ..., x_k]) / (x_k+1 - x_1)
    with f[x_1] = f(x_1).

    """

    # FIXME - do we need to check this every time?
    if spatialDirection not in range(dimension): 
        raise ValueError, "Spatial direction must be less than the dimension"

    spatialDirection -= dimension

    if flowDirection == 1:
        NDD1 = In.roll(-1,spatialDirection)-In
        c1   = NDD1
        if order == 1:
            return c1
        NDD2 = (NDD1.roll(-1,spatialDirection)-NDD1)/2.
        a2   = NDD2
        b2   = NDD2.roll(1,spatialDirection)
        RM2  = (a2.fabs()>b2.fabs())
        c2   = RM2*b2 + (1.-RM2)*a2
        if order == 2:
            return c1-c2
        NDD3 = (NDD2.roll(-1,spatialDirection)-NDD2)/3.
        a3   = RM2*NDD3.roll(1,spatialDirection)+(1.-RM2)*NDD3
        b3   = RM2*NDD3.roll(2,spatialDirection)+(1.-RM2)*NDD3.roll(1,spatialDirection)
        RM3  = (a3.fabs()>b3.fabs())
        c3   = RM3*b3 + (1.-RM3)*a3
        if order == 3:
            return c1-c2+(2.-3.*RM2)*c3
        NDD4 = (NDD3.roll(-1,spatialDirection)-NDD3)/4.
        a4   = RM2*RM3*NDD4.roll(2,spatialDirection)+(RM2+RM3-2.*RM2*RM3)*NDD4.roll(1,spatialDirection)+(1.-RM2)*(1.-RM3)*NDD4
        b4   = RM2*RM3*NDD4.roll(3,spatialDirection)+(RM2+RM3-2.*RM2*RM3)*NDD4.roll(2,spatialDirection)+(1.-RM2)*(1.-RM3)*NDD4.roll(1,spatialDirection)
        RM4  = (a4.fabs()>b4.fabs())
        c4   = RM4*b4 + (1.-RM4)*a4
        if order == 4:
            return c1-c2+(2.-3.*RM2)*c3-(6.-8.*(RM2+RM3)+12.*RM2*RM3)*c4
        NDD5 = (NDD4.roll(-1,spatialDirection)-NDD4)/5.
        a5   = RM2*RM3*RM4*NDD5.roll(3,spatialDirection)+(RM2*RM4+RM3*RM4+RM2*RM3-3.*RM2*RM3*RM4)*NDD5.roll(2,spatialDirection)+\
               ((1.-RM2)*(1.-RM3)*RM4+(1.-RM2)*(1.-RM4)*RM3+(1.-RM3)*(1.-RM4)*RM2)*NDD5.roll(1,spatialDirection)+\
               (1.-RM2)*(1.-RM3)*(1.-RM4)*NDD5
        b5   = RM2*RM3*RM4*NDD5.roll(4,spatialDirection)+(RM2*RM4+RM3*RM4+RM2*RM3-3.*RM2*RM3*RM4)*NDD5.roll(3,spatialDirection)+\
               ((1.-RM2)*(1.-RM3)*RM4+(1.-RM2)*(1.-RM4)*RM3+(1.-RM3)*(1.-RM4)*RM2)*NDD5.roll(2,spatialDirection)+\
               (1.-RM2)*(1.-RM3)*(1.-RM4)*NDD5.roll(1,spatialDirection)
        RM5  = (a5.fabs()>b5.fabs())
        c5   = RM5*b5 + (1.-RM5)*a5
        if order == 5:
            return c1-c2+(2.-3.*RM2)*c3-(6.-8.*(RM2+RM3)+12.*RM2*RM3)*c4+(24.-30.*(RM2+RM3+RM4)+40.*(RM2*RM3+RM2*RM4+RM3*RM4)-60.*RM2*RM3*RM4)*c5
    else:
        NDD1 = In - In.roll(1,spatialDirection)
        c1   = NDD1
        if order == 1:
            return c1
        NDD2 = (NDD1-NDD1.roll(1,spatialDirection))/2.
        a2   = NDD2
        b2   = NDD2.roll(-1,spatialDirection)
        RM2  = (a2.fabs()>b2.fabs())
        c2   = RM2*b2 + (1.-RM2)*a2
        if order == 2:
            return c1+c2
        NDD3 = (NDD2-NDD2.roll(1,spatialDirection))/3.
        a3   = RM2*NDD3.roll(-1,spatialDirection)+(1.-RM2)*NDD3
        b3   = RM2*NDD3.roll(-2,spatialDirection)+(1.-RM2)*NDD3.roll(-1,spatialDirection)
        RM3  = (a3.fabs()>b3.fabs())
        c3   = RM3*b3 + (1.-RM3)*a3
        if order == 3:
            return c1+c2+(2.-3.*RM2)*c3
        NDD4 = (NDD3-NDD3.roll(1,spatialDirection))/4.
        a4   = RM2*RM3*NDD4.roll(-2,spatialDirection)+(RM2+RM3-2.*RM2*RM3)*NDD4.roll(-1,spatialDirection)+(1.-RM2)*(1.-RM3)*NDD4
        b4   = RM2*RM3*NDD4.roll(-3,spatialDirection)+(RM2+RM3-2.*RM2*RM3)*NDD4.roll(-2,spatialDirection)+(1.-RM2)*(1.-RM3)*NDD4.roll(-1,spatialDirection)
        RM4  = (a4.fabs()>b4.fabs())
        c4   = RM4*b4 + (1.-RM4)*a4
        if order == 4:
            return c1+c2+(2.-3.*RM2)*c3+(6.-8.*(RM2+RM3)+12.*RM2*RM3)*c4
        NDD5 = (NDD4-NDD4.roll(1,spatialDirection))/5.
        a5   = RM2*RM3*RM4*NDD5.roll(-3,spatialDirection)+(RM2*RM4+RM3*RM4+RM2*RM3-3.*RM2*RM3*RM4)*NDD5.roll(-2,spatialDirection)+\
               ((1.-RM2)*(1.-RM3)*RM4+(1.-RM2)*(1.-RM4)*RM3+(1.-RM3)*(1.-RM4)*RM2)*NDD5.roll(-1,spatialDirection)+\
               (1.-RM2)*(1.-RM3)*(1.-RM4)*NDD5
        b5   = RM2*RM3*RM4*NDD5.roll(-4,spatialDirection)+(RM2*RM4+RM3*RM4+RM2*RM3-3.*RM2*RM3*RM4)*NDD5.roll(-3,spatialDirection)+\
               ((1.-RM2)*(1.-RM3)*RM4+(1.-RM2)*(1.-RM4)*RM3+(1.-RM3)*(1.-RM4)*RM2)*NDD5.roll(-2,spatialDirection)+\
               (1.-RM2)*(1.-RM3)*(1.-RM4)*NDD5.roll(-1,spatialDirection)
        RM5  = (a5.fabs()>b5.fabs())
        c5   = RM5*b5 + (1.-RM5)*a5
        if order == 5:
            return c1+c2+(2.-3.*RM2)*c3+(6.-8.*(RM2+RM3)+12.*RM2*RM3)*c4+(24.-30.*(RM2+RM3+RM4)+40.*(RM2*RM3+RM2*RM4+RM3*RM4)-60.*RM2*RM3*RM4)*c5


def WENO_Derivative(In,spatialDirection,flowDirection,order,dimension):
    """ needs to be fixed, should work for any direction smaller than the dimension """
    if spatialDirection not in [0,1,2]:
        raise ValueError, "Spatial direction must be 0, 1 or 2, corresponding to x, y or z."

    spatialDirection -= dimension

    ENO_order = (order+1)/2
    beta      = [None]*ENO_order
    alpha     = [None]*ENO_order
    d_f       = [[1.],[2./3.,1./3.],[0.3,0.6,0.1]]
    d_b       = [[1.],[1./3.,2./3.],[0.1,0.6,0.3]]
    epsilon   = 1.e-6
    if flowDirection == 1:
        NDD1 = In.roll(-1,spatialDirection)-In
        v1   = NDD1
        if order == 1:
            return v1
        NDD2 = (NDD1.roll(-1,spatialDirection)-NDD1)/2.
        a2   = NDD2
        b2   = NDD2.roll(1,spatialDirection)
        RM2  = (a2.fabs()>b2.fabs())
        c2   = RM2*b2 + (1.-RM2)*a2
        v2   = v1-c2
        if order == 3:
            v       = v1*d_f[1][0]+v2*d_f[1][1]
            beta[0] = (v.roll(-1,spatialDirection)-v)**2
            beta[1] = (v-v.roll(1,spatialDirection))**2
            for i in range(2):
                alpha[i]  = d_f[1][i]/((epsilon+beta[i])**2)
            omega = array(alpha)/(alpha[0]+alpha[1])
            return  omega[0]*v1+omega[1]*v2
        NDD3 = (NDD2.roll(-1,spatialDirection)-NDD2)/3.
        a3   = RM2*NDD3.roll(1,spatialDirection)+(1.-RM2)*NDD3
        b3   = RM2*NDD3.roll(2,spatialDirection)+(1.-RM2)*NDD3.roll(1,spatialDirection)
        RM3  = (a3.fabs()>b3.fabs())
        c3   = RM3*b3 + (1.-RM3)*a3
        v3   = v2+(2.-3.*RM2)*c3
        if order == 5:
            v       = v1*d_f[2][0]+v2*d_f[2][1]+v3*d_f[2][2]
            beta[0] = (13./12)*(v-2.*v.roll(-1,spatialDirection)+v.roll(-2,spatialDirection))**2+\
                      (1./4)*(3.*v-4.*v.roll(-1,spatialDirection)+v.roll(-2,spatialDirection))**2 
            beta[1] = (13./12)*(v.roll(1,spatialDirection)-2.*v+v.roll(-1,spatialDirection))**2+\
                      (1./4)*(v.roll(1,spatialDirection)-v.roll(-1,spatialDirection))**2 
            beta[2] = (13./12)*(v.roll(2,spatialDirection)-2.*v.roll(1,spatialDirection)+v)**2+\
                      (1./4)*(v.roll(2,spatialDirection)-4.*v.roll(1,spatialDirection)+3.*v)**2 
            for i in range(3):
                alpha[i]  = d_f[2][i]/((epsilon+beta[i])**2)
            omega = array(alpha)/(alpha[0]+alpha[1]+alpha[2])
            return  omega[0]*v1+omega[1]*v2+omega[2]*v3
    else: 
        NDD1 = In - In.roll(1,spatialDirection)
        v1   = NDD1
        if order == 1:
            return  v1 
        NDD2 = (NDD1-NDD1.roll(1,spatialDirection))/2.
        a2   = NDD2
        b2   = NDD2.roll(-1,spatialDirection)
        RM2  = (a2.fabs()>b2.fabs())
        c2   = RM2*b2 + (1.-RM2)*a2
        v2   = v1+c2
        if order == 3:
            v       = v1*d_b[1][0]+v2*d_b[1][1]
            beta[0] = (v.roll(-1,spatialDirection)-v)**2
            beta[1] = (v-v.roll(1,spatialDirection))**2
            for i in range(2):
                alpha[i]  = d_b[1][i]/((epsilon+beta[i])**2)
            omega = array(alpha)/(alpha[0]+alpha[1])
            return  omega[0]*v1+omega[1]*v2
        NDD3 = (NDD2-NDD2.roll(1,spatialDirection))/3.
        a3   = RM2*NDD3.roll(-1,spatialDirection)+(1.-RM2)*NDD3
        b3   = RM2*NDD3.roll(-2,spatialDirection)+(1.-RM2)*NDD3.roll(-1,spatialDirection)
        RM3  = (a3.fabs()>b3.fabs())
        c3   = RM3*b3 + (1.-RM3)*a3
        v3   = v2+(2.-3.*RM2)*c3
        if order == 5:
            v       = v1*d_b[2][0]+v2*d_b[2][1]+v3*d_b[2][2]
            beta[0] = (13./12)*(v-2.*v.roll(-1,spatialDirection)+v.roll(-2,spatialDirection))**2+\
                      (1./4)*(3.*v-4.*v.roll(-1,spatialDirection)+v.roll(-2,spatialDirection))**2 
            beta[1] = (13./12)*(v.roll(1,spatialDirection)-2.*v+v.roll(-1,spatialDirection))**2+\
                      (1./4)*(v.roll(1,spatialDirection)-v.roll(-1,spatialDirection))**2 
            beta[2] = (13./12)*(v.roll(2,spatialDirection)-2.*v.roll(1,spatialDirection)+v)**2+\
                      (1./4)*(v.roll(2,spatialDirection)-4.*v.roll(1,spatialDirection)+3.*v)**2 
            for i in range(3):
                alpha[i]  = d_b[2][i]/((epsilon+beta[i])**2)
            omega = array(alpha)/(alpha[0]+alpha[1]+alpha[2])
            return  omega[0]*v1+omega[1]*v2+omega[2]*v3



def Compare2Arrays(A,B):
    return B+(A-B)*(A<B),B+(A-B)*(A>B)

def Compare3Arrays(A,B,C):
    return B+(A-B)*(A<B)+(C-(B+(A-B)*(A<B)))*(C<(B+(A-B)*(A<B))),B+(A-B)*(A>B)+(C-(B+(A-B)*(A>B)))*(C>(B+(A-B)*(A>B)))

def Extreme_quad(A,B,U_B,U_F):
    ###### Y = A*U*U + B*U  = A(U+B/2A)**2 - B**2/4A #####
    M = - B/(2.*A)
    RangeBEnd = (A*U_B+B)*U_B 
    RangeFEnd = (A*U_F+B)*U_F
    RangeM    =  -B*B/(4.*A) 
    InRangeMin,InRangeMax   = Compare3Arrays(RangeBEnd,RangeFEnd,RangeM)
    OutRangeMin,OutRangeMax = Compare2Arrays(RangeBEnd,RangeFEnd)
    return InRangeMin,InRangeMax,OutRangeMin,OutRangeMax

def Extreme_linear(B,U_B,U_F):
    ##### Y = B*U  #####
    RangeBEnd = B*U_B 
    RangeFEnd = B*U_F
    RangeMin,RangeMax = Compare2Arrays(RangeBEnd,RangeFEnd)
    return RangeMin,RangeMax


def TVD_3rdRK(y,t,h,derivs):
    alpha = [[1.],[3./4.,1./4.],[1./3.,0.,2./3.]]
    beta  = [[1.],[0.,1./4.],[0.,0.,2./3.]]
 
    l_0   = h*derivs(t,y)
    f_0   = alpha[0][0]*y + beta[0][0]*l_0
    l_1   = h*derivs(t,f_0)
    f_1   = alpha[1][0]*y + beta[1][0]*l_0 + alpha[1][1]*f_0 + beta[1][1]*l_1
    l_2   = h*derivs(t,f_1)
    f_2   = alpha[2][0]*y + beta[2][0]*l_0 + alpha[2][1]*f_0 + beta[2][1]*l_1 + alpha[2][2]*f_1 + beta[2][2]*l_2
    
    return f_2



def rkck(y,dydx,x,h,derivs):
    """
    4th Order Runge-Kutta Method, originally from Numerical Recipes, 3rd edition.
    Transcribed and modified to work in python.
    """
    a2, a3, a4, a5, a6= 0.2, 0.3, 0.6, 1.0, 0.875
    b21, b31, b32, b41, b42, b43 = 0.2, 3.0/40.0, 9.0/40.0, 0.3, -0.9, 1.2
    b51, b52, b53, b54 = -11.0/54.0, 2.5, -70.0/27.0, 35.0/27.0
    b61, b62, b63, b64, b65 = 1631.0/55296.0, 175.0/512.0, 575.0/13824.0, 44275.0/110592.0, 253.0/4096.0
    c1, c3, c4, c6 = 37.0/378.0, 250.0/621.0, 125.0/594.0, 512.0/1771.0
    dc1, dc3, dc4, dc5, dc6 = c1-2825.0/27648.0, c3-18575.0/48384.0, c4-13525.0/55296.0, -277.00/14336.0, c6-0.25
      
    ytemp =y + b21*h*dydx
    ak2 = derivs(x+a2*h,ytemp)
    ytemp=y+h*(b31*dydx+b32*ak2)
    ak3 = derivs(x+a3*h,ytemp)
    ytemp=y+h*(b41*dydx+b42*ak2+b43*ak3)
    ak4 = derivs(x+a4*h,ytemp)
    ytemp=y+h*(b51*dydx+b52*ak2+b53*ak3+b54*ak4)
    ak5 = derivs(x+a5*h,ytemp)
    ytemp=y+h*(b61*dydx+b62*ak2+b63*ak3+b64*ak4+b65*ak5)
    ak6 = derivs(x+a6*h,ytemp)
    yout=y+h*(c1*dydx+c3*ak3+c4*ak4+c6*ak6)
    yerr=h*(dc1*dydx+dc3*ak3+dc4*ak4+dc5*ak5+dc6*ak6)
    return yout,yerr


def rkqs(y,dydx,x,htry,eps,yscal,derivs,PIStepsizeControl=False,pre_errmax=None):
    """
    Adaptive time step 4th Order Runge-Kutta Method, originally from Numerical Recipes, 3rd edition.
    Transcribed and modified to work in python.
    """
    SAFETY, PGROW, PSHRNK, ERRCON = 0.9, -0.2, -0.25, 1.89e-4

    h = htry
    Flag = True
    while Flag ==True:
        ytemp,yerr = rkck(y,dydx,x,h,derivs)
        errmax = float(0.)
        errmax=max(errmax,(yerr/yscal).fabs().max())
        errmax = errmax/eps
        if (errmax <= 1.0):
            Flag = False
            break
        htemp=SAFETY*h*errmax**PSHRNK
        if h >= 0.0 :
            h = max(htemp,0.1*h)
        else:
            h = min(htemp,0.1*h)
        xnew=x+h
        if (xnew == x):
            print "stepsize underflow in rkqs"
            print "...now exiting to system..."
            sys.exit(1)
    if (errmax > ERRCON):
        if PIStepsizeControl == True:
	    k     = 5.
            beta  = 0.4/k
            alpha = 1./k-0.75*beta
            hnext = SAFETY*h*errmax**(-alpha)*pre_errmax**beta
            pre_errmax = errmax
        else:
            hnext=SAFETY*h*errmax**PGROW
    else:
        hnext=5.0*h
    hdid = h
    x += hdid
    y = ytemp
    if PIStepsizeControl == True:
    	return hdid,hnext,x,y,pre_errmax
    else:
        return hdid,hnext,x,y


"""
rollfield and Min/Max/MinMod functions for arrays and fields
"""
def rollfield(field, n, dir):
    result = field.__class__(field.gridShape, components=field.components)
    for component in field.components:
        result[component] = field[component].roll( n, dir)
    return result

def MinMod2(a,b):
    cond_a = (a>0) 
    cond_b = (b>0) 
    cond_ab = (a<b)
    cond_pos_ab = cond_a*cond_b
    cond_not_ab = (1-cond_a)*(1-cond_b)
    result = cond_pos_ab*(cond_ab*a + (1-cond_ab)*b) + cond_not_ab*((1-cond_ab)*a + cond_ab*b)
    return result

def MinMod3(a,b,c):
    cond_a = (a>0) 
    cond_b = (b>0) 
    cond_c = (c>0) 
    cond_ab = (a<b)
    cond_bc = (b<c)
    cond_ac = (a<c)
    cond_abc = cond_a*cond_b*cond_c
    cond_not_abc = (1-cond_a)*(1-cond_b)*(1-cond_c)
    result = cond_abc*(cond_ab*cond_ac*a + (1-cond_ab)*cond_bc*b + (1-cond_bc)*(1-cond_ac)*c) + \
            cond_not_abc*((1-cond_ab)*(1-cond_ac)*a + cond_ab*(1-cond_bc)*b + cond_bc*cond_ac*c)
    return result

def MinMod4(a,b,c,d):
    cond_a = (a>0) 
    cond_b = (b>0) 
    cond_c = (c>0) 
    cond_d = (d>0) 
    cond_ab = (a<b)
    cond_bc = (b<c)
    cond_ac = (a<c)
    cond_ad = (a<d)
    cond_bd = (b<d)
    cond_cd = (c<d)
    cond_abcd = cond_a*cond_b*cond_c*cond_d
    cond_not_abcd = (1-cond_a)*(1-cond_b)*(1-cond_c)*(1-cond_d)
    result = cond_abcd*(cond_ab*cond_ac*cond_ad*a + (1-cond_ab)*cond_bc*cond_bd*b + (1-cond_bc)*(1-cond_ac)*cond_cd*c + (1-cond_ad)*(1-cond_bd)*(1-cond_cd)*d) + \
            cond_not_abcd*((1-cond_ab)*(1-cond_ac)*(1-cond_ad)*a + cond_ab*(1-cond_bc)*(1-cond_bd)*b + cond_bc*cond_ac*(1-cond_cd)*c + cond_ad*cond_bd*cond_cd*d)
    return result

def MinMod2Field(a, b):
    result = a.__class__(a.gridShape, components=a.components)
    for component in a.components:
        result[component] = MinMod2(a[component], b[component])
    return result

def MinMod3Field(a, b, c):
    result = a.__class__(a.gridShape, components=a.components)
    for component in a.components:
        result[component] = MinMod3(a[component], b[component], c[component])
    return result

def MinMod4Field(a, b, c, d):
    result = a.__class__(a.gridShape, components=a.components)
    for component in a.components:
        result[component] = MinMod4(a[component], b[component], c[component], d[component])
    return result

def MinMax2(a,b):
    cond = a>b
    max2 = cond*a+(1-cond)*b
    min2 = (a+b)-max2
    return min2,max2

def Max2(a,b):
    cond = a>b
    return cond*a+(1-cond)*b
    
def Max3(a,b,c):
    cond_ab = (a>b)
    cond_bc = (b>c)
    cond_ac = (a>c)
    result = cond_ab*cond_ac*a + (1-cond_ab)*cond_bc*b + (1-cond_bc)*(1-cond_ac)*c
    return result
    
def Max5(a,b,c,d,e):
    return Max3(a,b,Max3(c,d,e))

def Max9(a,b,c,d,e,f,g,h,i):
    return Max5(a,b,c,d,Max5(e,f,g,h,i))
    
def Max3Field(a,b,c):
    result = a.__class__(a.gridShape, components=a.components)
    for component in a.components:
        if iterable(a):
            ca = a[component]
        else:
            ca = a
        if iterable(b):
            cb = b[component]
        else:
            cb = b
        if iterable(c):
            cc = c[component]
        else:
            cc = c
        result[component] = Max3(ca, cb, cc)
    return result
    
def Max5Field(a,b,c,d,e):
    return Max3Field(a,b,Max3Field(c,d,e))
    
def Max9Field(a,b,c,d,e,f,g,h,i):
    return Max5Field(a,b,c,d,Max5Field(e,f,g,h,i))

def Min3(a,b,c):
    cond_ab = (a<b)
    cond_bc = (b<c)
    cond_ac = (a<c)
    result = cond_ab*cond_ac*a + (1-cond_ab)*cond_bc*b + (1-cond_bc)*(1-cond_ac)*c
    return result
    
def Min5(a,b,c,d,e):
    return Min3(a,b,Min3(c,d,e))

def Min9(a,b,c,d,e,f,g,h,i):
    return Min5(a,b,c,d,Min5(e,f,g,h,i))

def Min3Field(a,b,c):
    result = a.__class__(a.gridShape, components=a.components)
    for component in a.components:
        if iterable(a):
            ca = a[component]
        else:
            ca = a
        if iterable(b):
            cb = b[component]
        else:
            cb = b
        if iterable(c):
            cc = c[component]
        else:
            cc = c
        result[component] = Min3(ca, cb, cc)
    return result

def Min5Field(a,b,c,d,e):
    return Min3Field(a,b,Min3Field(c,d,e))

def Min9Field(a,b,c,d,e,f,g,h,i):
    return Min5Field(a,b,c,d,Min5Field(e,f,g,h,i))
