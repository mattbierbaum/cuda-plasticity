from numpy import iterable
import scipy.weave as W

import NumericalMethods
import GridArray

def BuildMacroAndLoop(dimension, spatialDirection, variables):
    macro = "\n#define v(dx) *(In"
    loophead = "int k = 0;\n"
    looptail = "k++;"
    for i in range(dimension):  
        if i == spatialDirection:
            xstr = "((x%d+dx+nx%d)%%nx%d)" % (i,i,i)
        else:
            xstr = "x%d" % i
        for j in range(i+1,dimension):
            xstr += "*nx%d" % j
        macro += "+"+xstr
        loophead += "for(int x%d = 0; x%d < nx%d; x%d++) {\n" % (i,i,i,i) 
        looptail += "\n}"
        variables.append("nx%d" % i)
    macro += ")\n"
    return macro, loophead, looptail

def SymmetricDerivative(In,gridShape,direction,order=1):
    """
    Calculate the symmetric numerical directional derivative of an array.

    direction refers to the axis that you want to take the derivative along.
    """
    result = GridArray.GridArray.empty(gridShape)
    dimension = len(gridShape)
    variables = ['In', 'result']
    macro, loophead, looptail = BuildMacroAndLoop(dimension, direction, variables)
    for i in range(dimension):
        exec("nx%d = In.shape[%d]" % (i,i))
    #nx, ny = gridShape

    if   order == 1:
        # This is general but runs a little bit slower due to % operations (0.12 vs 0.17) 
        factor = "double t = 0.5 * nx%d;\n" % direction
        # For rectangular simulations
        #factor = "double t = 0.5 * nx%d;\n" % dimension-1
        code = macro + factor + loophead + """
            *(result+k) = t * (v(1) - v(-1));
        """ + looptail
        W.inline(code, variables, extra_compile_args=["-w"])
        return result

        # this only works for 2 dimension and direction 0
        code = """
            int k = ny;
            double t = 0.5 * nx;
            for(int i=1; i<nx-1; i++, k += ny) {
                for(int j=0; j<ny; j++) {
                    *(result+k+j) = t * (*(In+k+ny+j) - *(In+k-ny+j));
                }
            }
            k = ny*(nx-1);
            // k = 0 case
            for(int j=0; j<ny; j++) {
                *(result+j) = t * (*(In+ny+j) - *(In+k+j));
            }
            for(int j=0; j<ny; j++) {
                *(result+k+j) = t * (*(In+j) - *(In+k-ny+j));
            }
        """
        W.inline(code, ['In', 'result', 'nx', 'ny']) 
        return result
    elif order == 2:
        # This is general but runs a little bit slower due to % operations (0.12 vs 0.17) 
        factor = "double t = (double)nx%d*nx%d;\n" % (direction, direction)
        # For rectangular simulations
        #factor = "double t = (double)nx%d*nx%d;\n" % (dimension-1, dimension-1)
        code = macro + factor + loophead + """
            *(result+k) = t * (v(1) - 2 * *(In+k) + v(-1));
        """ + looptail
        W.inline(code, variables, extra_compile_args=["-w"])
        return result

        # this only works for 2 dimension and direction 0
        code = """
            int k = ny;
            double nx2 = (double)nx*nx;
            for(int i=1; i<nx-1; i++, k += ny) {
                for(int j=0; j<ny; j++) {
                    *(result+k+j) = nx2 * (*(In+k+ny+j) - 2 * *(In+k+j) + *(In+k-ny+j));
                }
            }
            k = ny*(nx-1);
            // k = 0 case
            for(int j=0; j<ny; j++) {
                *(result+j) = nx2 * (*(In+ny+j) - 2 * *(In+j) + *(In+k+j));
            }
            for(int j=0; j<ny; j++) {
                *(result+k+j) = nx2 * (*(In+j) - 2 * *(In+k+j) + *(In+k-ny+j));
            }
        """
        W.inline(code, ['In', 'result', 'nx', 'ny']) 
        return result
    elif order == 4:
        # This is general but runs a little bit slower due to % operations (0.19 vs 0.23) 
        factor = "double t = (double)nx%d*nx%d*nx%d*nx%d;\n" % (direction, direction, direction, direction)
        # For rectangular simulations
        #factor = "double t = (double)nx%d*nx%d*nx%d*nx%d;\n" % (dimension-1, dimension-1, dimension-1, dimension-1)
        code = macro + factor + loophead + """
            *(result+k) = t * (v(2) - 4 * v(1) + 6 * *(In+k) - 4 * v(-1) + v(-2));
        """ + looptail
        W.inline(code, variables, extra_compile_args=["-w"])
        return result

        # this only works for 2 dimension and direction 0
        code = """
            int k = ny*2;
            double nx4 = (double)nx*nx*nx*nx;
            for(int i=2; i<nx-2; i++, k += ny) {
                for(int j=0; j<ny; j++) {
                    *(result+k+j) = nx4 * (*(In+k+2*ny+j) - 4 * *(In+k+ny+j) 
                         + 6 * *(In+k+j) - 4 * *(In+k-ny+j) + *(In+k-2*ny+j));
                }
            }
            k = ny*(nx-1);
            // k = 0 and k = 1case
            for(int j=0; j<ny; j++) {
                    *(result+j) = nx4 * (*(In+2*ny+j) - 4 * *(In+ny+j) 
                         + 6 * *(In+j) - 4 * *(In+k+j) + *(In+k-ny+j));
                    *(result+ny+j) = nx4 * (*(In+3*ny+j) - 4 * *(In+2*ny+j) 
                         + 6 * *(In+ny+j) - 4 * *(In+j) + *(In+k+j));
            }
            // k = ny* (nx-1) and (nx-2) cases
            for(int j=0; j<ny; j++) {
                    *(result+k+j) = nx4 * (*(In+ny+j) - 4 * *(In+j) 
                         + 6 * *(In+k+j) - 4 * *(In+k-ny+j) + *(In+k-2*ny+j));
                    *(result+k-ny+j) = nx4 * (*(In+j) - 4 * *(In+k+j) 
                         + 6 * *(In+k-ny+j) - 4 * *(In+k-2*ny+j) + *(In+k-3*ny+j));
            }
        """
        W.inline(code, ['In', 'result', 'nx', 'ny']) 
        return result
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
    #if dimension not in range(3):
        #raise ValueError, "Test values only"
    if flowDirection not in [1,-1]:
        raise ValueError, "Flow direction must be either +1 or -1"
    if spatialDirection not in range(dimension): 
        raise ValueError, "Spatial direction must be less than the dimension"

    fD = flowDirection
    variables = ['In', 'result', 'fD']
    macro, loophead, looptail = BuildMacroAndLoop(dimension, spatialDirection, variables)
    for i in range(dimension):
        exec("nx%d = In.shape[%d]" % (i,i))

    result = GridArray.GridArray.empty(In.shape)

    if order == 1:
        code = macro + loophead + """
                    double ndd1 = v(fD) - *(In+k);
                    *(result+k) = fD*ndd1;
        """ + looptail
        W.inline(code, variables, extra_compile_args=["-w"])
        return result
    
    if order == 2:
        code = macro + loophead + """
                    double ndd1 = v(fD) - *(In+k);
                    double ndda = (v(2*fD) - 2*v(fD) + *(In+k))/2;
                    double nddb = (v(fD) - 2* *(In+k) + v(-fD))/2;
                    if (fabs(ndda) > fabs(nddb)) {
                        *(result+k) = fD*(ndd1 - nddb);
                    } else {
                        *(result+k) = fD*(ndd1 - ndda);
                    }
        """ + looptail
        W.inline(code, variables, extra_compile_args=["-w"])
        return result
    if order == 3:
        code = macro + loophead + """
                    double ndd1 = v(fD) - *(In+k);
                    // 2nd order calculated
                    double ndda = (v(2*fD) - 2*v(fD) + *(In+k))/2;
                    double nddb = (v(fD) - 2* *(In+k) + v(-fD))/2;
                    if (fabs(ndda) > fabs(nddb)) {
                        *(result+k) = fD*(ndd1 - nddb);
                        // 3rd order
                        ndda = (ndda-nddb)/3; 
                        nddb = (v(fD) - 3**(In+k) + 3*v(-fD) - v(-2*fD))/6; 
                        // add 3rd order
                        if (fabs(ndda) > fabs(nddb)) {
                            *(result+k) -= fD*nddb;
                        } else {
                            *(result+k) -= fD*ndda;
                        }
                    } else {
                        *(result+k) = fD*(ndd1 - ndda);
                        nddb = (ndda-nddb)/3; 
                        ndda = (v(3*fD) - 3*v(2*fD) + 3*v(fD) - *(In+k))/6; 
                        // add 3rd order
                        if (fabs(ndda) > fabs(nddb)) {
                            *(result+k) += 2*fD*nddb;
                        } else {
                            *(result+k) += 2*fD*ndda;
                        }
                    }
        """ + looptail
        W.inline(code, variables, extra_compile_args=["-w"])
        return result

        """
        This version is a little bit faster
        But seems to have a bug (where?)
        """
        code = macro + """
            for(int x0 = 0, k = 0; x0 < nx0; x0++) {
                double prev_ndd1 = *(In+k) - v(x0,-fD);
                double next_ndd1 = v(x0,fD) - *(In+k);;
                for(int x1 = 0; x1 < nx1; x1++, k++) {
                    double ndd1 = next_ndd1;
                    next_ndd1 = v(2*fD) - v(fD);
                    // 2nd order calculated
                    double ndda = (next_ndd1 - ndd1)/2;
                    double nddb = (ndd1 - prev_ndd1)/2;
                    if (fabs(ndda) > fabs(nddb)) {
                        *(result+k) = fD*(ndd1 - nddb);
                        // 3rd order
                        ndda = (ndda-nddb)/3;
                        //nddb = (nddb-(prev_ndd1 - (v(-fD)-v(-2*fD)))/2)/3; 
                        nddb = (v(fD) - 3**(In+k) + 3*v(-fD) - v(-2*fD))/6; 
                        // add 3rd order
                        if (fabs(ndda) > fabs(nddb)) {
                            *(result+k) -= fD*nddb;
                        } else {
                            *(result+k) -= fD*ndda;
                        }
                        prev_ndd1 = ndd1;
                    } else {
                        *(result+k) = fD*(ndd1 - ndda);
                        nddb = (ndda-nddb)/3; 
                        ndda = (v(3*fD) - 3*v(2*fD) + 3*v(fD) - *(In+k))/6; 
                        //ndda = (((v(3*fD)-v(2*fD))-next_ndd1)/2 - ndda)/3; 
                        // add 3rd order
                        if (fabs(ndda) > fabs(nddb)) {
                            *(result+k) += 2*fD*nddb;
                        } else {
                            *(result+k) += 2*fD*ndda;
                        }
                        prev_ndd1 = ndd1;
                    }
                }
            }
        """
        W.inline(code, variables, extra_compile_args=["-w"])
        return result


    if order == 4:
        code = macro + loophead + """
                    double ndd1 = v(fD) - *(In+k);
                    // 2nd order calculated
                    double ndda = (v(2*fD) - 2*v(fD) + *(In+k))/2;
                    double nddb = (v(fD) - 2* *(In+k) + v(-fD))/2;
                    if (fabs(ndda) > fabs(nddb)) {
                        *(result+k) = fD*(ndd1 - nddb);
                        // 3rd order
                        ndda = (ndda-nddb)/3;
                        nddb = (v(fD) - 3**(In+k) + 3*v(-fD) - v(-2*fD))/6; 
                        // add 3rd order
                        if (fabs(ndda) > fabs(nddb)) {
                            *(result+k) -= fD*nddb;
                            ndda = (ndda-nddb)/4;
                            nddb = (v(fD) - 4**(In+k) + 6*v(-fD) - 4*v(-2*fD) + v(-3*fD))/24;
                            if (fabs(ndda) > fabs(nddb)) {
                                *(result+k) += -2*fD*nddb;
                            } else {
                                *(result+k) += -2*fD*ndda;
                            }
                        } else {
                            *(result+k) -= fD*ndda;
                            nddb = (ndda-nddb)/4;
                            ndda = (v(3*fD) - 4*v(2*fD) + 6*v(fD) - 4**(In+k) + v(-fD))/24;
                            if (fabs(ndda) > fabs(nddb)) {
                                *(result+k) += 2*fD*nddb;
                            } else {
                                *(result+k) += 2*fD*ndda;
                            }
                        }
                    } else {
                        *(result+k) = fD*(ndd1 - ndda);
                        nddb = (ndda-nddb)/3; 
                        ndda = (v(3*fD) - 3*v(2*fD) + 3*v(fD) - *(In+k))/6; 
                        // add 3rd order
                        if (fabs(ndda) > fabs(nddb)) {
                            *(result+k) += 2*fD*nddb;
                            ndda = (ndda-nddb)/4;
                            nddb = (v(2*fD) - 4*v(fD) + 6**(In+k) - 4*v(-fD) + v(-2*fD))/24;
                            if (fabs(ndda) > fabs(nddb)) {
                                *(result+k) += 2*fD*nddb;
                            } else {
                                *(result+k) += 2*fD*ndda;
                            }
                        } else {
                            *(result+k) += 2*fD*ndda;
                            nddb = (ndda-nddb)/4;
                            ndda = (v(4*fD) - 4*v(3*fD) + 6*v(2*fD) - 4*v(fD) + *(In+k))/24;
                            if (fabs(ndda) > fabs(nddb)) {
                                *(result+k) += -6*fD*nddb;
                            } else {
                                *(result+k) += -6*fD*ndda;
                            }
                        }
                    }
        """ + looptail
        W.inline(code, variables, extra_compile_args=["-w"])

        return result

    if order == 5:
        code = macro + loophead + """
                    double ndd1 = v(fD) - *(In+k);
                    // 2nd order calculated
                    double ndda = (v(2*fD) - 2*v(fD) + *(In+k))/2;
                    double nddb = (v(fD) - 2* *(In+k) + v(-fD))/2;
                    if (fabs(ndda) > fabs(nddb)) {
                        *(result+k) = fD*(ndd1 - nddb);
                        // 3rd order
                        ndda = (ndda-nddb)/3;
                        nddb = (v(fD) - 3**(In+k) + 3*v(-fD) - v(-2*fD))/6; 
                        // add 3rd order
                        if (fabs(ndda) > fabs(nddb)) {
                            *(result+k) -= fD*nddb;
                            ndda = (ndda-nddb)/4;
                            nddb = (v(fD) - 4**(In+k) + 6*v(-fD) - 4*v(-2*fD) + v(-3*fD))/24;
                            if (fabs(ndda) > fabs(nddb)) {
                                *(result+k) += -2*fD*nddb;
                                ndda = (ndda-nddb)/5;
                                nddb = (v(fD) - 5**(In+k) + 10*v(-fD) - 10*v(-2*fD) + 5*v(-3*fD) - v(-4*fD))/120;
                                if (fabs(ndda) > fabs(nddb)) {
                                    *(result+k) += -6*fD*nddb;
                                } else {
                                    *(result+k) += -6*fD*ndda;
                                }
                            } else {
                                *(result+k) += -2*fD*ndda;
                                nddb = (ndda-nddb)/5;
                                ndda = (v(3*fD) - 5*v(2*fD) + 10*v(fD) - 10**(In+k) + 5*v(-fD) - v(-2*fD))/120;
                                if (fabs(ndda) > fabs(nddb)) {
                                    *(result+k) += 4*fD*nddb;
                                } else {
                                    *(result+k) += 4*fD*ndda;
                                }
                            }
                        } else {
                            *(result+k) -= fD*ndda;
                            nddb = (ndda-nddb)/4;
                            ndda = (v(3*fD) - 4*v(2*fD) + 6*v(fD) - 4**(In+k) + v(-fD))/24;
                            if (fabs(ndda) > fabs(nddb)) {
                                *(result+k) += 2*fD*nddb;
                                ndda = (ndda-nddb)/5;
                                nddb = (v(2*fD) - 5*v(fD) + 10**(In+k) - 10*v(-fD) + 5*v(-2*fD) - v(-3*fD))/120;
                                if (fabs(ndda) > fabs(nddb)) {
                                    *(result+k) += 4*fD*nddb;
                                } else {
                                    *(result+k) += 4*fD*ndda;
                                }
                            } else {
                                *(result+k) += 2*fD*ndda;
                                nddb = (ndda-nddb)/5;
                                ndda = (v(4*fD) - 5*v(3*fD) + 10*v(2*fD) - 10*v(fD) + 5**(In+k) - v(-fD))/120;
                                if (fabs(ndda) > fabs(nddb)) {
                                    *(result+k) += -6*fD*nddb;
                                } else {
                                    *(result+k) += -6*fD*ndda;
                                }
                            }
                        }
                    } else {
                        *(result+k) = fD*(ndd1 - ndda);
                        nddb = (ndda-nddb)/3; 
                        ndda = (v(3*fD) - 3*v(2*fD) + 3*v(fD) - *(In+k))/6; 
                        // add 3rd order
                        if (fabs(ndda) > fabs(nddb)) {
                            *(result+k) += 2*fD*nddb;
                            ndda = (ndda-nddb)/4;
                            nddb = (v(2*fD) - 4*v(fD) + 6**(In+k) - 4*v(-fD) + v(-2*fD))/24;
                            if (fabs(ndda) > fabs(nddb)) {
                                *(result+k) += 2*fD*nddb;
                                ndda = (ndda-nddb)/5;
                                nddb = (v(2*fD) - 5*v(fD) + 10**(In+k) - 10*v(-fD) + 5*v(-2*fD) - v(-3*fD))/120;
                                if (fabs(ndda) > fabs(nddb)) {
                                    *(result+k) += 4*fD*nddb;
                                } else {
                                    *(result+k) += 4*fD*ndda;
                                }
                            } else {
                                *(result+k) += 2*fD*ndda;
                                nddb = (ndda-nddb)/5;
                                ndda = (v(4*fD) - 5*v(3*fD) + 10*v(2*fD) - 10*v(fD) + 5**(In+k) - v(-fD))/120;
                                if (fabs(ndda) > fabs(nddb)) {
                                    *(result+k) += -6*fD*nddb;
                                } else {
                                    *(result+k) += -6*fD*ndda;
                                }
                            }
                        } else {
                            *(result+k) += 2*fD*ndda;
                            nddb = (ndda-nddb)/4;
                            ndda = (v(4*fD) - 4*v(3*fD) + 6*v(2*fD) - 4*v(fD) + *(In+k))/24;
                            if (fabs(ndda) > fabs(nddb)) {
                                *(result+k) += -6*fD*nddb;
                                ndda = (ndda-nddb)/5;
                                nddb = (v(3*fD) - 5*v(2*fD) + 10*v(fD) - 10**(In+k) + 5*v(-fD) - v(-2*fD))/120;
                                if (fabs(ndda) > fabs(nddb)) {
                                    *(result+k) += -6*fD*nddb;
                                } else {
                                    *(result+k) += -6*fD*ndda;
                                }
                            } else {
                                *(result+k) += -6*fD*ndda;
                                nddb = (ndda-nddb)/5;
                                ndda = (v(5*fD) - 5*v(4*fD) + 10*v(3*fD) - 10*v(2*fD) + 5*v(fD) - *(In+k))/120;
                                if (fabs(ndda) > fabs(nddb)) {
                                    *(result+k) += 24*fD*nddb;
                                } else {
                                    *(result+k) += 24*fD*ndda;
                                }
                            }
                        }
                    }
        """ + looptail
        W.inline(code, variables, extra_compile_args=["-w"])

        return result


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


def TVD_3rdRK(y,hj,order,dt,l=None,f=None):
    alpha = [[1.],[3./4.,1./4.],[1./3.,0.,2./3.]]
    beta  = [[1.],[0.,1./4.],[0.,0.,2./3.]]
    if   order == 1:
        l_0   = -dt*hj
        f_0   = alpha[0][0]*y + beta[0][0]*l_0
        return l_0,f_0
    elif order == 2:
        l_1   = -dt*hj
        f_1   = alpha[1][0]*y + beta[1][0]*l[0] + alpha[1][1]*f[0] + beta[1][1]*l_1
        return [l[0],l_1],[f[0],f_1]
    elif order == 3:
        l_2   = -dt*hj
        f_2   = alpha[2][0]*y + beta[2][0]*l[0] + alpha[2][1]*f[0] + beta[2][1]*l[1] + alpha[2][2]*f[1] + beta[2][2]*l_2
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


def rkqs(y,dydx,x,htry,eps,yscal,derivs):
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
        hnext=SAFETY*h*errmax**PGROW
    else:
        hnext=5.0*h
    hdid = h
    x += hdid
    y = ytemp
    return hdid,hnext,x,y

NumericalMethods.SymmetricDerivative = SymmetricDerivative
NumericalMethods.ENO_Derivative = ENO_Derivative

def Max2(a,b):
    code = """
    for(int i=0;i<ntot;i++) {
        double av = *(a+i);
    """
    if iterable(b):
        code += """
        double bv = *(b+i);
        """
    else:
        code += """
        double bv = b;
        """
    code += """
        *(result+i) = (av>bv) ? av : bv;
    }
    """
    ntot = a.size
    result = GridArray.GridArray.empty(a.shape)
    W.inline(code, ['a','b','result','ntot'], extra_compile_args=["-w"])
    return result
         
def Max3(a,b,c):
    code = """
    for(int i=0;i<ntot;i++) {
        double av = *(a+i);
    """
    if iterable(b):
        code += """
        double bv = *(b+i);
        """
    else:
        code += """
        double bv = b;
        """
    if iterable(c):
        code += """
        double cv = *(c+i);
        """
    else:
        code += """
        double cv = c;
        """
    code += """
        *(result+i) = (av>bv) ? ((cv>av) ? cv:av) : ((cv>bv) ? cv:bv);
    }
    """
    ntot = a.size
    result = GridArray.GridArray.empty(a.shape)
    W.inline(code, ['a','b','c','result','ntot'], extra_compile_args=["-w"])
    return result

def Max5(a,b,c,d,e):
    code = """
    for(int i=0;i<ntot;i++) {
        double av = *(a+i);
    """
    if iterable(b):
        code += """
        double bv = *(b+i);
        """
    else:
        code += """
        double bv = b;
        """
    if iterable(c):
        code += """
        double cv = *(c+i);
        """
    else:
        code += """
        double cv = c;
        """
    if iterable(d):
        code += """
        double dv = *(d+i);
        """
    else:
        code += """
        double dv = d;
        """
    if iterable(e):
        code += """
        double ev = *(e+i);
        """
    else:
        code += """
        double ev = e;
        """
    code += """
        double abcm = (av>bv) ? ((cv>av) ? cv:av) : ((cv>bv) ? cv:bv);
        *(result+i) = (abcm>dv) ? ((ev>abcm) ? ev:abcm) : ((ev>dv) ? ev:dv);
    }
    """
    ntot = a.size
    result = GridArray.GridArray.empty(a.shape)
    W.inline(code, ['a','b','c','d','e','result','ntot'], extra_compile_args=["-w"])
    return result

def Min2(a,b):
    code = """
    for(int i=0;i<ntot;i++) {
        double av = *(a+i);
    """
    if iterable(b):
        code += """
        double bv = *(b+i);
        """
    else:
        code += """
        double bv = b;
        """
    code += """
        *(result+i) = (av<bv) ? av : bv;
    }
    """
    ntot = a.size
    result = GridArray.GridArray.empty(a.shape)
    W.inline(code, ['a','b','result','ntot'], extra_compile_args=["-w"])
    return result
         
def Min3(a,b,c):
    code = """
    for(int i=0;i<ntot;i++) {
        double av = *(a+i);
    """
    if iterable(b):
        code += """
        double bv = *(b+i);
        """
    else:
        code += """
        double bv = b;
        """
    if iterable(c):
        code += """
        double cv = *(c+i);
        """
    else:
        code += """
        double cv = c;
        """
    code += """
        *(result+i) = (av<bv) ? ((cv<av) ? cv:av) : ((cv<bv) ? cv:bv);
    }
    """
    ntot = a.size
    result = GridArray.GridArray.empty(a.shape)
    W.inline(code, ['a','b','c','result','ntot'], extra_compile_args=["-w"])
    return result

def Min5(a,b,c,d,e):
    code = """
    for(int i=0;i<ntot;i++) {
        double av = *(a+i);
    """
    if iterable(b):
        code += """
        double bv = *(b+i);
        """
    else:
        code += """
        double bv = b;
        """
    if iterable(c):
        code += """
        double cv = *(c+i);
        """
    else:
        code += """
        double cv = c;
        """
    if iterable(d):
        code += """
        double dv = *(d+i);
        """
    else:
        code += """
        double dv = d;
        """
    if iterable(e):
        code += """
        double ev = *(e+i);
        """
    else:
        code += """
        double ev = e;
        """
    code += """
        double abcm = (av<bv) ? ((cv<av) ? cv:av) : ((cv<bv) ? cv:bv);
        *(result+i) = (abcm<dv) ? ((ev<abcm) ? ev:abcm) : ((ev<dv) ? ev:dv);
    }
    """
    ntot = a.size
    result = GridArray.GridArray.empty(a.shape)
    W.inline(code, ['a','b','c','d','e','result','ntot'], extra_compile_args=["-w"])
    return result

def MinMod2(a,b):
    code = """
    for(int i=0;i<ntot;i++) {
        double av = *(a+i);
    """
    if iterable(b):
        code += """
        double bv = *(b+i);
        """
    else:
        code += """
        double bv = b;
        """
    code += """
        *(result+i) = (av*bv>0) ? 
                            (av>0 ? ((av<bv) ? av : bv) 
                                  : ((av<bv) ? bv : av))
                            : 0;
    }
    """
    ntot = a.size
    result = GridArray.GridArray.empty(a.shape)
    W.inline(code, ['a','b','result','ntot'], extra_compile_args=["-w"])
    return result
         
def MinMod3(a,b,c):
    code = """
#define max2(a,b) ((a>b) ? a:b)
#define min2(a,b) ((a<b) ? a:b)
    for(int i=0;i<ntot;i++) {
        double av = *(a+i);
    """
    if iterable(b):
        code += """
        double bv = *(b+i);
        """
    else:
        code += """
        double bv = b;
        """
    if iterable(c):
        code += """
        double cv = *(c+i);
        """
    else:
        code += """
        double cv = c;
        """
    code += """
        double r = 0;
        if ((av>0) && (bv>0) && (cv>0)) {
            r = min2(av,min2(bv,cv));
        } else {
            if ((av<0) && (bv<0) && (cv<0)) {
                r = max2(av,max2(bv,cv));
            }
        } 
        *(result+i) = r; 
    }
    """
    ntot = a.size
    result = GridArray.GridArray.empty(a.shape)
    W.inline(code, ['a','b','c','result','ntot'], extra_compile_args=["-w"])
    return result

def MinMod4(a,b,c,d):
    code = """
#define max2(a,b) ((a>b) ? a:b)
#define min2(a,b) ((a<b) ? a:b)
    for(int i=0;i<ntot;i++) {
        double av = *(a+i);
    """
    if iterable(b):
        code += """
        double bv = *(b+i);
        """
    else:
        code += """
        double bv = b;
        """
    if iterable(c):
        code += """
        double cv = *(c+i);
        """
    else:
        code += """
        double cv = c;
        """
    if iterable(d):
        code += """
        double dv = *(d+i);
        """
    else:
        code += """
        double dv = d;
        """
    code += """
        double r = 0;
        if ((av>0) && (bv>0) && (cv>0) && (dv>0)) {
            r = min2(av,min2(bv,min2(cv,dv)));
        } else {
            if ((av<0) && (bv<0) && (cv<0) && (dv<0)) {
                r = max2(av,max2(bv,max2(cv,dv)));
            }
        } 
        *(result+i) = r; 
    }
    """
    ntot = a.size
    result = GridArray.GridArray.empty(a.shape)
    W.inline(code, ['a','b','c','d','result','ntot'], extra_compile_args=["-w"])
    return result

NumericalMethods.Max2 = Max2
NumericalMethods.Max3 = Max3
NumericalMethods.Max5 = Max5
NumericalMethods.Min2 = Min2
NumericalMethods.Min3 = Min3
NumericalMethods.Min5 = Min5
NumericalMethods.MinMod2 = MinMod2
NumericalMethods.MinMod3 = MinMod3
NumericalMethods.MinMod4 = MinMod4

