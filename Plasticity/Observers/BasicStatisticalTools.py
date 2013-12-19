import numpy

import os

from scipy.cluster import vq 
from scipy.optimize import leastsq
from numpy import fft

from Plasticity.Constants import *
from Plasticity.FieldInitializers import FieldInitializer


def PowerlawFittingFromPoints(data,p0):
    if data.shape[1] != 2:
        raise ValueError, "The shape of the input data must be (N,2)!"
    residuals = lambda p,y,x: y-p[0]*numpy.power(x,p[1])
    plsq = leastsq(residuals, p0, args=(data[:,1],data[:,0]))
    print "The fitting parameters are:  C = ",plsq[0][0],"  alpha = ",plsq[0][1]
    return plsq[0] 

def BoxCountingMethod(data,cutoff):
    """
    This method works for self-similar patterns in 2D.
    
    Given a cutoff, all
    """
    N,M = data.shape
    if (N != M) :
        raise ValueError, 'Data should be a 2D square array.'
    boxsize = range(1,N/2+1) 
    result = numpy.zeros(len(boxsize),float) 
    result[0] = numpy.sum(data>=cutoff)
    for i in range(1,len(boxsize)):
        if (N%boxsize[i])==0:
            coarsedata = numpy.zeros((N/boxsize[i],N/boxsize[i]),float) 
            newdata = data
        else:
            coarsedata = numpy.zeros((N/boxsize[i]+1,N/boxsize[i]+1),float) 
            newN =(N/boxsize[i]+1)*boxsize[i]
            newdata = numpy.zeros((newN,newN),float)
            newdata[:N,:N] = data
            newdata[N:,:N] = data[:(newN-N),:N]
            newdata[:N,N:] = data[:N,:(newN-N)]
            newdata[N:,N:] = data[:(newN-N),:(newN-N)]
        for j in range(coarsedata.shape[0]):
            for k in range(coarsedata.shape[1]):
                nextj,nextk = (j+1)*boxsize[i],(k+1)*boxsize[i]
                coarsedata[j,k] = numpy.max(newdata[j*boxsize[i]:nextj,k*boxsize[i]:nextk])
        result[i] = numpy.sum(coarsedata>cutoff)
    boxsize = numpy.array(boxsize)
    return boxsize/float(N),result

def BoxCountingMethodwithManyCutoffs(data,cutoffs):
    results = []
    for i in range(len(cutoffs)):
        boxes,result = BoxCountingMethod(data,cutoffs[i]) 
        result *= boxes**2
        results.append(boxes)
        results.append(result)
    results = numpy.array(results).reshape(len(cutoffs),2,len(boxes))
    return results

def FittingPowerlawForBoxCountingMethod(data,boxmin=None,boxmax=None):
    if len(data.shape) > 2:
        pool = numpy.zeros((2,data.shape[0]*data.shape[-1]),float) 
        for i in range(data.shape[0]):
            pool[:,i*data.shape[-1]:(i+1)*data.shape[-1]] = data[i,:,:]
        x,y = pool[0],pool[1]
    else:
        x,y = data[0],data[1]
    newx = (x>=boxmin)*(x<=boxmax)*x
    fitx = x[newx.nonzero()]
    fity = y[newx.nonzero()]
    result = numpy.zeros((len(fitx),2),float)
    result[:,0] = fitx 
    result[:,1] = fity
    C,m = PowerlawFittingFromPoints(result,[1.,2.])
    return C, m

def FittingPowerLawForHistogram(histresult,xmax,xlow,xup):
    x,y = histresult[0], histresult[1]
    newx = (x>xlow*xmax)*(x<xup*xmax)*x
    fitx = x[newx.nonzero()]
    fity = y[newx.nonzero()]
    data = numpy.zeros((len(fitx),2),float)
    data[:,0] = fitx 
    data[:,1] = fity
    C,m = PowerlawFittingFromPoints(data,[1.,2.])
    return C,m 

def FittingPowerLawForCorrelationFunction(corrfunc,xlow,xup):
    x,y = corrfunc[0], corrfunc[1]
    newx = (x>xlow)*(x<xup)*x
    fitx = x[newx.nonzero()]
    fity = y[newx.nonzero()]
    data = numpy.zeros((len(fitx),2),float)
    data[:,0] = fitx 
    data[:,1] = fity
    C,m = PowerlawFittingFromPoints(data,[1.,2.])
    return C,m 

def SpatialCorrelationFunctionA(Field1,Field2):
    """
    Corr_12(r) = <Phi_1(r)Phi_2(0)>

    Corr_12(k) = Phi_1(k)* Phi_2(k)/V 
    """ 
    dim = len(Field1.shape)
    if dim == 1:
        V=float(Field1.shape[0])
    elif dim == 2:
        V=float(Field1.shape[0]*Field1.shape[1])
    elif dim == 3:
        V=float(Field1.shape[0]*Field1.shape[1]*Field1.shape[2])
    KField1 = fft.rfftn(Field1).conj()
    KField2 = fft.rfftn(Field2) 
    KCorr = KField1*KField2/V 
    Corr  = fft.irfftn(KCorr)
    return Corr

def SpatialCorrelationFunctionB(Field1,Field2):
    """
    Corr_12(r) = <(Phi_1(r)-Phi_2(0))^2> = <Phi_1^2> + <Phi_2^2> - 2<Phi_1(r)*Phi_2(0)>
    """ 
    meansquare1 = numpy.average(Field1*Field1)
    meansquare2 = numpy.average(Field2*Field2)
    return (meansquare1+meansquare2-2.*SpatialCorrelationFunctionA(Field1,Field2))


def RadialCorrFuncFromSpatialCorrFunc(Corr):
    """
    1D is trivial; so it works for 2D or higher D.
    """
    dim = len(Corr.shape)
    result = {}
    if dim == 2:
        N = Corr.shape[0]
        for i in range(-N/2,N/2):
            for j in range(-N/2,N/2): 
                r = i**2+j**2
                if result.has_key(r): 
                    result[r].append(Corr[i,j])
                else:
                    result[r]=[Corr[i,j]]
        rs = numpy.sort(result.keys())
        fs = numpy.zeros((len(result),),float)
        for i in range(len(rs)):
            fs[i] = numpy.average(result[rs[i]]) 
        return numpy.array([numpy.sqrt(rs)/float(N),fs]) 
    else:
        pass 

def CalculateCorrelationFunctions(field,fieldtype,corrfunctype='B'):
    if fieldtype == 'scaler':
        if corrfunctype == 'A':
            corr_func = SpatialCorrelationFunctionA(field,field)
        elif corrfunctype == 'B':
            corr_func = SpatialCorrelationFunctionB(field,field)
        else:
            pass
    elif fieldtype == 'vector':
        corr_func = 0
        for i in [x,y,z]:
            if corrfunctype == 'A':
                corr_func += SpatialCorrelationFunctionA(field[i],field[i])
            elif corrfunctype == 'B':
                corr_func += SpatialCorrelationFunctionB(field[i],field[i])
            else:
                pass
    elif fieldtype == 'tensor':
        corr_func = 0
        for i in [x,y,z]:
            for j in [x,y,z]:
                if corrfunctype == 'A':
                    corr_func += SpatialCorrelationFunctionA(field[i,j],field[i,j])
                elif corrfunctype == 'B':
                    corr_func += SpatialCorrelationFunctionB(field[i,j],field[i,j])
                else:
                    pass
    else:
        pass
    radial_corr_func = RadialCorrFuncFromSpatialCorrFunc(corr_func)
    return radial_corr_func
    


def Example():
    gridShape = (128,128)
    directory ='SampleFiles/' 
    file = directory + 'A2D128_D=0_5_betaP25_0.dat'
    figfile = directory+'A2D128_D=0_5_BCM_cutoffs.png'
    cutoff = [10.,20.,30.,40.,50.] 
    state = FieldInitializer.NumpyTensorInitializer(gridShape,file)
    rhoModulus = state.CalculateRhoFourier().modulus() 
    data = BoxCountingMethodwithManyCutoffs(rhoModulus,cutoff)
    marks = ['s','p','*','+','H','D','o','v','x']
    import pylab
    pylab.figure()
    for i in range(len(cutoff)):
        pylab.loglog(data[i,0,:],data[i,1,:],'r'+marks[i],label='cutoff = '+str(cutoff[i]))
    pylab.xlim(1.e-2,1.)
    pylab.ylim(0.1,1.e4)
    pylab.legend()
    pylab.xlabel(r"$\Delta x$",fontsize=20)
    pylab.ylabel(r"$N(\Delta x)$",fontsize=20)
    pylab.savefig(figfile)
    pylab.show()
        

if __name__=='__main__':
    Example()


