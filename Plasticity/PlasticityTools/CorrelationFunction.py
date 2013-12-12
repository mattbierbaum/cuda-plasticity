from numpy import fft
import numpy

import scipy.weave as W
import sys
import getopt

def SpatialCorrelationFunction(Field1,Field2):
    """
	Designed for Periodic Boundary Condition.
	
	.. math::
		C_{12}(r) = <F_1(r) F_2(0)>
	
		C_{12}(k) = F_1(k)* F_2(k)/V 
    """ 
    V = float(numpy.array(Field1.shape).prod())
    KField1 = fft.rfftn(Field1).conj()
    KField2 = fft.rfftn(Field2) 
    KCorr = KField1*KField2/V 
    Corr  = fft.irfftn(KCorr)
    return Corr

def SpatialCorrelationFunction_HeightHeight(Field1,Field2):
    """
	Designed for Periodic Boundary Condition.
	
	.. math::
		C_{12}(r) = <(F_1(r)-F_1(0))(F_2(r)-F_2(0))> \\
	   	             = 2<F_1*F_2> - <F_1(r)*F_2(0)> - <F_2(r)*F_1(0)>
    """ 
    meansquare = numpy.average(Field1*Field2)
    Corr = 2*meansquare-SpatialCorrelationFunctionA(Field1,Field2)-SpatialCorrelationFunctionA(Field2,Field1)
    return Corr

def SpatialCorrelationFunction_ZeroAverage(Field1,Field2):
    """
	Designed for Periodic Boundary Condition.
	
	.. math::
		C_{12}(r) = <(F_1(r)-<F_1>)(F_2(0)-<F_2>)>
	
    """ 
    meansquare1 = numpy.average(Field1)
    meansquare2 = numpy.average(Field2)
    Corr = SpatialCorrelationFunctionA(Field1-meansquare1,Field2-meansquare2)
    return Corr 

def RadialCorrFuncFromSpatialCorrFuncFast(corr):
    """ 
	Averages angle dependence in correlation function assuming
	spherical symmetry (in the appropriate dimension). This version
	uses weave to increase speed. 
	
	.. math::
		C_R(r) = \int_{\Omega} C_R(r,\\theta,\phi) d\Omega
	
    """

    dim = len(corr.shape)
    N = corr.shape[0]
    rcorr = numpy.zeros((numpy.array(corr.shape).prod(),2),float)
    code = """
        #define clip(x) (((x)+%(N)d)%%%(N)d)
        #define pclip(x) ((clip(x)>%(N)d/2) ? %(N)d-clip(x) : clip(x))
        #define square(x) ((x)*(x))

        int n = %(N)d;
       
        if (dim == 1) {
            for(int i=0; i<n; i++){
                *(rcorr+i*2) = square(pclip(i));
                *(rcorr+i*2+1) = *(corr+i); 
            }
        }
        else if (dim == 2) {
            for(int i=0; i<n; i++){
                for(int j=0; j<n; j++){
                    int index = i*n + j;
                    *(rcorr+index*2) = square(pclip(i))+square(pclip(j));
                    *(rcorr+index*2+1) = *(corr+index); 
                }
            }
        }
        else {
            for(int i=0; i<n; i++){
                for(int j=0; j<n; j++){
                    for(int k=0; k<n; k++){
                        int index = i*n*n+j*n+k;
                        *(rcorr+index*2) = square(pclip(i))+square(pclip(j))+square(pclip(k));
                        *(rcorr+index*2+1) = *(corr+index); 
                    }
                }
            }
        }
    """ % {'N' : N}
    variables = ['dim','corr', 'rcorr']
    W.inline(code, variables, extra_compile_args=["-w"])
    ordered_corr = numpy.array(sorted(rcorr, key=lambda item: item[0]))
    size = ordered_corr.shape[0]
    result = numpy.zeros(ordered_corr.shape,float) 
    rearrange = """
        *(result) = *(ordered_corr);
        *(result+1) = *(ordered_corr+1); 
        int K = 0;
        int count = 1;
        for (int i=0; i<%(size)d-1; i++) {
            if (*(ordered_corr+2*i+2) == *(result+2*K)) {
                *(result+2*K+1) += *(ordered_corr+2*i+3);
                count++;
                if (i==%(size)d-2)
                    *(result+2*K+1) /= double(count);
            }
            else {
                *(result+2*K+1) /= double(count);
                count = 1;
                K++;
                *(result+2*K) = *(ordered_corr+2*i+2);
                *(result+2*K+1) = *(ordered_corr+2*i+3);
            }
        }
        return_val = PyFloat_FromDouble(K+1);
    """ % {'size' :size}
    K = int(W.inline(rearrange, ['ordered_corr','result'], extra_compile_args=["-w"]) )
    return numpy.array([numpy.sqrt(result[1:K,0])/float(N),result[1:K,1]]) 

def RadialCorrFuncFromSpatialCorrFunc(Corr):
    """ 	
	Averages angle dependence in correlation function assuming
	spherical symmetry (in the appropriate dimension).  
	
	.. math::
		C_R(r) = \int_{\Omega} C_R(r,\\theta,\phi) d\Omega

    """

    dim = len(Corr.shape)
    N = Corr.shape[0]
    result = {}
    if dim == 1:
        for i in range(-N/2,N/2):
            r = i**2
            if result.has_key(r): 
                result[r].append(Corr[i])
            else:
                result[r]=[Corr[i]]
    elif dim == 2:
        for i in range(-N/2,N/2):
            for j in range(-N/2,N/2): 
                r = i**2+j**2
                if result.has_key(r): 
                    result[r].append(Corr[i,j])
                else:
                    result[r]=[Corr[i,j]]
    else:
        for i in range(-N/2,N/2):
            for j in range(-N/2,N/2): 
                for k in range(-N/2,N/2): 
                    r = i**2+j**2+k**2
                    if result.has_key(r): 
                        result[r].append(Corr[i,j,k])
                    else:
                        result[r]=[Corr[i,j,k]]
    rs = numpy.sort(result.keys())
    fs = numpy.zeros((len(result),),float)
    for i in range(len(rs)):
        fs[i] = numpy.average(result[rs[i]]) 
    return numpy.array([numpy.sqrt(rs[1:])/float(N),fs[1:]]) 

def BinData_Log(x,y,num_of_bin=50,vmin=None,vmax=None):
    """
	Takes log-space histogram of data, dividing x into equally
	spaced bins in log space and average the y values inside 
	every bin.
    """
    if vmin is None:
        vmin = numpy.floor(numpy.log10(x.min()))
    if vmax is None:
        vmax = 0.
    bins = numpy.logspace(vmin,vmax,num_of_bin)
    xs,ys = [],[]
    for i in range(num_of_bin):
        if i == num_of_bin-1:
            newx = x[(x>=bins[-1]).nonzero()] 
            newy = y[(x>=bins[-1]).nonzero()]
        else:
            newx = x[((x<bins[i+1])*(x>=bins[i])).nonzero()] 
            newy = y[((x<bins[i+1])*(x>=bins[i])).nonzero()]
        if len(newx)!= 0:
            xs.append(newx.mean())
            ys.append(newy.mean()) 
    return numpy.array([xs,ys])

def BinData_Linear(x,y,num_of_bin=100,vmin=None,vmax=None):
    if vmin is None:
        vmin = 0. 
    if vmax is None:
        vmax = 1.
    bins = numpy.linspace(vmin,vmax,num_of_bin)
    xs,ys = [],[]
    for i in range(num_of_bin):
        if i == num_of_bin-1:
            newx = x[(x>=bins[-1]).nonzero()] 
            newy = y[(x>=bins[-1]).nonzero()]
        else:
            newx = x[((x<bins[i+1])*(x>=bins[i])).nonzero()] 
            newy = y[((x<bins[i+1])*(x>=bins[i])).nonzero()]
        if len(newx)!= 0:
            xs.append(newx.mean())
            ys.append(newy.mean()) 
    return numpy.array([xs,ys])



def CorrelationFunctionsOfScalarField(field,type='default'):
    """
	Return the correlation function of variety 'type'.  Can be in:
	    'default':        typical autocorrelation function from convolution
	    'height-height':  height-height correlation function
	    'zero-average':   remove any non-zero offset from all data
	                      then take default autocorrelation
    """
    if type == 'default':
        corr_func = SpatialCorrelationFunction(field,field) 
    elif type == 'height-height':
        corr_func = SpatialCorrelationFunction_HeightHeight(field,field) 
    elif type == 'zero-average':
        corr_func = SpatialCorrelationFunction_ZeroAverage(field,field) 
    else:
        print "Check the available type options with help"
        corr_func = None
    return corr_func

def CorrelationFunctionsOfVectorField(field,type='default'):
    """
    Return the correlation function of variety 'type' for a 
    three component vector field, adding all components together. 
    'type' can be one of:
        * 'default':        typical autocorrelation function from convolution
        * 'height-height':  height-height correlation function
        * 'zero-average':   remove any non-zero offset from all data
                          then take default autocorrelation
    """
    sets = [0,1,2]
    corr_func = 0
    for i in sets:
        if type == 'default':
            corr_func += SpatialCorrelationFunction(field[i],field[i]) 
        elif type == 'height-height':
            corr_func += SpatialCorrelationFunction_HeightHeight(field[i],field[i]) 
        elif type == 'zero-average':
            corr_func += SpatialCorrelationFunction_ZeroAverage(field[i],field[i]) 
        else:
            print "Check the available type options with help"
            corr_func = None
    return corr_func

def CorrelationFunctionsOfTensorField(field, type='default', option='total'):
    """
	Return the correlation function of variety 'type' for a 
	3x3 component tensor field, adding all components together. 
	'type' can be one of:
	    * 'default'      :  typical autocorrelation function from convolution
	    * 'height-height':  height-height correlation function
	    * 'zero-average' :  remove any non-zero offset from all data
						  then take default autocorrelation
	
	The argument 'option' specifies the symmetry group of the correlation function, being:
	    * 'total'       :  :math:`C(A_{ij}, A_{ij})`
	    * 'permutation' :  :math:`C(A_{ij}, A_{ji})`
	    * 'trace'       :  :math:`C(A_{ii}, A_{jj})`
    """
    
    sets = [0,1,2]
    if option == 'total':
        corr_func = 0
        for i in sets:
            for j in sets:
                if type == 'default':
                    corr_func += SpatialCorrelationFunction(field[i,j],field[i,j]) 
                elif type == 'height-height':
                    corr_func += SpatialCorrelationFunction_HeightHeight(field[i,j],field[i,j]) 
                elif type == 'zero-average':
                    corr_func += SpatialCorrelationFunction_ZeroAverage(field[i,j],field[i,j]) 
                else:
                    pass
    elif option == 'permutation':
        corr_func = 0
        for i in sets:
            for j in sets:
                if type == 'default':
                    corr_func += SpatialCorrelationFunction(field[i,j],field[j,i]) 
                elif type == 'height-height':
                    corr_func += SpatialCorrelationFunction_HeightHeight(field[i,j],field[j,i]) 
                elif type == 'zero-average':
                    corr_func += SpatialCorrelationFunction_ZeroAverage(field[i,j],field[j,i]) 
                else:
                    pass
    elif option == 'trace':
        trace_field = 0.
        for i in sets:
            trace_field += field[i,i]
        corr_func = CorrelationFunctionsOfScalarField(trace_field,type) 
    else:
        print "Please choose a valid option, see help"
    return corr_func


def RadialCorrelationFunctions(field,fieldtype,corrfunctype='default',symmetrytype='total',logbinning=True,binnum=50,fast=True):
    """
    Calculates the radial correlation function of a field of type fieldtype. 

    * fieldtype can be one of {'tensor', 'vector', 'scalar'}

    * symmetrytype specifies the symmetries of the tensor field and is one of 
        {'total', 'permutation', 'trace'}
    
    * corrfunctype specifies the variety of correlation function to run:
        {'default', 'height-height', 'zero-average'}
    """

    if fieldtype == 'scalar':
        corr_func = CorrelationFunctionsOfScalarField(field,corrfunctype)
    elif fieldtype == 'vector':
        corr_func = CorrelationFunctionsOfVectorField(field,corrfunctype)
    elif fieldtype == 'tensor':
        corr_func = CorrelationFunctionsOfTensorField(field,corrfunctype,combinationtype)
    else:
        pass
    if fast:
        radial_corr_func = RadialCorrFuncFromSpatialCorrFuncFast(corr_func)
    else:
        radial_corr_func = RadialCorrFuncFromSpatialCorrFunc(corr_func)
    if logbinning:
        radial_corr_func = BinData_Log(radial_corr_func[0],radial_corr_func[1],binnum) 
    return radial_corr_func


def Run(inputfile,input,shape,fieldtype,corrfunctype,symmetrytype,logbinning,binnum,outputfile,plot):
    """
    Performs RadialCorrelationFunctions on an inputfile that holds field data
    in the space separated values of arbitrary arrangement (i.e. rows / columns don't matter).
    
    Input is either by filename or by numpy.array.  If input is specified it is taken to be the 
    data array.  If not, the inputfile is opened and read for data.

    data structure: array.shape should be like (dimension, dimension[0], dimension[1], ...).
        examples:
            * one dimensional 3-component vector field with 256 values - (3, 256)
            * three dimensional 3-component vector field - (3, 256, 256, 256)
            * 3x3 3d tensor field - (3, 3, 256, 256, 256)
    """
    if input is None:
        import Data_IO
        data = Data_IO.ReadInScalarField(inputfile,shape) 
    else:
        data = input
    corr_func = RadialCorrelationFunctions(data,fieldtype,corrfunctype,symmetrytype,logbinning,binnum,fromState=False)
    if outputfile is None:
        outputfile = inputfile+'_correlationfunction.dat'
    Data_IO.OutputXY(corr_func[0],corr_func[1],outputfile)
    if plot:
        import pylab
        pylab.figure()
        pylab.loglog(corr_func[0],corr_func[1],'.-')
        pylab.xlabel(r'$C(r)$',fontsize=20)
        pylab.ylabel(r'$r$',fontsize=20)
        pylab.show()

def main(argv):
    try:
        opts, args = getopt.getopt(argv,'hi:o:s:f:c:r:l:b:p:d',['help','input=','output=',\
                                        'shape=','fieldtype=','corrfunctype=','combinationtype=',\
                                        'logbinning=','binnum=','plot=','demo'])
    except getopt.GetoptError, err:
        print str(err)
        sys.exit(2)
    inputfile,outputfile,shape,fieldtype,corrfunctype,combinationtype,logbinning,binnum,plot = None,None,None,None,None,None,None,None,None
    for opt,arg in opts:
        if opt in ('-i','--input'):
            inputfile = arg
        elif opt in ('-o','--output'):
            outputfile = arg
        elif opt in ('-s','--shape'):
            shape = numpy.array(arg.rsplit(',')).astype(int) 
        elif opt in ('-f','--fieldtype'):
            fieldtype = arg
        elif opt in ('-c','--corrfunctype'):
            corrfunctype = arg
        elif opt in ('-r','--combinationtype'):
            combinationtype = arg
        elif opt in ('-l','--logbinning'):
            logbinning = bool(arg)
        elif opt in ('-b','--binnum'):
            binnum = int(arg) 
        elif opt in ('-p','--plot'):
            plot = bool(arg)
        elif opt in ('-h','--help'):
            print """ 
                    [OPTIONS]                   FUNCTIONS 
                  -h,--help                  This help page 
                  -d,--demo                  Show the demonstration 
                  -i,--input=<inputfile>     Pass the name of input data file 
                  -o,--output=<outputfile>   Pass the name of output data file 
                  -s,--shape=<L,L>           Specify the shape of data array 
                  -c,--corrfunctype=<str>    Choose the type of correlation function 
                  -r,--combinationtype=<str> Choose the type of rotational invariance  
                  -f,--fieldtype=<str>       Choose the type of field 
                  -l,--logbinning=<bool>     Turn on/off the log binning
                  -b,--binnum=<int>          Specify the number of log bins 
                  -p,--plot=<bool>           Turn on/off the plotting
                  """
    if (inputfile is not None) and (shape is not None) and (field is not None)\
        and (corrfunctype is not None) and (combinationtype is not None) and (logbinning is not None) and \
        (binnum is not None) and (plot is not None):
        Run(inputfile,shape,fieldtype,corrfunctype,combinationtype,logbinning,binnum,outputfile,plot)
        
if __name__ == "__main__":
    main(sys.argv[1:])  
