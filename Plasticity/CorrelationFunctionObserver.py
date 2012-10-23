from numpy import fft
import numpy

import scipy.weave as W
import sys
import getopt

def SpatialCorrelationFunctionA(Field1,Field2):
    """
    Designed for Periodic Boundary Condition.

    Corr_12(r) = <Phi_1(r)Phi_2(0)>
    Corr_12(k) = Phi_1(k)* Phi_2(k)/V 
    """ 
    V = float(numpy.array(Field1.shape).prod())
    KField1 = fft.rfftn(Field1).conj()
    KField2 = fft.rfftn(Field2) 
    KCorr = KField1*KField2/V 
    Corr  = fft.irfftn(KCorr)
    return Corr

def SpatialCorrelationFunctionB(Field1,Field2):
    """
    Designed for Periodic Boundary Condition.

    Corr_12(r) = <(Phi_1(r)-Phi_1(0))(Phi_2(r)-Phi_2(0))> 
               = 2<Phi_1*Phi_2> - <Phi_1(r)*Phi_2(0)> - <Phi_2(r)Phi_1(0)>
    """ 
    meansquare = numpy.average(Field1*Field2)
    Corr = 2*meansquare-SpatialCorrelationFunctionA(Field1,Field2)-SpatialCorrelationFunctionA(Field2,Field1)
    return Corr

def SpatialCorrelationFunctionC(Field1,Field2):
    """
    Designed for Periodic Boundary Condition.

    Corr_12(r) = <(Phi_1(r)-<Phi_1>)(Phi_2(0)-<Phi_2>)>
    """ 
    meansquare1 = numpy.average(Field1)
    meansquare2 = numpy.average(Field2)
    Corr = SpatialCorrelationFunctionA(Field1-meansquare1,Field2-meansquare2)
    return Corr


def RadialCorrFuncFromSpatialCorrFuncFast(corr):
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

def LogBinningData(x,y,num_of_bin=50,vmin=None,vmax=None):
    """
    Given the number of bins, x will be devided into log bins;
    y will be taken average inside every bin.
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


def CorrelationFunctionsOfScalarField(field,type):
    if type == 'A':
        corr_func = SpatialCorrelationFunctionA(field,field) 
    elif type == 'B':
        corr_func = SpatialCorrelationFunctionB(field,field) 
    elif type == 'C':
        corr_func = SpatialCorrelationFunctionC(field,field) 
    else:
        pass
    return corr_func

def CorrelationFunctionsOfVectorField(field,type,fromState=True):
    if fromState:
        sets = ['x','y','z']
    else:
        sets = [0,1,2]
    corr_func = 0
    for i in sets:
        if type == 'A':
            corr_func += SpatialCorrelationFunctionA(field[i],field[i]) 
        elif type == 'B':
            corr_func += SpatialCorrelationFunctionB(field[i],field[i]) 
        elif type == 'C':
            corr_func += SpatialCorrelationFunctionC(field[i],field[i]) 
        else:
            pass
    return corr_func

def CorrelationFunctionsOfTensorField(field,type,option=0,fromState=True):
    """
    Usually there are three rotational invariances for the correlation functions of
    tensor fields: C(A_{ij}A_{ij}),C(A_{ij}A_{ji}),C(A_{ii}A_{jj}).
    """
    if fromState:
        sets = ['x','y','z']
    else:
        sets = [0,1,2]
    if option == 0:
        corr_func = 0
        for i in sets:
            for j in sets:
                if type == 'A':
                    corr_func += SpatialCorrelationFunctionA(field[i,j],field[i,j]) 
                elif type == 'B':
                    corr_func += SpatialCorrelationFunctionB(field[i,j],field[i,j]) 
                elif type == 'C':
                    corr_func += SpatialCorrelationFunctionC(field[i,j],field[i,j]) 
                else:
                    pass
    elif option == 1:
        corr_func = 0
        for i in sets:
            for j in sets:
                if type == 'A':
                    corr_func += SpatialCorrelationFunctionA(field[i,j],field[j,i]) 
                elif type == 'B':
                    corr_func += SpatialCorrelationFunctionB(field[i,j],field[j,i]) 
                elif type == 'C':
                    corr_func += SpatialCorrelationFunctionC(field[i,j],field[j,i]) 
                else:
                    pass
    elif option == 2:
        trace_field = 0.
        for i in sets:
            trace_field += field[i,i]
        corr_func = CorrelationFunctionsOfScalarField(trace_field,type) 
    else:
        pass
    return corr_func


def RadialCorrelationFunctions(field,fieldtype,corrfunctype,combinationtype=0,logbinning=True,binnum=50,fast=True,fromState=True):
    if fieldtype == 'scalar':
        corr_func = CorrelationFunctionsOfScalarField(field,corrfunctype)
    elif fieldtype == 'vector':
        corr_func = CorrelationFunctionsOfVectorField(field,corrfunctype,fromState)
    elif fieldtype == 'tensor':
        corr_func = CorrelationFunctionsOfTensorField(field,corrfunctype,combinationtype,fromState)
    else:
        pass
    if fast:
        radial_corr_func = RadialCorrFuncFromSpatialCorrFuncFast(corr_func)
    else:
        radial_corr_func = RadialCorrFuncFromSpatialCorrFunc(corr_func)
    if logbinning:
        radial_corr_func = LogBinningData(radial_corr_func[0],radial_corr_func[1],binnum) 
    return radial_corr_func

def Demo():
    pass

def Run(inputfile,shape,fieldtype,corrfunctype,combinationtype,logbinning,binnum,outputfile,plot,input=None):
    if input is None:
        import Data_IO
        data = Data_IO.ReadInScalarField(inputfile,shape) 
    else:
        data = input
    corr_func = RadialCorrelationFunctions(data,fieldtype,corrfunctype,combinationtype,logbinning,binnum,fromState=False)
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
        elif opt in ('-d','--demo'):
            Demo() 
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
