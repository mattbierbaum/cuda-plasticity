#import SlipSystemState
#import SlipSystemBetaPState
from Plasticity.PlasticityStates import RhoState
from Plasticity.PlasticityStates import VacancyState
from Plasticity.PlasticityStates import PlasticityState
from Plasticity.Fields import Fields
from scipy import fromfunction, sin, pi, outer, copy, fromfile, random, exp, sqrt
from numpy import fft
import numpy

from Constants import *

from Plasticity.GridArray import GridArray
from Plasticity.GridArray import FourierSpaceTools
import copy

def ReformatState(state):
    """
    Reformat the states to use GridArray.

    Use of numpy arrays prohibited in calculation
    """
    field = state.GetOrderParameterField()
    for comp in field.components:
        field[comp] = GridArray.GridArray(field[comp])
    return state

def LoadState(filename, time=None):
    """
    Load a state from a pickled file.

    By default, it picks up the last state.
    If time argument is supplied, it loads the first snapshot that is
    after this time.

    Returns time, state
    """
    import pickle
    file = open(filename)
    T = None
    state = None
    try:
        while True:
            T = pickle.load(file)
            state = pickle.load(file)
            if time is not None and T >= time:
                file.close()
                return T, state 
    except EOFError:
        if time is None and state is not None:
            file.close()
            return T, state
        else:
            if time is None:
                print "No state could be loaded from file %s." % filename
            else:
                print "No state after time %f found in file %s." % (time, filename)
        assert False
 
def LoadStateRaw(filename, N, dim, time=None, dtype='float64', hastimes=True):
    """
    Load a state from raw file
    """
    if dtype.__class__ != numpy.dtype:
        dtype = numpy.dtype(dtype)
    elemsize = dtype.itemsize
    datasize = 9*(N**dim)
    if isinstance(filename, str):
        file = open(filename, 'rb')
    else:
        file = filename
    file.seek(0,2)
    file_size = int(file.tell() / (elemsize*(datasize+1)))
    cnt = file_size
    file.seek(0,0)

    if hastimes == False:
        t=0
    else:
        if time is None:
            file.seek((cnt-1)*(elemsize*(datasize+1)),0)
            t = numpy.fromstring(file.read(elemsize), dtype=dtype)
        else:
            for i in range(cnt):
                file.seek(i*(elemsize*(datasize+1)),0)
                t = numpy.fromstring(file.read(elemsize), dtype=dtype)
                if t>=time:
                    break
    gridShape = tuple([N] * dim)
    state = PlasticityState.PlasticityState(gridShape)
    dict = {('x','x') : (0,0), ('x','y') : (0,1), ('x','z') : (0,2),\
            ('y','x') : (1,0), ('y','y') : (1,1), ('y','z') : (1,2),\
            ('z','x') : (2,0), ('z','y') : (2,1), ('z','z') : (2,2)}
    field = state.GetOrderParameterField()
    data = numpy.fromstring(file.read(elemsize*datasize),
            dtype=dtype).reshape(tuple([3,3] + list(gridShape)))
    tp = range(dim)
    tp.reverse()
    for component in field.components:
        field[component] = numpy.copy(data[dict[component]].transpose(tp))
    state = ReformatState(state)
    return t, state


def GenerateGaussianRandomArray(gridShape, temp, sigma):
    dimension = len(gridShape)
    if dimension == 1:
        kfactor = fromfunction(lambda kz: exp(-0.5*(sigma*kz)**2),[gridShape[0]/2+1,])
        ktemp = fft.rfft(temp)
        ktemp *= kfactor
        data = fft.irfft(ktemp)
    elif dimension == 2:
        X,Y = gridShape
        kfactor = fromfunction(lambda kx,ky: exp(-0.5*sigma**2*((kx*(kx<X/2)+(X-kx)*(kx>=X/2))**2+ky**2)),[X,Y/2+1])
        ktemp = fft.rfftn(temp)
        ktemp *= kfactor
        data = fft.irfftn(ktemp)
    elif dimension == 3:
        X,Y,Z = gridShape
        kfactor = fromfunction(lambda kx,ky,kz: exp(-0.5*sigma**2*( (kx*(kx<X/2)+(X-kx)*(kx>=X/2))**2 + \
                                                (ky*(ky<Y/2)+(Y-ky)*(ky>=Y/2))**2 + kz**2)),[X,Y,Z/2+1])
        ktemp = fft.rfftn(temp)
        ktemp *= kfactor
        data = fft.irfftn(ktemp)
    return data 


def GaussianRandomInitializer(gridShape, sigma=0.2, seed=None, slipSystem=None, slipPlanes=None, slipDirections=None, vacancy=None):

    oldgrid = copy.copy(gridShape)
   
    if len(gridShape) == 1:
	    gridShape = (128,)
    if len(gridShape) == 2:
	    gridShape = (128,128)
    if len(gridShape) == 3:
	    gridShape = (128,128,128)

    """ Returns a random initial set of fields of class type PlasticityState """
    if slipSystem=='gamma':
        state = SlipSystemState.SlipSystemState(gridShape,slipPlanes=slipPlanes,slipDirections=slipDirections)
    elif slipSystem=='betaP':
        state = SlipSystemBetaPState.SlipSystemState(gridShape,slipPlanes=slipPlanes,slipDirections=slipDirections)
    else:
        if vacancy is not None:
            state = VacancyState.VacancyState(gridShape,alpha=vacancy)
        else:
            state = PlasticityState.PlasticityState(gridShape)

    field = state.GetOrderParameterField()
    Ksq_prime = FourierSpaceTools.FourierSpaceTools(gridShape).kSq * (-sigma**2/4.)

    if seed is None:
        seed = 0
    n = 0
    random.seed(seed)

    Ksq = FourierSpaceTools.FourierSpaceTools(gridShape).kSq.numpy_array()

    for component in field.components:
        temp = random.normal(scale=gridShape[0],size=gridShape)
        ktemp = fft.rfftn(temp)*(sqrt(pi)*sigma)**len(gridShape)*exp(-Ksq*sigma**2/4.)
        field[component] = numpy.real(fft.irfftn(ktemp))
        #field[component] = GenerateGaussianRandomArray(gridShape, temp ,sigma)
        n += 1

    """
    t, s = LoadState("2dstate32.save", 0)
    for component in field.components:
        for j in range(0,32):
            field[component][:,:,j] = s.betaP[component].numpy_array()
    """

    ## To make seed consistent across grid sizes and convergence comparison
    gridShape = copy.copy(oldgrid)
    if gridShape[0] != 128:
        state = ResizeState(state,gridShape[0],Dim=len(gridShape))

    state = ReformatState(state)
    state.ktools = FourierSpaceTools.FourierSpaceTools(gridShape)
    
    return state 

def GenerateSineArray(gridShape, dimension, randomPhase = False):
    if randomPhase:
        phase = numpy.random.random()*2.*pi 
    else:
        phase = 0.
    result = fromfunction(lambda x: sin(pi*(x)/gridShape[0]+phase), [gridShape[0]])  
    for dim in range(1, dimension):
        if randomPhase:
            phase = numpy.random.random()*2.*pi 
        else:
            phase = 0.
        temp = fromfunction(lambda x: sin(pi*(x)/gridShape[dim]+phase), [gridShape[dim]])  
        result = outer(result, temp)        
    return result

def SineWaveInitializer(gridShape, randomPhase=False, seed=None, slipSystem=False, slipPlanes=None, slipDirections=None):
    """
    Initialize a plasticity state by setting all its components in any dimension with a
    single period of a sine function.
    """
    if seed is not None:
        random.seed(seed)
    if slipSystem=='gamma':
        pass
#state = SlipSystemState.SlipSystemState(gridShape,slipPlanes=slipPlanes,slipDirections=slipDirections)
    elif slipSystem=='betaP':
        pass
#state = SlipSystemState.SlipSystemBetaPState(gridShape,slipPlanes=slipPlanes,slipDirections=slipDirections)
    else:
        state = PlasticityState.PlasticityState(gridShape)
    field = state.GetOrderParameterField()
    for component in field.components:
        field[component] = GenerateSineArray(gridShape, field.GridDimension(), randomPhase = randomPhase)
    return state 

def NumpyTensorInitializer(gridShape, filename, bin=True):
    """
    Initialize a 9 component plasticity state by reading from a numpy "tofile" type file.
    """
    if bin:
        data = fromfile(filename)
    else:
        data = fromfile(filename,sep='  ')
    data = data.reshape([3,3] + list(gridShape))
    state = PlasticityState.PlasticityState(gridShape)
    dict = {('x','x') : (0,0), ('x','y') : (0,1), ('x','z') : (0,2),\
            ('y','x') : (1,0), ('y','y') : (1,1), ('y','z') : (1,2),\
            ('z','x') : (2,0), ('z','y') : (2,1), ('z','z') : (2,2)}
    field = state.GetOrderParameterField()
    for component in field.components:
        field[component] = copy(data[dict[component]]) 
    return state

def NumpyTensorInitializerForRho(gridShape, filename):
    """
    Initialize a 9 component plasticity state by reading from a numpy "tofile" type file.
    """
    data = fromfile(filename)
    data = data.reshape([3,3] + list(gridShape))
    dict = {('x','x') : (0,0), ('x','y') : (0,1), ('x','z') : (0,2),\
            ('y','x') : (1,0), ('y','y') : (1,1), ('y','z') : (1,2),\
            ('z','x') : (2,0), ('z','y') : (2,1), ('z','z') : (2,2)}
    rho = Fields.TensorField(gridShape) 
    for component in rho.components:
        rho[component] = copy(data[dict[component]]) 
    state = RhoState.RhoState(gridShape, field=rho)
    return state

def NumpyTensorInitializerForVacancy(gridShape, filename, vacancyfile=None):
    """
    Initialize a 10 component plasticity state by reading from a numpy "tofile" type file or two files.
    """
    dict = {('x','x') : (0,0), ('x','y') : (0,1), ('x','z') : (0,2),\
            ('y','x') : (1,0), ('y','y') : (1,1), ('y','z') : (1,2),\
            ('z','x') : (2,0), ('z','y') : (2,1), ('z','z') : (2,2)}
    data = fromfile(filename)
    if vacancyfile is None:
        data = data.reshape([10] + list(gridShape))
    else:
        data = data.reshape([3,3] + list(gridShape))
        dataV = fromfile(vacancyfile)
        dataV = dataV.reshape(list(gridShape))
    state = VacancyState.VacancyState(gridShape)
    field = state.GetOrderParameterField() 
    if vacancyfile is None:
        i = 0
        for component in field.components:
            field[component] = copy(data[i]) 
            i += 1
    else:
        for component in field.components:
            if component[0] not in [x,y,z]:
                field[component] = copy(dataV) 
            else:
                field[component] = copy(data[dict[component]]) 
    return state

def zeropadding1D(N,filename):
    a = numpy.fromfile(filename)
    a = a.reshape(9,N)
    b = numpy.zeros((9,2*N),float)
    for i in range(9):
        ka = fft.fft(a[i])
        kb = numpy.zeros(2*N,complex)
        kb[:N/2]  = ka[:N/2]
        kb[-N/2:] = ka[-N/2:]
        b[i] = fft.ifft(kb)
    b *= 2.
    b.tofile(filename.replace(str(N),str(2*N)))

def zeropadding2D(N,filename):
    a = numpy.fromfile(filename)
    a = a.reshape(9,N,N)
    b = numpy.zeros((9,2*N,2*N),float)
    for i in range(9):
        ka = fft.fft2(a[i])
        kb = numpy.zeros((2*N,2*N),complex)
        kb[:N/2,:N/2]   = ka[:N/2,:N/2]
        kb[-N/2:,:N/2]  = ka[-N/2:,:N/2]
        kb[:N/2,-N/2:]  = ka[:N/2,-N/2:]
        kb[-N/2:,-N/2:] = ka[-N/2:,-N/2:]
        b[i] = fft.ifft2(kb)
    b *= 4. 
    b.tofile(filename.replace(str(N),str(2*N)))
  
def zeroextract1d(N,filename):
    a = numpy.fromfile(filename)
    a = a.reshape(9,N)
    b = numpy.zeros((9,N/2),float)
    for i in range(9):
        ka = fft.fft(a[i])
        kb = numpy.zeros((N/2,),complex)
        kb[:N/4]=ka[:N/4]
        kb[-N/4:]=ka[-N/4:]
        b[i] = fft.ifft(kb)
    b /= 2.
    b.tofile(filename.replace(str(N),str(N/2)))

def zeroextract2d(N,filename):
    a = fromfile(filename)
    a = a.reshape(9,N,N)
    b = numpy.zeros((9,N/2,N/2),float)
    for i in range(9):
        ka = fft.rfftn(a[i])
        kb = numpy.zeros((N/2,N/4+1),complex)
        kb[:N/4,:]=ka[:N/4,:N/4+1]
        kb[-N/4:,:]=ka[-N/4:,:N/4+1]
        b[i] = fft.irfftn(kb)
    b /= 4.
    b.tofile(filename.replace(str(N),str(N/2)))


def SaveNumpyFileFromState(state,filename):
    field = state.GetOrderParameterField()
    data = []
    for comp in field.components:
        data.append(field[comp])
    numpy.array(data).tofile(filename)


def KspaceInitializer(gridShape, fileDictionary, state = None):
    """
    Initialize plasticity state by reading from files given by the file
    dictionary.

    File dictionary must be a dictionary(hash) with component of field as
    its keys and filename pair(for real and imaginary) as its values.
   
    State must first be initialized and passed in for non default plasticity
    states.
 
    example:
    fileDictionary = {('x','z') : \
        ("InitialConditions/InitFourierCoeffXZ_real256.dat", \
         "InitialConditions/InitFourierCoeffXZ_im256.dat"),\
        ('y','z') : ("InitialConditions/InitFourierCoeffYZ_real256.dat", \
         "InitialConditions/InitFourierCoeffYZ_im256.dat"),\
        ('z','x') : ("InitialConditions/InitFourierCoeffZX_real256.dat", \
         "InitialConditions/InitFourierCoeffZX_im256.dat")}
    """
    if state is None:
        state = PlasticityState.PlasticityState(gridShape)
    field = state.GetOrderParameterField() 
 
    kGridShape = list(gridShape)
    kGridShape[-1] = int(kGridShape[-1]/2)+1 
    kGridShape = tuple(kGridShape)
    totalSize = 1
    for sz in kGridShape:
        totalSize *= sz  

    for component in fileDictionary: 
        rePart = numpy.fromfile(fileDictionary[component][0])
        imPart = numpy.fromfile(fileDictionary[component][1])
        kSpaceArray = rePart + 1.0j*imPart
       
        """
        Strip down only first half rows as this is the only data that gets
        used. Notice that we are actually taking real fourier transform on
        last axis, nevertheless we only take half data from the top. i.e.
        this may not very intuitive, but is coded this way for compatibility
        with Yor's old C++ version. i.e., strip down only half rows, and re-
        arrange so that column is halved. (That is, second half of first row
        becomes the second row and first half of second row becomes the third.
        """
        kSpaceArray = kSpaceArray[:totalSize].reshape(kGridShape)
       
        field[component] = fft.irfftn(kSpaceArray)
    return state

def ResampleArray(arr,N,Dim=2):
    from scipy import signal
    if Dim == 1:
        return signal.resample(arr, N, axis=0)
    elif Dim == 2:
        return signal.resample(signal.resample(arr, N, axis=0), N, axis=1)
    elif Dim == 3:
        return signal.resample(signal.resample(signal.resample(arr, N, axis=0), N, axis=1),N,axis=2)

def ResizeState(state,N,Dim=2):
    if Dim == 1:
        newgridShape = tuple([N,])
    elif Dim == 2:
        newgridShape = tuple([N,N])
    elif Dim == 3:
        newgridShape = tuple([N,N,N])
    newstate = state.__class__(newgridShape)
    field = state.GetOrderParameterField()
    newfield = newstate.GetOrderParameterField()
    for component in field.components:
        newfield[component] = ResampleArray(field[component],N,Dim)
    return newstate

def RotateState_45degree(N,bstate):
    """
    rotate a square beta field state by 45 degrees
    """
    orgN = bstate.gridShape[0]
    from scipy import ndimage as ndi
    field = bstate.GetOrderParameterField()
    tempnewfield = {}
    for comp in field.components:
        """
        Resample to a very large size first to get rid of boundary effects
        by ndimage.rotate
        """
        tempnewfield[comp] = ResampleArray(ndi.rotate(ResampleArray(field[comp],N*4), 45, mode='wrap'),N)
        newgridShape = tempnewfield[comp].shape
    newbstate = PlasticityState.PlasticityState(newgridShape)
    newfield = newbstate.GetOrderParameterField()
    """
    """
    # X and Y both rotated
    newfield[x,x] = 0.5*(tempnewfield[x,x]-tempnewfield[x,y]-tempnewfield[y,x]+tempnewfield[y,y])
    newfield[x,y] = 0.5*(tempnewfield[x,x]+tempnewfield[x,y]-tempnewfield[y,x]-tempnewfield[y,y])
    newfield[y,x] = 0.5*(tempnewfield[x,x]-tempnewfield[x,y]+tempnewfield[y,x]-tempnewfield[y,y])
    newfield[y,y] = 0.5*(tempnewfield[x,x]+tempnewfield[x,y]+tempnewfield[y,x]+tempnewfield[y,y])
    # X or Y rotated
    newfield[x,z] = numpy.sqrt(0.5)*(tempnewfield[x,z]-tempnewfield[y,z])
    newfield[y,z] = numpy.sqrt(0.5)*(tempnewfield[x,z]+tempnewfield[y,z])
    newfield[z,x] = numpy.sqrt(0.5)*(tempnewfield[z,x]-tempnewfield[z,y])
    newfield[z,y] = numpy.sqrt(0.5)*(tempnewfield[z,x]+tempnewfield[z,y])
    # non changed
    newfield[z,z] = tempnewfield[z,z]

    newN = newfield[z,z].shape[0] 
    N = newN
    """
    Rescale to set it to the right time scale
    """ 
    newstate = ReformatState(newbstate)
    bstate = newbstate*bstate.CalculateRhoFourier().modulus().max()/newbstate.CalculateRhoFourier().modulus().max()
    #bstate = newbstate*((float(orgN)/N)**(2./3.))
    """
    """ 
    return bstate

def RotateState_90degree(state):
    tempnewfield = Fields.TensorField(state.gridShape) 
    from scipy import ndimage as ndi
    tempnewfield[x,x] = state.betaP[y,y]
    tempnewfield[x,y] = -state.betaP[y,x] 
    tempnewfield[x,z] = state.betaP[y,z] 
    tempnewfield[y,x] = -state.betaP[x,y]
    tempnewfield[y,y] = state.betaP[x,x] 
    tempnewfield[y,z] = -state.betaP[x,z] 
    tempnewfield[z,x] = state.betaP[z,y]
    tempnewfield[z,y] = -state.betaP[z,x] 
    tempnewfield[z,z] = state.betaP[z,z] 
    for comp in state.betaP.components:
        tempnewfield[comp] = ndi.rotate(tempnewfield[comp], 90, mode='wrap')
    state.UpdateOrderParameterField(tempnewfield)
    return state

