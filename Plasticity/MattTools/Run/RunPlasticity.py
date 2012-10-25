#!/usr/bin/python2.4 
import PlasticitySystem
import FieldInitializer
import FieldDynamics
import FieldMover
import VacancyDynamics
import Observer
import pylab
import VacancyState, PlasticityState
import numpy
import scipy.weave as weave
from Constants import *
import GridArray
import CentralUpwindHJBetaPDynamics
import NumericalMethods
import CentralUpwindHJBetaPGlideOnlyDynamics
import WallInitializer
import Fields

# shape and size of sim.
N = 128 
gridShape = (N,N)
 
dir = "./"
lengthscale = 0.2

# load parameters
loadrate = 0.01
loadconst= 0.0
loaddir  = numpy.array([-0.5,-0.5,1.0]) 
loadtype = 'strain'

# vacancy parameters
gamma  = 1e-2
alpha  = 1e0
beta   = 1
c0     = 0

# upwind glide only parameter
Lambda = 0

# old file to use for ics=7
oldfile = "RND_Upwind_L0_S0_2D128.save"


method = 5
"""
0 : Vacancies 
1 : NewGlideOnly
2 : Upwind
3 : Vacancies w/ load 
4 : NewGlideOnly w/ load
5 : Upwind w/ load
"""

ics = 7
"""
0 : random gaussian with lengthscale
1 : 4 point sources
2 : 2 tilt walls - perpendicular point
3 : 2 tilt walls - parallel point
4 : 2 offset parallel point dislocation
5 : Nabarro-Herring hexagons
6 : add walls to a relaxed state
7 : load old file
"""

LoadRate = loadrate*loaddir 
LoadInit = loadconst*loaddir
LoadType = loadtype
loaddstr = '['+str(loaddir[0])+','+str(loaddir[1])+','+str(loaddir[2])+']'


mu,nu = 0.5,0.3 
lamb = 2.*mu*nu/(1.-2.*nu)
def ExternalStrain(sigma,primaryStrain):
    strains = {x:primaryStrain[0],y:primaryStrain[1],z:primaryStrain[2]}
    strain_trace = strains[x]+strains[y]+strains[z]
    for i in [x,y,z]:
        sigma[i,i] += lamb*strain_trace + 2.*mu*strains[i] 
    return sigma

def ExternalStress(sigma,primaryStress):
    stresses = {x:primaryStress[0],y:primaryStress[1],z:primaryStress[2]}
    for i in [x,y,z]:
        sigma[i,i] += stresses[i] 
    return sigma


class UpwindLoadingBetaPDynamics(CentralUpwindHJBetaPDynamics.BetaPDynamics):
    def __init__(self, Dx=None, Lambda=0, coreEnergy=0, coreEnergyLog=0, rate=None, type='strain', \
                       initial=numpy.array([0.,0.,0.]),initT=0.):
        CentralUpwindHJBetaPDynamics.BetaPDynamics.__init__(self,Dx,coreEnergy,coreEnergyLog)
        self.rate = rate
        self.type = type
        self.initial = initial
        self.initT = initT

    def GetSigma(self,state,time):
        sigma = state.CalculateSigma()
        if self.type == 'strain':
            return ExternalStrain(sigma,self.initial+self.rate*(time-self.initT)) 
        else:
            return ExternalStress(sigma,self.initial+self.rate*(time-self.initT)) 


class NewGlideOnlyLoadDynamics(CentralUpwindHJBetaPGlideOnlyDynamics.BetaPDynamics):
    def __init__(self, Dx=None, Lambda=0, coreEnergy=0, coreEnergyLog=0, rate=None, type='strain', \
                       initial=numpy.array([0.,0.,0.]),initT=0.):
        CentralUpwindHJBetaPGlideOnlyDynamics.BetaPDynamics.__init__(self,Dx,coreEnergy,coreEnergyLog)
        self.rate = rate
        self.type = type
        self.initial = initial
        self.initT = initT

    def GetSigma(self,state,time):
        sigma = state.CalculateSigma()
        if self.type == 'strain':
            return ExternalStrain(sigma,self.initial+self.rate*(time-self.initT)) 
        else:
            return ExternalStress(sigma,self.initial+self.rate*(time-self.initT)) 

  
class VacancyDynamicsExternalLoad(VacancyDynamics.BetaP_VacancyDynamics):
    def __init__(self, alpha=1.0, gamma=1.0, beta=1.0, Dx=None, Lambda=0, coreEnergy=0, coreEnergyLog=0, rate=None, type='strain', \
                       initial=numpy.array([0.,0.,0.]),initT=0.):
        VacancyDynamics.BetaP_VacancyDynamics.__init__(self,Dx,coreEnergy,coreEnergyLog,alpha=alpha,gamma=gamma,beta=beta)
        self.rate = rate
        self.type = type
        self.initial = initial
        self.initT = initT

    def GetSigma(self,state,time, cfield):
        sigma = state.CalculateSigma()
        for i in [x,y,z]:
            sigma[i,i] -= self.alpha*cfield 

        if self.type == 'strain':
            return ExternalStrain(sigma,self.initial+self.rate*(time-self.initT)) 
        else:
            return ExternalStress(sigma,self.initial+self.rate*(time-self.initT)) 

class TotalFreeEnergyDownhillObserver(Observer.Observer):
    """
    Checks whether energy goes downhill.
    """
    def __init__(self):
        self.PreviousEnergy = None
        self.countTotal = 0
        self.countBad = 0

    def Update(self, time, state):
        DisE = state.CalculateElasticEnergy()
        VE = state.CalculateVacancyEnergy()
        Energy = (DisE+VE).sum()
        self.countTotal += 1
        if self.PreviousEnergy is not None and self.PreviousEnergy < Energy:
            self.countBad += 1
        """
        For debuging.
        """
        if self.countBad >0 :
            print "%d/%d" % (self.countBad, self.countTotal)
            sys.exit(1)
        self.PreviousEnergy = Energy
 
class TraceBetaPObserver(Observer.Observer):
    def __init__(self, name):
        self.timecount = 0
        self.filename = name
        pylab.figure(10)

    def Update(self, time, state):
        field = state.betaP_V #GetOrderParameterField()
        trace = field['x','x'] + field['y','y'] + field['z','z']
        pylab.figure(10)
        pylab.clf()
        pylab.imshow(trace)
        
        if self.filename is not None:
            pylab.savefig(self.filename+"%.3f"%time+".png")
        else:
            pylab.show()

class VacancyObserver(Observer.Observer):
    def __init__(self, name):
        self.timecount = 0
        self.filename = name
        pylab.figure(11)

    def Update(self, time, state):
        field = state.betaP_V #GetOrderParameterField()
        trace = field['s','s']
        pylab.figure(11)
        pylab.clf()
        pylab.imshow(trace)
        
        if self.filename is not None:
            pylab.savefig(self.filename+"%.3f"%time+".png")
        else:
            pylab.show()

"""
class MemoryObserver(Observer.Observer):
    def __init__(self):
        pass

    def Update(self, time, state):
        GridArray.print_mem_usage()
"""

def identity(a):
    greater = (a>numpy.pi).astype(float)
    return a-2*numpy.pi*greater

def generate_gaussian(shape, x0, y0, width, scale=1, cut=None):
    if cut is None:
        if len(shape) == 2:
            return scale*numpy.fromfunction(lambda x,y: numpy.exp(-((x-x0)**2+(y-y0)**2)/(2.*width**2)), shape)
        else:
            return scale*numpy.fromfunction(lambda x,y,z: numpy.exp(-((x-x0)**2+(y-y0)**2+(z-y0)**2)/(2.*width**2)), shape)
    else:
        if len(shape) == 2:
            return scale*numpy.fromfunction(lambda x,y: numpy.exp(-((x-x0)**2+(y-y0)**2)/(2.*width**2))*(abs(x-x0)<cut)*(abs(y-y0)<cut), shape)
        else:
            return scale*numpy.fromfunction(lambda x,y,z: numpy.exp(-((x-x0)**2+(y-y0)**2+(z-y0)**2)/(2.*width**2))*(abs(x-x0)<cut)*(abs(y-y0)<cut)*(abs(z-y0)<cut), shape)

def generate_wall(shape, point, axis='y', blur=True):
    wall = numpy.zeros(shape)
    if blur == True:
        if axis == 'x':
            wall[:,point] = 1
        if axis == 'y':
            wall[point,:] = 1
    if blur == False:
        if axis == 'x':
            wall = numpy.fromfunction(lambda x,y: numpy.exp(-(y-point)**2), shape)
        if axis == 'y':
            wall = numpy.fromfunction(lambda x,y: numpy.exp(-(x-point)**2), shape)
    return wall

def generate_kvecs(gridShape):
    if len(gridShape) == 2:
        func = identity
        newgridShape = [gridShape[0], gridShape[1]/2+1]
        kx    = numpy.fromfunction(lambda x,y: gridShape[1]*func(2.*numpy.pi*(x)/float(gridShape[0])),newgridShape)
        ky    = numpy.fromfunction(lambda x,y: gridShape[1]*func(2.*numpy.pi*(y)/float(gridShape[1])),newgridShape)
        kz    = numpy.zeros(newgridShape)
        kxkx, kxky, kyky  = kx*kx, kx*ky, ky*ky
        kSq   = kxkx + kyky
        kSqSq = kSq*kSq
    else:
        func = identity
        newgridShape = [gridShape[0], gridShape[1], gridShape[2]/2+1]
        kx    = numpy.fromfunction(lambda x,y,z: gridShape[0]*func(2.*numpy.pi*(x)/float(gridShape[0])),newgridShape)
        ky    = numpy.fromfunction(lambda x,y,z: gridShape[1]*func(2.*numpy.pi*(y)/float(gridShape[1])),newgridShape)
        kz    = numpy.fromfunction(lambda x,y,z: gridShape[2]*func(2.*numpy.pi*(z)/float(gridShape[2])),newgridShape)
        kxkx, kxky, kyky  = kx*kx, kx*ky, ky*ky
        kSq   = kxkx + kyky + kz*kz
        kSqSq = kSq*kSq
    return kx, ky, kSq
  

def createHexagons(gridShape, n):
    """
    n is the number of hexagons along a given direction
    which is also the grain size given by d = n/gridShape[0]
    """
    N = gridShape[0]
    a = 1.*N/n
    R = 3

    # create a voronoi of a triangular lattice
    rot = []
    latticex = []
    latticey = []

    for i in range(-2,n+2):
        for j in range(-2,n+2):
            latticex.append(j*a + (i%2)*a/2)
            latticey.append(i*a)
            rot.append(((R-1)*(i%2) + j)%R)
 
    lx = numpy.array(latticex).flatten()
    ly = numpy.array(latticey).flatten()
    rot = numpy.array(rot).flatten()
    #rot = numpy.random.randint(low=0, high=len(lx),size=len(lx))
    result = numpy.zeros(gridShape)
    PTS = len(lx)

    code = """
        for (int i=0; i<N; i++){
            for (int j=0; j<N; j++){
                double dist = 1e10;
                int index = 0;
                for (int z=0; z<PTS; z++){
                    double dx = abs(i-*(lx+z));
                    double dy = abs(j-*(ly+z));

                    /*if (dx > N/2)
                        dx -= N/2;
                    if (dy > N/2)
                        dy -= N/2;*/

                    double temp = sqrt(dx*dx + dy*dy);
                    
                    if (temp < dist){
                        index = z;
                        dist = temp;
                    }
                }
                
                *(result+i+j*N) = *(rot+index);
            }
        }
    """

    weave.inline(code, ['N', 'PTS', 'lx', 'ly', 'rot', 'result'], extra_compile_args=["-w"])
    result = numpy.rot90(result.astype('int'))

    if False:
        pylab.imshow(result)
        pylab.plot(lx, ly, 'o')
        pylab.xlim(0,N)
        pylab.ylim(0,N)

    rodvecsx = 0*numpy.arange(-n/2+1, n/2+1)[::-1]/n**3
    rodvecsy = 0*numpy.arange(-n/2-2, n/2-2)/n**3
    rodvecsz = 1*numpy.arange(-R/(R-1.), 1.*R, R/(R-1.))/R**3
    #rodvecsz = n/2 - abs(numpy.arange(-n/2,n/2)) 
    #rodvecsz = numpy.random.rand(len(lx))

    rodrigues = Fields.VectorField(gridShape)
    rodrigues['x'] = GridArray.GridArray(rodvecsx[result])
    rodrigues['y'] = GridArray.GridArray(rodvecsy[result])
    rodrigues['z'] = GridArray.GridArray(rodvecsz[result])

    state = VacancyState.VacancyState(gridShape, alpha=alpha)
    betap, cfield = state.DecoupleState()
    betap = PlasticityState.PlasticityState(gridShape, 
                WallInitializer.InitializeRhoFromRodriguesField(gridShape, rodrigues).CalculateBetaP())
    state.RecoupleState(betap, cfield) #FieldInitializer.RotateState_45degree(N,betap), cfield)
    return state
 

def createFourPoints(gridShape):
    state = VacancyState.VacancyState(gridShape,alpha = alpha)
    field = state.GetOrderParameterField()
    
    temp = numpy.arange(0, numpy.prod(gridShape)).reshape(gridShape) 
    foo1 = numpy.zeros(gridShape)
    foo2 = numpy.zeros(gridShape)
 
    sigma = 1 
    offset = 1*gridShape[0]/6
    foo1 += generate_gaussian(gridShape, gridShape[0]/2, gridShape[0]/2-offset, sigma)
    foo1 -= generate_gaussian(gridShape, gridShape[0]/2, gridShape[0]/2+offset, sigma)

    foo2 += generate_gaussian(gridShape, gridShape[0]/2-offset, gridShape[0]/2, sigma)
    foo2 -= generate_gaussian(gridShape, gridShape[0]/2+offset, gridShape[0]/2, sigma)

    #vac = generate_gaussian(gridShape, gridShape[0]/2, gridShape[0]/2, 16)

    kx,ky,kSq = generate_kvecs(gridShape)

    byx = numpy.fft.irfftn(1.j  * kx / (kSq+NumericalMethods.ME) * numpy.fft.rfftn(foo1))
    bxx = numpy.fft.irfftn(-1.j * ky / (kSq+NumericalMethods.ME) * numpy.fft.rfftn(foo1))

    bxy = numpy.fft.irfftn(-1.j * ky / (kSq+NumericalMethods.ME) * numpy.fft.rfftn(foo2))
    byy = numpy.fft.irfftn(1.j  * kx / (kSq+NumericalMethods.ME) * numpy.fft.rfftn(foo2))

    field[y,x] = GridArray.GridArray(byx)
    field[x,x] = GridArray.GridArray(bxx)
    #field['s','s'] = GridArray.GridArray(vac)
    
    field[x,y] = GridArray.GridArray(bxy)
    field[y,y] = GridArray.GridArray(byy)
    
    state.UpdateOrderParameterField(field)

    return state

def createTwoTiltWalls_ParallelPointDislocations(gridShape):
    state = VacancyState.VacancyState(gridShape,alpha = alpha)
    field = state.GetOrderParameterField()
    
    temp = numpy.arange(0, numpy.prod(gridShape)).reshape(gridShape) 
    foo1 = numpy.zeros(gridShape)
    foo2 = numpy.zeros(gridShape)

    # create the shapes necessary 
    #foo1 +=  1*generate_wall(gridShape, 1*gridShape[0]/3, 'y', blur=False)
    #foo1 += -1*generate_wall(gridShape, 2*gridShape[0]/3, 'y', blur=False)

    # matt: was 1/3+1/6, 1
    #foo1 +=  1*generate_gaussian(gridShape, int((1./3+1./14)*gridShape[0]), 1*gridShape[0]/3, 4)
    #foo1 += -1*generate_gaussian(gridShape, int((1./3+1./14)*gridShape[0]), 2*gridShape[0]/3, 4)

    foo1 +=  1*generate_wall(gridShape, 1*gridShape[0]/4, 'y', blur=False)
    foo1 += -1*generate_wall(gridShape, 3*gridShape[0]/4, 'y', blur=False)
    foo1 +=  1*generate_gaussian(gridShape, int((1./4+1./12)*gridShape[0]), 1*gridShape[0]/4, 6)
    foo1 += -1*generate_gaussian(gridShape, int((1./4+1./12)*gridShape[0]), 3*gridShape[0]/4, 6)

    # translate them into an order parameter field
    kx,ky,kSq = generate_kvecs(gridShape)

    byx = numpy.fft.irfftn(1.j  * kx / (kSq+NumericalMethods.ME) * numpy.fft.rfftn(foo1))
    bxx = numpy.fft.irfftn(-1.j * ky / (kSq+NumericalMethods.ME) * numpy.fft.rfftn(foo1))
    field[y,x] = GridArray.GridArray(byx)
    field[x,x] = GridArray.GridArray(bxx)

    byy = numpy.fft.irfftn(1.j  * kx / (kSq+NumericalMethods.ME) * numpy.fft.rfftn(foo2))
    bxy = numpy.fft.irfftn(-1.j * ky / (kSq+NumericalMethods.ME) * numpy.fft.rfftn(foo2))
    field[y,y] = GridArray.GridArray(byy)
    field[x,y] = GridArray.GridArray(bxy)

    state.UpdateOrderParameterField(field)

    return state


def createTwoTiltWalls_PerpendicularPointDislocations(gridShape):
    state = VacancyState.VacancyState(gridShape,alpha = alpha)
    field = state.GetOrderParameterField()
    
    temp = numpy.arange(0, numpy.prod(gridShape)).reshape(gridShape) 
    foo1 = numpy.zeros(gridShape)
    foo2 = numpy.zeros(gridShape)
 
    foo1 +=  1*generate_wall(gridShape, 1*gridShape[0]/3, 'y', blur=False)
    foo1 += -1*generate_wall(gridShape, 2*gridShape[0]/3, 'y', blur=False)
    
    foo2 +=  1*generate_gaussian(gridShape, int((2./3+1./12)*gridShape[0]), gridShape[0]/2, 4)
    foo2 += -1*generate_gaussian(gridShape, int((1./3-1./12)*gridShape[0]), gridShape[0]/2, 4)

    kx,ky,kSq = generate_kvecs(gridShape)
    byx = numpy.fft.irfftn(1.j  * kx / (kSq+NumericalMethods.ME) * numpy.fft.rfftn(foo1))
    bxx = numpy.fft.irfftn(-1.j * ky / (kSq+NumericalMethods.ME) * numpy.fft.rfftn(foo1))
    field[y,x] = GridArray.GridArray(byx)
    field[x,x] = GridArray.GridArray(bxx)

    byy = numpy.fft.irfftn(1.j  * kx / (kSq+NumericalMethods.ME) * numpy.fft.rfftn(foo2))
    bxy = numpy.fft.irfftn(-1.j * ky / (kSq+NumericalMethods.ME) * numpy.fft.rfftn(foo2))
    field[y,y] = GridArray.GridArray(byy)
    field[x,y] = GridArray.GridArray(bxy)

    state.UpdateOrderParameterField(field)

    return state

def createOffsetParallelPointDislocations(gridShape):
    state = VacancyState.VacancyState(gridShape,alpha = alpha)
    field = state.GetOrderParameterField()
    
    temp = numpy.arange(0, numpy.prod(gridShape)).reshape(gridShape) 
    foo1 = numpy.zeros(gridShape)
    foo2 = numpy.zeros(gridShape)
 
    #foo1 +=  1*generate_gaussian(gridShape, int((1./3+1./12)*gridShape[0]), gridShape[0]/2, 2)
    #foo1 += -1*generate_gaussian(gridShape, int((1./3-1./12)*gridShape[0]), gridShape[0]/2, 2)

    #foo2 +=  1*generate_gaussian(gridShape, 3*gridShape[0]/8, 1*gridShape[0]/3, 2)
    #foo2 += -1*generate_gaussian(gridShape, 3*gridShape[0]/8, 2*gridShape[0]/3, 2)

    foo1 +=  1*generate_gaussian(gridShape, gridShape[0]/2, int((1/2.+1/36.)*gridShape[0]), 2)
    foo1 += -1*generate_gaussian(gridShape, gridShape[0]/2, int((1/2.-1/36.)*gridShape[0]), 2)

    kx,ky,kSq = generate_kvecs(gridShape)
    byx = numpy.fft.irfftn(1.j  * kx / (kSq+NumericalMethods.ME) * numpy.fft.rfftn(foo1))
    bxx = numpy.fft.irfftn(-1.j * ky / (kSq+NumericalMethods.ME) * numpy.fft.rfftn(foo1))
    field[y,x] = GridArray.GridArray(byx)
    field[x,x] = GridArray.GridArray(bxx)

    byy = numpy.fft.irfftn(1.j  * kx / (kSq+NumericalMethods.ME) * numpy.fft.rfftn(foo2))
    bxy = numpy.fft.irfftn(-1.j * ky / (kSq+NumericalMethods.ME) * numpy.fft.rfftn(foo2))
    field[y,y] = GridArray.GridArray(byy)
    field[x,y] = GridArray.GridArray(bxy)

    state.UpdateOrderParameterField(field)

    return state


def addWallsToRelaxedState(filename, gridShape, axis='y'):
    t,state = FieldInitializer.LoadState(filename)
    field = state.GetOrderParameterField()
    scale = field.modulus().max()

    temp = numpy.arange(0, numpy.prod(gridShape)).reshape(gridShape) 
    foo1 = numpy.zeros(gridShape)
    
    foo1 +=  1*generate_wall(gridShape, 1*gridShape[0]/3, axis)
    foo1 += -1*generate_wall(gridShape, 2*gridShape[0]/3, axis)

    kx,ky,kSq = generate_kvecs(gridShape)
    byx = numpy.fft.irfftn(1.j  * kx / (kSq+NumericalMethods.ME) * numpy.fft.rfftn(foo1))
    bxx = numpy.fft.irfftn(-1.j * ky / (kSq+NumericalMethods.ME) * numpy.fft.rfftn(foo1))
    field[y,x] += scale*GridArray.GridArray(byx)
    field[x,x] += scale*GridArray.GridArray(bxx)

    state.UpdateOrderParameterField(field)

    return state

   

def Relaxation_BetaPV(seed=0):

    ##=======================================================================
    if   ics == 0:
        state = FieldInitializer.GaussianRandomInitializer(gridShape, lengthscale,seed,vacancy=None)
        info  = "RND_"

    elif ics == 1:
        state = createFourPoints(gridShape)
        info  = "TESTINGVACANCIES_4PT_"

    elif ics == 2:
        state = createTwoTiltWalls_PerpendicularPointDislocations(gridShape)
        info  = "WALLINTERACTIONS_TwoTiltWalls_Perpendicular_"

    elif ics == 3:
        state = createTwoTiltWalls_ParallelPointDislocations(gridShape)
        info  = "WALLINTERACTIONS_TwoTiltWalls_Parallel_"

    elif ics == 4:
        state = createOffsetParallelPointDislocations(gridShape)
        info  = "TESTINGVACANCIES_2PT_"

    elif ics == 5:
        state = createHexagons(gridShape, 12)
        info  = "NABARRO_2_"

    elif ics == 6:
        state = addWallsToRelaxedState("WALLINTERACTIONS_2PtsRelaxationParallel_Upwind_B1.0_L0_G1.0_A1.0C0_0.0S_02D256ST_0.001_yay.save", (256,256), 'y')
        info  = "WALLINTERACTIONS_Prerelaxed_AddWalls_" 
    
    elif ics == 7:
        tt,state = FieldInitializer.LoadState(oldfile)
        info  = oldfile+"_" 

    else:
        state = FieldInitializer.GaussianRandomInitializer(gridShape, lengthscale,seed,vacancy=None)
        info  = "AUTO_"
    ##========================================================================
    

    ##--------------------------------------------------------------------------------------
    if   method == 0:
        dynamics = VacancyDynamics.BetaP_VacancyDynamics(Lambda=0,gamma=gamma,alpha=alpha,beta=beta)
        dlabel = "VacancyDynamics_G"+str(gamma)+"_A"+str(alpha)+"_C"+str(c0)
        mover = FieldMover.OperatorSplittingTVDRK_FieldMover(CFLsafeFactor=0.5,dtBound=1./N)

    elif method == 1:
        dynamics = CentralUpwindHJBetaPGlideOnlyDynamics.BetaPDynamics()
        dlabel = "NewGlideOnly"
        mover = FieldMover.TVDRungeKutta_FieldMover(CFLsafeFactor=0.1)

    elif method == 2:
        dynamics = CentralUpwindHJBetaPDynamics.BetaPDynamics(Lambda=Lambda)
        dlabel = "Upwind"
        mover = FieldMover.TVDRungeKutta_FieldMover(CFLsafeFactor=0.1)

    elif method == 3:
        dynamics = VacancyDynamicsExternalLoad(Lambda=0,gamma=gamma, alpha=alpha, beta=beta,rate=LoadRate,initial=LoadInit,type=LoadType)
        dlabel = "VacancyWithLoad_"+loadtype+"_"+loaddstr+"_G"+str(gamma)+"_A"+str(alpha)+"_C"+str(c0)+"_STR"+str(loadrate)
        mover = FieldMover.OperatorSplittingTVDRK_FieldMover(CFLsafeFactor=0.5,dtBound=1./N)
 
    elif method == 4:
        dynamics = NewGlideOnlyLoadDynamics(rate=LoadRate,initial=LoadInit,type=LoadType)
        dlabel = "NewGlideOnlyWithLoad_"+loadtype+"_"+loaddstr+"_STR"+str(loadrate)
        mover = FieldMover.TVDRungeKutta_FieldMover(CFLsafeFactor=0.1)

    elif method == 5:
        dynamics = UpwindLoadingBetaPDynamics(Lambda=Lambda,rate=LoadRate,initial=LoadInit,type=LoadType)
        dlabel = "UpwindLoad_"+loadtype+"_"+loaddstr+"_STR"+str(loadrate)
        mover = FieldMover.TVDRungeKutta_FieldMover(CFLsafeFactor=0.1)

    else:
        dynamics = CentralUpwindHJBetaPDynamics.BetaPDynamics(Lambda=Lambda)
        dlabel = "Upwind"
        mover = FieldMover.TVDRungeKutta_FieldMover(CFLsafeFactor=0.1)
    ##-------------------------------------------------------------------------------------




    ##===========================================================================
    ##========== do not change below here typically =============================
    ##=========================================================================== 
    filename = dir+info+dlabel+"_L"+str(Lambda)+"_S"+str(seed)+"_"+str(len(gridShape))+"D"+str(N)+".save"
    #state.betaP_V['s','s'] *= c0
   
    obsState = Observer.RecallStateObserver()
    energychecking = TotalFreeEnergyDownhillObserver()

    startTime = 0. 
    endTime   = 30.
    dt = 0.025

    t = startTime 
    if startTime == 0. :
        recordState = Observer.RecordStateObserver(filename=filename)
        recordState.Update(t, state)
    else:
        T,state = FieldInitializer.LoadState(filename)
        recordState = Observer.RecordStateObserver(filename=filename,mode='a')

    system= PlasticitySystem.PlasticitySystem(gridShape, state, mover, dynamics, [obsState,Observer.VerboseTimestepObserver()])#,energychecking]), traceBetaP, vacancyObs])

    #obsState.Update()
    #vacancyObs.Update(0, system.state)
    #traceBetaP.Update(0, system.state)
    #state = FieldInitializer.GaussianRandomInitializer(gridShape,lengthscale,seed,vacancy=alpha)
    #if N != 128:
    #    state = FieldInitializer.ResizeState(state,N)
    #state = FieldInitializer.ReformatState(state)
 
    while t<=(endTime):
        preT = t
        #"""
        #if t<=0.01-0.001:
        #    dt = 0.001
        if t<=0.1-0.01:
            dt = 0.01
        elif t<=1.:
            dt = 0.05
        elif t<=5.:
            dt = 0.5
        else:
            dt = 2.5
        #"""
        t += dt
        system.Run(startTime=preT, endTime = t)
        system.state = obsState.state
        recordState.Update(t, system.state)

def main():
    Relaxation_BetaPV()

if __name__ == "__main__":
    main()



