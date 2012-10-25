import PlasticitySystem
import FieldInitializer
import FieldDynamics
import FieldMover
import VacancyDynamics
import Observer
import pylab
import VacancyState
import numpy
from Constants import *
import GridArray
import CentralUpwindHJBetaPDynamics
import CentralUpwindHJBetaPGlideOnlyDynamics
import NumericalMethods

exstress = 0.15

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


#class StrainLoadingBetaPDynamics(CentralUpwindHJBetaPDynamics.BetaPDynamics):
class StrainLoadingBetaPDynamics(CentralUpwindHJBetaPGlideOnlyDynamics.BetaPDynamics):
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


class ExternalStressDynamics(CentralUpwindHJBetaPDynamics.BetaPDynamics):
    def GetSigma(self, state, time):
        sigma = state.CalculateSigma()
	sigma['y','x'] += exstress
	return sigma
	
class VacancyDynamicsExternalStress(VacancyDynamics.BetaP_VacancyDynamics):
    def GetSigma(self, state, time, cfield):
        sigma = state.CalculateSigma()
        for i in [x,y,z]:
            sigma[i,i] -= self.alpha*cfield 
        sigma['y','x'] += exstress
        return sigma

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


def createBox(gridShape):
    state = VacancyState.VacancyState(gridShape,alpha = numpy.zeros(gridShape))
    field = state.GetOrderParameterField()
    
    temp = numpy.arange(0, numpy.prod(gridShape)).reshape(gridShape) 
    foo1 = numpy.zeros(gridShape)
    foo2 = numpy.zeros(gridShape)
    #foo[gridShape[0]/4:3*gridShape[0]/4,gridShape[1]/4:3*gridShape[1]/4] = 1
    #field[z,x] = GridArray.GridArray(foo)
    #field[z,y] = GridArray.GridArray(foo.copy())

    #foo[:,:] = 1
    #foo[gridShape[0]/4:3*gridShape[0]/4, :] = 0
    #foo[:, gridShape[0]/4:3*gridShape[0]/4] = 0
    
    #foo[1*gridShape[0]/3, :] = 1
    #foo[2*gridShape[0]/3, :] = 1
    #foo[:, gridShape[0]/3:2*gridShape[0]/3] = 0

    sigmax = 1#gridShape[0]/64
    sigmay = 1#gridShape[0]/64
    p1  = 1*gridShape[0]/3
    p2  = 2*gridShape[0]/3
    mid = 1*gridShape[0]/2

    foo1 += numpy.fromfunction(lambda x,y: numpy.exp(-((y-p1)**2/(2*sigmay**2)) - ((x-mid)**2/(2*sigmax**2))), gridShape)
    foo1 -= numpy.fromfunction(lambda x,y: numpy.exp(-((y-p2)**2/(2*sigmay**2)) - ((x-mid)**2/(2*sigmax**2))), gridShape)

    foo2 += numpy.fromfunction(lambda x,y: numpy.exp(-((y-p1)**2/(2*sigmax**2)) - ((x-mid)**2/(2*sigmay**2))), gridShape)
    foo2 -= numpy.fromfunction(lambda x,y: numpy.exp(-((y-p2)**2/(2*sigmax**2)) - ((x-mid)**2/(2*sigmay**2))), gridShape)

    func = identity
    newgridShape = [gridShape[0], gridShape[1]/2+1]
    kx    = numpy.fromfunction(lambda x,y: gridShape[1]*func(2.*numpy.pi*(x)/float(gridShape[0])),newgridShape)
    ky    = numpy.fromfunction(lambda x,y: gridShape[1]*func(2.*numpy.pi*(y)/float(gridShape[1])),newgridShape)
    kz    = numpy.zeros(newgridShape)
    kxkx, kxky, kyky  = kx*kx, kx*ky, ky*ky
    kSq   = kxkx + kyky
    kSqSq = kSq*kSq

    byx = numpy.fft.irfftn(1.j * kx  / (kSq+NumericalMethods.ME) * numpy.fft.rfftn(foo1))
    bxx = numpy.fft.irfftn(-1.j * ky / (kSq+NumericalMethods.ME) * numpy.fft.rfftn(foo1))

    bxy = numpy.fft.irfftn(-1.j * ky / (kSq+NumericalMethods.ME) * numpy.fft.rfftn(foo1))
    byy = numpy.fft.irfftn(1.j  * kx / (kSq+NumericalMethods.ME) * numpy.fft.rfftn(foo1))

    field[y,x] = GridArray.GridArray(byx)
    field[x,x] = GridArray.GridArray(bxx)
    
    field[x,y] = GridArray.GridArray(bxy)
    field[y,y] = GridArray.GridArray(byy)
    
    state.UpdateOrderParameterField(field)

    #pylab.imshow(state.CalculateRhoFourier()['z','x'])
    return state



def Relaxation_BetaPV(seed=0):
    N = 128
    gridShape = (N,N)
    
    gamma = 1. # Diffusion constant for vacancies
    alpha = 1. # Vacancy concentration constant.
    beta = 1.
    c0 = 0.1
    Lambda = 0

    #dynamics = VacancyDynamics.BetaP_VacancyDynamics(gamma=gamma,alpha=alpha,beta=beta)
    #dlabel = "VacancyDynamics"

    dynamics = VacancyDynamicsExternalStress(gamma=gamma, alpha=alpha,beta=beta)
    dlabel = "VacancyWithStress"

    #dynamics = ExternalStressDynamics(Lambda=Lambda)
    #dlabel = "UpwindWithStress"

    #dynamics = CentralUpwindHJBetaPDynamics.BetaPDynamics(Lambda=Lambda)
    #dlabel = "Upwind"

    dir = "aaa/"
    lengthscale = 0.2

    filename = dir+"POINTSOURCE_"+dlabel+"_L"+str(Lambda)+"_B"+str(beta)+"_G"+str(gamma)+"_A"+str(alpha)+"C0_"+str(c0)+"S_"+str(seed)+"2D"+str(N)+"ST_"+str(exstress)+"_yay.save"
    #state = FieldInitializer.GaussianRandomInitializer(gridShape,lengthscale,seed,vacancy=alpha)
    #if N != 128:
    #    state = FieldInitializer.ResizeState(state,N)
    #state = FieldInitializer.ReformatState(state)
    state = createBox(gridShape)
    state.betaP_V['s','s'] *= 0.
    state.betaP_V['s','s'] += c0
    
    mover = FieldMover.OperatorSplittingTVDRK_FieldMover(CFLsafeFactor=0.5,dtBound=1./N)
    #mover = FieldMover.TVDRungeKutta_FieldMover(CFLsafeFactor=0.1)

    obsState = Observer.RecallStateObserver()
    energychecking = TotalFreeEnergyDownhillObserver()
    #traceBetaP = TraceBetaPObserver(dir+"trace/trace")
    #vacancyObs = VacancyObserver(dir+"vacancies/vacancies")
    #memObs = MemoryObserver()

    startTime = 0. 
    endTime   = 15.
    dt = 1.

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

    while t<=(endTime):
        preT = t
        #"""
        if t<=0.1:
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


def ExternalLoading2D_CU(seed):
    N = 64
    gridShape = (N,N)

    Lambda = 0
    if Lambda == 0:
        motion = 'GlideClimb'
    else:
        motion = 'GlideOnly'

    strainrate = 0.0
    direction = numpy.array([-0.5,-0.5,1.0])
    Rate = strainrate*direction 
    Type = 'stress'
    Initial = 0.1*direction

    #lengthscale = 0.2 
    #relaxfile = "NewGlideOnly_ls0_28_S_"+str(seed)+"_2D"+str(N)+".save"
    #"FixV_CU_S_"+str(seed)+"_2D"+str(N)+"_lengthscale_"+str(lengthscale).replace(".","_")+"_"+motion+".save"
    #os.system("msscmd cd result, get "+relaxfile)
    #timef,state = FieldInitializer.LoadState(relaxfile)
    state = createBox(gridShape)

    #loadfile = "UNI_zz_S_"+str(seed)+"_rate_"+str(strainrate).replace('.','_')+"_CU_2D"+str(N)+'_betaP.save'
    loadfile = "UNI_BOX_zz_S_"+str(seed)+"_constantstress_"+str(0.03)+"_2D"+str(N)+"_betaP.save"

    FinialStrain = 2.
    dynamics = StrainLoadingBetaPDynamics(Lambda=Lambda,rate=Rate,type=Type,initial=Initial)
    mover = FieldMover.TVDRungeKutta_FieldMover(CFLsafeFactor=0.5,dtBound=0.01)
    obsState = Observer.RecallStateObserver()
    filename = loadfile 

    startTime = 0. 
    #endTime   = FinialStrain/strainrate
    endTime = 10.0
    t = startTime 
    dt = 0.05
    if startTime == 0. :
        recordState = Observer.RecordStateObserver(filename=filename)
        recordState.Update(t, state)
    else:
        T,state = FieldInitializer.LoadState(filename,t)
        print "we restart the simulation from T = "+str(T)
        recordState = Observer.RecordStateObserver(filename=filename,mode='a')
    system= PlasticitySystem.PlasticitySystem(gridShape, state, mover, dynamics, [obsState,Observer.VerboseTimestepObserver()])
    while t<=(endTime):
        preT = t
        t += dt
        system.Run(startTime=preT, endTime = t)
        system.state = obsState.state
        recordState.Update(t, system.state)


def main():
    #Relaxation_BetaPV()
    ExternalLoading2D_CU(0)
    
if __name__ == "__main__":
    main()
