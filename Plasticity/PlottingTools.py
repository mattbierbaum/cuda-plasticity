from Plasticity.FieldInitializers import FieldInitializer
from Plasticity.Constants import *
import numpy
import pylab
import pickle
from pylab import *
import copy

figDir = 'fig/'
THREEPANELFIG   = 0
RGBSCALEFIG     = 1      
GRAYSCALEFIG    = 2
STRAINSTRESSFIG = 3

class timestep:
    def __init__(self, startTime=0.0, endTime=0.1):
        self.t = startTime
        self.endTime = endTime

    def __iter__(self):
        return self

    def next(self):
        if self.t > self.endTime:
            raise StopIteration
        else:
            dt = self.dt()
            self.t += dt
            return self.t - dt

    def dt(self):
        dt = 1
        if self.t<=0.1:
            dt = 0.01
        elif self.t<=1.:
            dt = 0.05
        elif self.t<=5.:
            dt = 0.5
        else:
            dt = 2.5
        return dt

def CalculateVacancyConcentration(state):
    rho = state.CalculateRhoFourier()
    sig = state.CalculateSigma()

    num = numpy.zeros(state.gridShape)
    den = numpy.zeros(state.gridShape)

    for m in ['x','y','z']:
	for n in ['x','y','z']:
	    den += rho[m,n]*rho[m,n] - rho[m,n]*rho[n,m]

    for a in ['x','y','z']:
	for b in ['x','y','z']:
	    for c in ['x','y','z']:
		num += sig[a,b]*(rho[c,b]*rho[c,a] - rho[c,b]*rho[a,c])

    return num/den    

def OutputXY(x,y,filename):
    file = open(filename,'w+') 
    for i in range(len(x)):
        file.write('%.14f'%(x[i])+' ')
        file.write('%.14f'%(y[i])+' ')
        file.write("\n")
    file.close() 

def StrainStress(statefile,outputfile='strainstress',loadrate=0.01,loaddir=numpy.array([-0.5,-0.5,1.0]),dir='z',plot=True):
    """
    For instance, loadrate = 0.01; loaddir = numpy.array([-0.5,-0.5,1.]); dir = 'z';
    """
    mu,nu = 0.5,0.3 
    lamb = 2.*mu*nu/(1.-2.*nu)
    file = open(statefile)
    stress,strain = [],[]
    try:
	T = 0
        while True and T < 2/0.015 - 10:
            T = pickle.load(file)
            state = pickle.load(file)
            sigma = state.CalculateSigma(source='betaE')
            strain_tot = loaddir*loadrate*T
            strains = {x:strain_tot[0],y:strain_tot[1],z:strain_tot[2]}
            strain_tr = strains[x]+strains[y]+strains[z]
            for i in [x,y,z]:
                sigma[i,i] += lamb*strain_tr + 2.*mu*strains[i]
            stress.append(sigma[dir,dir].average())
            strain.append(T*loadrate)
    except EOFError:
        file.close()
    OutputXY(strain,stress,outputfile+'.dat')

    if plot:
	pylab.figure(STRAINSTRESSFIG)
        pylab.plot(strain,stress,'.-')
        pylab.savefig(outputfile+'.png')
        pylab.show()

    return strain, stress

def StatisticsRuns(prefix, postfix, seeds, loadrate, loaddir=numpy.array([-0.5,-0.5,1.0]), dir='z'):
    strains, stresses = [], []
    for s in seeds:
	filename = "%s%i%s" % (prefix, s, postfix)
	print filename
	a, b = StrainStress(filename, loadrate=loadrate, loaddir=loaddir, dir=dir)
	strains.append(a)
	stresses.append(b)

    return numpy.array(strains), numpy.array(stresses)

def LoadStates(filename,lists=None,tmax=None):
    file = open(filename)
    ts = []
    rhos,rods,sigmas = [],[],[]
    for i in lists:
        t,s = FieldInitializer.LoadState(filename,i)
        ts.append(t)
        rhos.append(s.CalculateRhoFourier().modulus())
        sigmas.append(s.CalculateSigma(source='rho').modulus())
        rodrigues = s.CalculateRotationRodrigues()
        rod = numpy.zeros((s.gridShape[0],s.gridShape[0],3),float)
        rod[:,:,0] = rodrigues[x]
        rod[:,:,2] = rodrigues[y]
        rod[:,:,1] = rodrigues[z]
        rods.append(rod)
    """
    try:
        i = 0
        while True:
            t = pickle.load(file)
            state = pickle.load(file)
            if (tmax!=None) and (t > tmax):
                break
            if lists is None:
                ts.append(t)
                rhos.append(state.CalculateRhoFourier().modulus())
                rodrigues = state.CalculateRotationRodrigues()
                rod = numpy.zeros((N,N,3),float)
                rod[:,:,0] = rodrigues[x]
                rod[:,:,2] = rodrigues[y]
                rod[:,:,1] = rodrigues[z]
                rods.append(rod)
                sigmas.append(state.CalculateSigma(source='rho').modulus())
            else:
                if i in lists:
                    ts.append(t)
                    rhos.append(state.CalculateRhoFourier().modulus())
                    rodrigues = state.CalculateRotationRodrigues()
                    rod = numpy.zeros((N,N,3),float)
                    rod[:,:,0] = rodrigues[x]
                    rod[:,:,2] = rodrigues[y]
                    rod[:,:,1] = rodrigues[z]
                    rods.append(rod)
                    sigmas.append(state.CalculateSigma(source='rho').modulus())
                i += 1
    except EOFError:
        file.close()
    """
    return ts,rhos,rods,sigmas
 
def PlotAll2Dimages(filename):
    """
    This method generates Fig.1(a)~(d), Fig.4(a)-(b) and SFig.2(a)-(b).
    """
    #newdataDir = 'No Backup/FixVData/'
    #figDir = 'No Backup/FixVData/LargeData/Figure/'
    t,state1 = FieldInitializer.LoadState(filename)
    t,state2 = FieldInitializer.LoadState(filename,0.)
    N = state1.gridShape[0]
    rodrigues1 = state1.CalculateRotationRodrigues()
    rod1 = numpy.zeros((N,N,3),float)
    rod1[:,:,0] = rodrigues1[x]
    rod1[:,:,2] = rodrigues1[y]
    rod1[:,:,1] = rodrigues1[z]
    rodrigues2 = state2.CalculateRotationRodrigues()
    rod2 = numpy.zeros((N,N,3),float)
    rod2[:,:,0] = rodrigues2[x]
    rod2[:,:,2] = rodrigues2[y]
    rod2[:,:,1] = rodrigues2[z]
    data = numpy.array([rod1,rod2])
    Range = SetColorScaleRange(numpy.array([rod1]),frac=0.005)
    maxRange = (Range[1]-Range[0])
    center = [numpy.average(data[:,:,:,0]),numpy.average(data[:,:,:,1]),numpy.average(data[:,:,:,2])]
    return maxRange,center
 

def StressEvolutionPlot(filename, RhoRange=None):
    """
    This method generates Movie 3.
    """
    figDir = 'fig/'

    times  = []
    stress = []
    for i in timestep(0,15):
	t,s = FieldInitializer.LoadState(filename, i)
	times.append(t)
	stress.append(s.CalculateSigma(source='rho').modulus().max())

    pylab.figure()
    pylab.plot(times, stress, 'o-')
    return times, stress

def StrainedEvolutionMovie(filename, RhoRange=None):
    """
    This method generates Movie 3.
    """
    N = 128
    maxRange,center = PlotAll2Dimages(filename)
    figDir = 'fig/'

    #dataDir = 'No Backup/FixVData/Loading/'
    #filename = dataDir+"No_dtmax_UNI_zz_S_2_rate_0_05_CU_2D512_betaP.save"
    type = 'Strained'
    #filename = dataDir+"256/ZZLoading_S_0_rate_0_05_CU_2D256_betaP.save"

    lists = list(numpy.arange(0,60,1))#time list
    #lists = list(timestep(0,20))
    ts,rhos,rods,sigmas = LoadStates(filename,lists,60.)
    sigmamin,sigmamax = numpy.array(sigmas).min(),numpy.array(sigmas).max()
    print "sigma ratio of strained state is ",sigmamax/sigmamin
    if RhoRange is None:
        RhoRange = SetColorScaleRange(numpy.array([rhos[-1]]),0.01)
    for i in range(len(ts)):
        #GenerateSinglePic(sigmas[i],(sigmamin,sigmamax),figDir+'Strained'+str(N)+'_sigma_t_'+str(ts[i]).replace('.','_'),False)
        rodcolormap = ConvertToRGB(rods[i],maxRange,center)    
        rodcolormap = rodcolormap.astype('uint8')
        #GenerateSinglePic(colormap,None,figDir+'Strained'+str(N)+'_rod_t_'+str(ts[i]).replace('.','_'),True)
        Rhomap = ConvertToGrayRScale(rhos[i],(RhoRange[1]-rhos[i].min()),(RhoRange[1]+rhos[i].min())/2.,'GC')
        #GenerateSinglePic(Rhomap,None,figDir+'Strained'+str(N)+'_rho_t_'+str(ts[i]).replace('.','_'),True)
        pylab.figure(THREEPANELFIG, figsize=(5.12*3,6.12))
        pylab.clf()
        pylab.subplot(131)
        pylab.imshow(Rhomap.astype('uint8'),interpolation='nearest')
        pylab.xticks([])
        pylab.yticks([])
        pylab.xlabel(r'$||\rho||$',fontsize=25)
        pylab.subplot(132)
        pylab.imshow(rodcolormap.astype('uint8'),interpolation='nearest')
        pylab.xticks([])
        pylab.yticks([])
        pylab.xlabel('Crystalline orientation map',fontsize=25)
        pylab.subplot(133)
        pylab.imshow(sigmas[i],vmin=sigmamin,vmax=sigmamax,interpolation='nearest')
        pylab.xticks([])
        pylab.yticks([])
        pylab.xlabel(r'$||\sigma_{int}||$',fontsize=25)
        pylab.suptitle(r'$\epsilon_{zz}= %.3f\;\beta_0$'%(lists[i]*0.015),fontsize=20)
        pylab.subplots_adjust(0.,0.,1.,1.,0.01,0.05)
        if i >=10:
            pylab.savefig(figDir+type+str(i)+'.png')
        else:
            pylab.savefig(figDir+type+'0'+str(i)+'.png')
 
def SetColorScaleRange(data,frac=0.01):
    newdata = data.flatten()
    newdata = numpy.sort(newdata) 
    cutpos = int(len(newdata)*frac)
    rangemin = newdata[cutpos]
    rangemax = newdata[-cutpos-1]
    return (rangemin,rangemax)

def ConvertToRGB(vector,maxRange,centerR,filename=None,ThreeD=False):
    colormap = numpy.zeros(vector.shape,float)
    if ThreeD:
        colormap[:,:,:,0] = 255.*(0.5+(vector[:,:,:,0]-centerR[0])/maxRange)
        colormap[:,:,:,1] = 255.*(0.5+(vector[:,:,:,1]-centerR[1])/maxRange)
        colormap[:,:,:,2] = 255.*(0.5+(vector[:,:,:,2]-centerR[2])/maxRange)
        colormap = colormap.astype('int')
        colormap = colormap*(colormap<=255)*(colormap>=0)+(colormap>255)*255
        return colormap
    else:
        colormap[:,:,0] = 255.*(0.5+(vector[:,:,0]-centerR[0])/maxRange)
        colormap[:,:,1] = 255.*(0.5+(vector[:,:,1]-centerR[1])/maxRange)
        colormap[:,:,2] = 255.*(0.5+(vector[:,:,2]-centerR[2])/maxRange)
        colormap = colormap.astype('int')
        colormap = colormap*(colormap<=255)*(colormap>=0)+(colormap>255)*255
        if filename is not None:
            index = {0:'R',1:'G',2:'B'}
            for i in range(3):
                pylab.figure(RGBSCALEFIG)
                pylab.hist(colormap[:,:,i].flatten(),bins=500,normed=True)
                pylab.savefig(figDir+filename+'_ColorHist_'+index[i]+'.png')
        return colormap

def ConvertToGrayRScale(scalar,maxRange,centerR,filename=None,amp=255.,shift=0.):
    colormap = numpy.zeros(list(scalar.shape)+[3],float)
    colormap[:,:,0] = amp*(0.5-(scalar-centerR)/maxRange)
    colormap[:,:,1] = amp*(0.5-(scalar-centerR)/maxRange)
    colormap[:,:,2] = amp*(0.5-(scalar-centerR)/maxRange)
    colormap += shift 
    colormap = colormap.astype('int')
    colormap = colormap*(colormap>=0)*(colormap<=255)+(colormap>255)*255
    if filename is not None:
        pylab.figure(GRAYSCALEFIG)
        pylab.clf()
        pylab.hist(colormap.flatten(),bins=500,normed=True)
        pylab.savefig(figDir+filename+'_ColorHist.png')
    return colormap


