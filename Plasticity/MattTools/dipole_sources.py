from scipy import *
from pylab import *
import numpy.fft as fft 
import scipy.special as funcs
import scipy.ndimage as nd
import PlasticitySystem, FieldInitializer
import CentralUpwindHJ
import VacancyDynamics
import PlottingToolsForMatt as PTFM

rr = abs(arange(-64,65,0.1))
def paper_single(): 
    rc('figure.subplot', bottom=0.14)
    rc('xtick.major', size=6)
    rc('xtick.minor', size=3)
    rc('ytick.major', size=6)
    rc('ytick.minor', size=3)
    rc('font', size=22)
    rc('xtick', labelsize='small')
    rc('ytick', labelsize='small')
    rc('axes', labelsize=26)
    rc('axes.formatter', limits=[-3,3])
    rc('lines', linewidth=2.0, markersize=8)

paper_single()

def find_velocity(filename, tf):
    for t in PTFM.timestep(0,tf):
        t0,s = FieldInitializer.LoadState(filename,t)
        v = PTFM.CalculateVelocityMinusVacancies(s)
        print t0, v['x'].max(), s.betaP_V['s','s'].max()
    t,s = FieldInitializer.LoadState(filename, t)
    imshow(s.CalculateRhoFourier().modulus())
    
def matchlogtoexpn_score(p):
    m, r0, off = p
    return ((funcs.expn(2,rr/20.) - m*log(r0/rr) + off)**2).sum()

def matchlogtoexpn(p0 = [30, 2e2, -20]):
    import scipy.optimize as opt
    p = opt.fmin(matchlogtoexpn_score, p0, xtol=1e-9, ftol=1e-9)
   
    m, r0, off = p 
    plot(funcs.expn(2,rr/20.))
    plot(m*log(r0/rr) - off)
    return p

# create one theoretical point source
def one_source(x0, y0, mag, rad, core, grid, type=0):
    r = fromfunction(lambda x,y: sqrt((x-x0)**2 + (y-y0)**2), grid)

    if type == 0:
        # log sources 
        #[  1.60121182e-06   7.82761030e+03  -6.36886579e-01]
        s = mag*log(rad/r) + core #mag*log(1/(rad+r))+core
    elif type == 1:
        # from the 3d diffusion green's function
        #[  3.24852149e-05   8.97514165e+00  -6.11534255e-02]
        s = mag*funcs.erf(r/rad)/r + core
    elif type == 2:
        # just a plain 'ol gaussian
        s = mag*exp(-r**2/rad) + core 
    elif type == 3:
        # from the 2d green's function
        #[  5.99160034e-06   2.39560167e+01   9.78734951e-03]
        s = mag*funcs.expn(2,r/rad) + core
    else:
        s = mag/(r+rad) + core
    return s 

def all_sources(mag, rad, core, grid, type=0):
    g = grid[0]
    mid = g / 2
    offset = grid[0]/24 
    p1 = mid - offset
    p2 = mid + offset
    #core = 0.001
    
    #first, the left ones
    s11 =  one_source(p1-core, mid, mag, rad, core, grid, type)
    s12 = -0*one_source(p1+core, mid, mag, rad, core, grid, type)

    #then, the right ones
    s21 = -0*one_source(p2-core, mid, mag, rad, core, grid, type)
    s22 =  one_source(p2+core, mid, mag, rad, core, grid, type)

    #then, the top ones
    s31 = -one_source(mid, p1-core, mag, rad, core, grid, type)
    s32 =  0*one_source(mid, p1+core, mag, rad, core, grid, type)

    #last, the bottom ones
    s41 =  0*one_source(mid, p2-core, mag, rad, core, grid, type)
    s42 = -one_source(mid, p2+core, mag, rad, core, grid, type)

    #s = one_source(mid-0.01, mid-0.01, mag, rad, core, grid, type)
    return s11+s12 + s21+s22 + s31+s32 + s41+s42


N = 128 
#filename = "TESTINGVACANCIES_4PT_VacancyWithStress_yy-xx_G0.158314349441_A1_C0_STR0.005_L0_S0_2D128.save"
#filename = "TESTINGVACANCIES_4PT_VacancyWithStress_yy-xx_G1000.0_A1000_C0_STR0.05_L0_S0_2D128.save"
#filename = "TESTINGVACANCIES_4PT_VacancyWithStress_yy-xx_G6.3165468167_A1000_C0_STR0.005_L0_S0_2D128.save"
#filename = "TESTINGVACANCIES_4PT_VacancyWithStress_yy-xx_G1.58314349441_A1_C0_STR0.005_L0_S0_2D128.save"
# goood one
filename = "TESTINGVACANCIES_4PT_VacancyWithStress_yy-xx_G0.01_A100.0_C0_STR0.05_L0_S0_2D128.save"
#filename = "TESTINGVACANCIES_4PT_VacancyWithStress_yy-xx_G0.01_A10.0_C0_STR0.05_L0_S0_2D512.save"


t,s = FieldInitializer.LoadState(filename, 0.40)
tm,sm = FieldInitializer.LoadState(filename,0)
cfield = s.betaP_V['s','s']
mask = ones((N,N))
#mask = (s.CalculateRhoFourier().modulus() < 1).astype('int')
#mask = fromfunction(lambda x,y: 1-((x>N/3)*(x<2*N/3))*((y>N/3)*(y<2*N/3)), s.gridShape)

def PLOT_vacancyEIfit():
    filename = "TESTINGVACANCIES_4PT_VacancyWithStress_yy-xx_G0.01_A100.0_C0_STR0.05_L0_S0_2D128.save"
    t,s = FieldInitializer.LoadState(filename, 0.40)
    cfield = s.betaP_V['s','s']
    mask = ones((N,N))
    f = run_fitter(type=3)
    

"""
def simulate_constantsource(T=1000):
    source = fromfunction(lambda x,y: (x==N/2)*(y==N/2), (N,N))
    #source = s.CalculateRhoFourier()['z','y'].abs()-s.CalculateRhoFourier()['z','x'].abs()
    f = source*0
    ksq = s.ktools.kSq

    for i in range(T):
        #f[N/2,N/2] = 1
        f = f + source*0.01
        f = fft.irfftn(fft.rfftn(f) * exp(-ksq * 1e-6))

    return f

cfield = simulate_constantsource(5e5)

sigxx = s.CalculateSigma()['x','x']
sigyy = s.CalculateSigma()['y','y']
J = rhozx**2*sigxx + rhozy**2*sigyy
"""

def PLOT_results():
    filename = "TESTINGVACANCIES_4PT_VacancyWithStress_yy-xx_G0.01_A100.0_C0_STR0.05_L0_S0_2D128.save"
    x = arange(0,1.,1./N)
    s0 = sources_from_params(p=[  9.81528469e-05 ,  4.77205378e+05,  1.34815841e+01],type=0)
    s3 = sources_from_params(p=[  4.87346077e-04 , 1.14597572e+01,  1.47292858e+01],type=3)

    plot(x,cfield[:, N/2], 'go', label='data')
    plot(x, s0[:,N/2], 'b--', label="fit - log")
    plot(x, s3[:,N/2], 'r-', label="fit - Ei")
    ylim(0, 5e-4)
    xlabel(r'$x$')
    ylabel(r'$c$')
    title("Fit to vacancy profile")

def score(params,type):
    mag, rad, core = params
    return (mask*(all_sources(mag, rad, core, (N,N), type) - cfield)**2).sum()
    
def sources_from_params(p,type):
    return all_sources(p[0], p[1], p[2], (N,N), type)#*mask

def run_fitter(p0 = [5e-5,3,2],type=0):
    # results for 256 file
    # pretty good [-7e-7, 1, 0.1] => [-1.54e-05, 1.13e1, -2.96e-2]
    # even better [5e-5, 30, 2]   => [1.92e-7, 63.7, 1.35]
    # best so far with r^-1       => [3.22e-11, 1711, 2.5]
    # even better with r^-2       => [-5.92e-6,  1.09, 1.25]

    # finally, with the erf       => [-3.6e-7, 16, 1]
    # erf with a core factor      => [-4.4e-7, 17.6, 1.1]
    
    import scipy.optimize as opt
    p = opt.fmin(score, p0, args=(type,), xtol=1e-9, ftol=1e-9)
    
    print "Found the parameters, ", p
    s = sources_from_params(p,type)

    figure()
    plot(s[:, N/2])
    plot(cfield[:, N/2], 'o')

    max = array([s.max(),cfield.max()]).max()
    min = array([cfield.min(), s.min()]).max()
    
    figure(); imshow(s, vmax=max, vmin=min)
    #figure(); imshow(cfield, vmax=max, vmin=min)
    #figure(); imshow(s-cfield, vmax=max, vmin=min)
    
    show()
    return p


"""
# this is to check whether the gaussian initial
# conditions can account for the lack of discontinuity 
# present in the fit
def calculate_cfieldfromrho(rho, mag, gamma):
    field = 0*rhos[0]
    for i in range(0,11):
        field += mag*nd.gaussian_filter(rhos[i], sqrt(2*gamma*(t-ts[i])),mode='wrap')/(4*gamma*pi*(t-ts[i]))
    return field

def all_sources2(rho, mag, gamma):
    return calculate_cfieldfromrho(rho, mag, gamma)

def score(params):
    mag, gamma = params
    return ((all_sources2(rho, mag, gamma) - cfield)**2).sum()

def sources_from_params(p):
    return all_sources2(rho,p[0],p[1])

def run_fitter(p0 = [5e-5,1]):
    import scipy.optimize as opt
    p = opt.fmin(score, p0, xtol=1e-9, ftol=1e-9)
    
    print "Found the parameters, ", p
    s = sources_from_params(p)

    figure()
    plot(s[:, N/2])
    plot(cfield[:, N/2], 'o')

    max = array([s.max(),cfield.max()]).max()
    min = array([cfield.min(), s.min()]).max()
    
    figure(); imshow(s, vmax=max, vmin=min)
    figure(); imshow(cfield, vmax=max, vmin=min)
    figure(); imshow(s-cfield, vmax=max, vmin=min)
    
    show()
    return p
"""


