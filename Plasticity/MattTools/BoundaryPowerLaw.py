import numpy as np
import scipy as sp
from scipy.special import gamma

import powerlaw
import pylab as pl
from functools import partial
from Plasticity.Observers.BasicStatisticalTools import FittingPowerLawForHistogram
from boundarypruningutil import *

def setPlotOptions(labelsize=22,tickmajor=10,tickminor=7,markersize=10,legendsize=20,legendspacing=1.5,labelsizexy=16):
    pl.rcdefaults()
    pl.rcParams.update({'xtick.labelsize':labelsizexy,
            'xtick.major.size':tickmajor,
            'xtick.minor.size':tickminor,
            'ytick.labelsize':labelsizexy,
            'ytick.major.size':tickmajor,
            'ytick.minor.size':tickminor,
            'lines.markersize':markersize,
            'axes.labelsize':labelsize,
            'legend.fontsize':legendsize,
            'legend.columnspacing':legendspacing,
            'lines.linewidth': 1.5,
            'figure.autolayout': True,
        })

setPlotOptions()


def CalculateSinglePowerLaw(grains, cmin=35, cmax=180, bins=np.logspace(1,3, 100)):
    y,x = np.histogram(grains, bins=bins)

    x = (x[:-1] + x[1:])/2
    fit, error = FittingPowerLawForHistogram([x,y], 1, cmin, cmax)

    xi = abs(x-(cmin+cmax)/2).argmin()
    scale = y[xi] / x[xi]**fit[1]

    pl.figure()
    pl.hist(grains, bins=bins, histtype='step')
    pl.loglog()
    pl.plot(x, scale * x**fit[1])
    return fit, error

CalculateSinglePowerLaw2D = partial(CalculateSinglePowerLaw, cmin=39, cmax=180, bins=np.logspace(1,3, 100))
CalculateSinglePowerLaw3D = partial(CalculateSinglePowerLaw, cmin=750, cmax=3600, bins=np.logspace(2,4, 80))


def run_2d_timeslices(type=0):
    import cPickle as pickle
    f = []
    e = []
    times = np.arange(0,160,10)
    for t in times:
        z = pickle.load(open("./bp_t_%0.3f/prune_all.pickle" % t))
        #fit, error = CalculateSinglePowerLaw2D(np.hstack([z['grain'][i] for i in xrange(3)]))
        fit, error = CalculateSinglePowerLaw2D(z['grain'][type])
        f.append(fit)
        e.append(error)

    f = np.array(f)
    e = np.array(e)

    pl.figure()
    pl.errorbar(0.01*times, f[:,1], yerr=e[:,1], marker='o', fmt='o')


def SpinAndEvolveCluster(indexmap, size=0):
    from mayavi import mlab
    mlab.figure()

    cindex = MayaVIPlotCluster(indexmap, size)
    com = tuple((ClusterCOM(indexmap, cindex)%128).astype('int'))
    #indexmap[cindex]

    numclusters = 10
    rotstep = 5
    rotcurr = 0
    rotint = 360 / numclusters

    i = 0
    for t in np.arange(0, 40, 1):
        mlab.clf()
        temp = pickle.load(open("./bp_timeseries_lvp3d128/lvp3d128_s0_d0_3d_t=%03d_3e-07_1.2.indexmap.pickle" % int(t))).reshape(128,128,128)
        #temp = pickle.load(open("./bp_t_%0.3f/mdp3d128_s3_d0_3d_3e-07_1.2.indexmap.pickle" % t)).reshape(128,128,128)
        MayaVIPlotCluster(temp, clusterindex=temp[com])
        i, rotcurr = AnimateOpenPlot(rotcurr, rotcurr+rotint, rotstep, i)


def MisorientationScalingCollapse(mis, bdlength, alpha=2.5):
    """ good values for alpha seem to be 4, but 2.5 for experiment """

    t = mis*bdlength/(mis*bdlength).mean()
    dx = 5./100.
    y,x = np.histogram(t, bins=np.linspace(0, 5, 100))
    x = (x[:-1]+x[1:])/2
    y = y.astype('float')/y.sum() / dx
    pl.plot(x, y, 'o')

    #t = mis/(mis).mean()
    #dx = 5./100.
    #y,x = np.histogram(t, bins=np.linspace(0, 5, 100))
    #x = (x[:-1]+x[1:])/2
    #y = y.astype('float')/y.sum() / dx
    #pl.plot(x, y, 'o-')

    alpha = 2.5
    scaling = lambda x, alpha: alpha**alpha / gamma(alpha) * x**(alpha-1) * np.exp(-alpha*x)
    xt = np.linspace(0,5, 1000)
    pl.plot(xt, scaling(xt, alpha), 'r-', label=r"Fit, $\alpha$ = 2.5")

    alpha = 4.8
    pl.plot(xt, scaling(xt, alpha), 'k-', label=r"Fit, $\alpha$ = 4.8")

    pl.xlabel(r"$\theta / \theta_{av}$")
    pl.ylabel(r"$\theta_{av}\,P(\theta, \theta_{av})$")


def MisorientationScalingCollapseCompare(misses, bdlengths, labels, alpha=2.5):
    """ good values for alpha seem to be 4, but 2.5 for experiment """

    for mis, label, bdlength in zip(misses, labels, bdlengths):
        t = mis*bdlength/(mis*bdlength).mean()
        dx = 5./100.
        y,x = np.histogram(t, bins=np.linspace(0, 5, 100))
        x = (x[:-1]+x[1:])/2
        y = y.astype('float')/y.sum() / dx
        pl.plot(x, y, 'o', label=label)

    #t = mis/(mis).mean()
    #dx = 5./100.
    #y,x = np.histogram(t, bins=np.linspace(0, 5, 100))
    #x = (x[:-1]+x[1:])/2
    #y = y.astype('float')/y.sum() / dx
    #pl.plot(x, y, 'o-')

    alpha = 2.5
    scaling = lambda x, alpha: alpha**alpha / gamma(alpha) * x**(alpha-1) * np.exp(-alpha*x)
    xt = np.linspace(0,5, 1000)
    pl.plot(xt, scaling(xt, alpha), 'b-', label=r"Fit, $\alpha$ = 2.5")

    alpha = 4.8
    pl.plot(xt, scaling(xt, alpha), 'g-', label=r"Fit, $\alpha$ = 4.8")

    pl.xlabel(r"$\theta / \theta_{av}$")
    pl.ylabel(r"$\theta_{av}\,P(\theta, \theta_{av})$")


def GrainSizeCollapseCompare(grains, labels, alpha=2.5):
    """ good values for alpha seem to be 4, but 2.5 for experiment """

    for grain, label in zip(grains, labels):
        t = np.sqrt(grain)/np.sqrt(grain).mean()
        dx = 5./100.
        #y,x = np.histogram(t, bins=np.logspace(-1, 0.8, 100))
        y,x = np.histogram(t, bins=np.linspace(0, 5, 100))
        x = (x[:-1]+x[1:])/2
        y = y.astype('float')/y.sum() / dx
        pl.plot(x, y, 'o-', label=label)

    #fit, error = FittingPowerLawForHistogram([x,y], 1, 1, 2.7)

    #print error
    #xi = 24
    #scale = y[xi] / x[xi]**fit[1]
    #pl.plot(x, scale * x**fit[1], 'k-', label=r"Fit, $\alpha$ = 2.5 $\pm$ 0.1")

    pl.xlabel(r"$D / D_{av}$")
    pl.ylabel(r"$D_{av}\,P(D, D_{av})$")
    pl.gcf().subplots_adjust(bottom=0.13)


def GrainSizeCollapseCompareInset(grains, labels, alpha=2.5):
    """ good values for alpha seem to be 4, but 2.5 for experiment """
    colors = ['b', 'r', 'g']

    for i, (grain, label) in enumerate(zip(grains, labels)):
        t = np.sqrt(grain)/np.sqrt(grain).mean()
        dx = 3.5/100.
        #y,x = np.histogram(t, bins=np.logspace(-1, 0.8, 100))
        y,x = np.histogram(t, bins=np.linspace(0, 3.5, 80))
        x = (x[:-1]+x[1:])/2
        y = y.astype('float')/y.sum() / dx
        pl.plot(x, y, colors[i]+'o-', label=label)

    pl.xlabel(r"$D / D_{av}$")
    pl.ylabel(r"$D_{av}\,P(D, D_{av})$")
    pl.gcf().subplots_adjust(bottom=0.13)
    pl.legend(loc='lower right')

    ax = pl.axes([0.52, 0.52, 0.35, 0.35])
    for i, (grain, label) in enumerate(zip(grains, labels)):
        t = np.sqrt(grain)/np.sqrt(grain).mean()
        dx = 3.5/100.
        #y,x = np.histogram(t, bins=np.logspace(-1, 0.8, 100))
        y,x = np.histogram(t, bins=np.linspace(0, 3.5, 80))
        x = (x[:-1]+x[1:])/2
        y = y.astype('float')/y.sum() / dx
        pl.plot(x, y, colors[i]+'o-')#, label=label)

        fit, error = FittingPowerLawForHistogram([x,y], 1, 1, 2.7)
        print fit, error
        print abs(error/fit)
        if abs(error[1] / fit[1]) < 0.045:
            xi = 24
            scale = y[xi] / x[xi]**fit[1]
            pl.plot(x, scale * x**fit[1], 'k-', label=r"Fit, $\alpha$ = %0.1f $\pm$ %0.1f" % (fit[1], error[1]))
            pl.text(x[xi], scale* x[xi]**fit[1], r"   $\alpha$ = %0.1f $\pm$ %0.1f" % (fit[1], error[1]))

    pl.loglog()
    pl.xlim(0.254, 6.99)
    pl.ylim(0.00418, 2.834)
    #pl.xticks([])
    #pl.yticks([])
    #pl.legend(fontsize=10)

scaling = lambda alpha, x: alpha**alpha / gamma(alpha) * x**(alpha-1) * np.exp(-alpha*x)

def fit_error(alpha, x, y):
    return np.sum((y-scaling(alpha, x))**2)

def fit_alpha(x, y):
    from scipy.optimize import fmin
    alpha = fmin(fit_error, [2], args=(x,y))
    return alpha

def MisorientationScalingCollapseCompareInset(misses, bdlengths, labels, alpha=2.5):
    """ good values for alpha seem to be 4, but 2.5 for experiment """
    colors = ['b', 'r', 'g']

    pl.rcParams.update({'legend.fontsize': 14,
            'legend.columnspacing':1.2,
        })
    for i, (mis, label, bdlength) in enumerate(zip(misses, labels, bdlengths)):
        t = mis*bdlength/(mis*bdlength).mean()
        dx = 5./100.
        y,x = np.histogram(t, bins=np.linspace(0, 5, 100))
        x = (x[:-1]+x[1:])/2
        y = y.astype('float')/y.sum() / dx
        pl.plot(x, y, colors[i]+'o--', label=label)

        alpha = fit_alpha(x, y)
        xt = np.linspace(0,5, 1000)
        pl.plot(xt, scaling(alpha, xt), colors[i]+'-', label=r"Fit, $\alpha$ = %0.1f" % alpha)

    pl.xlabel(r"$\theta / \theta_{av}$")
    pl.ylabel(r"$\theta_{av}\,P(\theta, \theta_{av})$")
    pl.legend(loc='lower right')

    ax = pl.axes([0.52, 0.52, 0.35, 0.35])
    for i, (mis, label, bdlength) in enumerate(zip(misses, labels, bdlengths)):
        t = mis*bdlength/(mis*bdlength).mean()
        dx = 5./100.
        y,x = np.histogram(t, bins=np.linspace(0, 5, 100))
        x = (x[:-1]+x[1:])/2
        y = y.astype('float')/y.sum() / dx
        pl.plot(x, y, colors[i]+'o-', label=label)
    pl.loglog()

    return x, y

def PlotTimeEvolutionAverages():
    times = []
    dav, dstd = [], []
    mav, mstd = [], []
    for t in np.arange(0, 100, 10):
        #tmp = pickle.load(open("./bp_t_%0.3f/prune_all3.pickle" % t))
        tmp = pickle.load(open("./bp_t_%0.3f/prune_all_2.5e-06_1.2e+00.pickle" % t))
        g = tmp['grain'][0]
        l = tmp['length'][0]
        m = tmp['mis'][0]

        times.append(t)
        dav.append(g.mean())
        mav.append((m*l).mean())
        dstd.append(g.std())
        mstd.append((m*l).std())

    return times, dav, mav, dstd, mstd

