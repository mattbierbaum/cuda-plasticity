import numpy as np
import powerlaw
import pylab as pl
from functools import partial
from Plasticity.Observers.BasicStatisticalTools import FittingPowerLawForHistogram

def CalculateSinglePowerLaw(grains, cmin=35, cmax=180, bins=np.logspace(1,3, 100)):
    y,x = np.histogram(grains, bins=bins)

    x = (x[:-1] + x[1:])/2
    A, alpha = FittingPowerLawForHistogram([x,y], 1, cmin, cmax)

    xi = abs(x-(cmin+cmax)/2).argmin()
    scale = y[xi] / x[xi]**alpha

    print alpha
    pl.hist(grains, bins=bins, histtype='step')
    pl.loglog()
    pl.plot(x, scale * x**alpha)


CalculateSinglePowerLaw2D = partial(CalculateSinglePowerLaw, cmin=35, cmax=180, bins=np.logspace(1,3, 100))
CalculateSinglePowerLaw3D = partial(CalculateSinglePowerLaw, cmin=750, cmax=3600, bins=np.logspace(2,4, 80))

