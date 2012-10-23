import PlasticitySystem, FieldInitializer
import OrientationField
from Constants import *
from pylab import *
from scipy import *
import pylab
import scipy.ndimage as nd

ex = (0.075, 0.090)
ey = (-0.492, -0.48)
ez = (-0.5, 0.5)
gs = 200
cm = pylab.cm.hot_r

def plot_timeslice(filename, N, dim, time, Max=None):
    t,s = FieldInitializer.LoadStateRaw(filename, N, dim, time)

    rod = s.CalculateRotationRodrigues()
    rho = s.CalculateRhoFourier().modulus()
    if len(s.gridShape) == 3:
        rho = rho[:,:,0]
        rod[x] = rod[x][:,:,0]
        rod[y] = rod[y][:,:,0]
        rod[z] = rod[z][:,:,0]
    crod = OrientationField.RodriguesToUnambiguousColor(rod[x], rod[y], rod[z])
    
    fig = figure(0)
    fig.clf()
    fxy = subplot(231)
    fxz = subplot(232)
    fyz = subplot(233)
    frho = subplot(234)
    frod = subplot(235)
    fmask = subplot(236)
 
    fxy.cla(); fxz.cla(); fyz.cla()
    if Max is not None:
        fxy.hexbin(rod[x].flatten(), rod[y].flatten(), gridsize=gs, extent=ex+ey, cmap=cm, vmax=Max)
        fxz.hexbin(rod[x].flatten(), rod[z].flatten(), gridsize=gs, extent=ex+ez, cmap=cm, vmax=Max)
        fyz.hexbin(rod[y].flatten(), rod[z].flatten(), gridsize=gs, extent=ey+ez, cmap=cm, vmax=Max)
    else:
        fxy.hexbin(rod[x].flatten(), rod[y].flatten(), gridsize=gs, extent=ex+ey, cmap=cm)
        fxz.hexbin(rod[x].flatten(), rod[z].flatten(), gridsize=gs, extent=ex+ez, cmap=cm)
        fyz.hexbin(rod[y].flatten(), rod[z].flatten(), gridsize=gs, extent=ey+ez, cmap=cm)

    mrodx = ones_like(rod[x])
    mrody = ones_like(rod[y])
    mrodz = ones_like(rod[z])
    
    mrodx = (rod[x] > ex[0]) * (rod[x] < ex[1])
    mrody = (rod[y] > ey[0]) * (rod[y] < ey[1])
    mrodz = (rod[z] > ez[0]) * (rod[z] < ez[1])

    mask1 = ones_like(rod[x]) * mrodx * mrody * mrodz
    mask = (nd.laplace(mask1) > 1e-6) == False
    
    tcrod = ones_like(crod)
    trho = rho * mask
    tcrod[:,:,0] = crod[:,:,0] * mask
    tcrod[:,:,1] = crod[:,:,1] * mask
    tcrod[:,:,2] = crod[:,:,2] * mask
    frod.imshow(tcrod)
    frho.imshow(trho, vmin=trho.min(), vmax=trho.max())
    fmask.imshow(mask1, vmin=mask1.min(), vmax=mask1.max())

file = "/b/plasticity/gcd2d1024/gcd2d1024_s0_debug_r1l.plas"
t,s = FieldInitializer.LoadStateRaw(file, 1024, 2, 0)
rod = s.CalculateRotationRodrigues()
a = hexbin(rod['x'].flatten(), rod['y'].flatten(), extent=ex+ey, gridsize=gs)
Max = a.get_array().max()

for ts in arange(0, 20, 0.5):
    plot_timeslice(file, 1024, 2, ts, Max)
    savefig("xray_slices_lower_zoom%04d.png" % int(ts/0.5))
    clf()
    print ts

