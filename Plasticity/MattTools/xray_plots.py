import PlasticitySystem, FieldInitializer
import OrientationField
from Constants import *
from pylab import *
from scipy import *
import pylab

def plot_timeslice(filename, N, dim, time, Max=None):
    gs = 200
    cm = pylab.cm.hot_r
    t,s = FieldInitializer.LoadStateRaw(filename, N, dim, time)

    rod = s.CalculateRotationRodrigues()
    rho = s.CalculateRhoFourier().modulus()
    if len(s.gridShape) == 3:
        rho = rho[:,:,0]
        rod[x] = rod[x][:,:,0]
        rod[y] = rod[y][:,:,0]
        rod[z] = rod[z][:,:,0]
    crod = OrientationField.RodriguesToUnambiguousColor(rod[x], rod[y], rod[z])
    ps = 0.5 #max(abs(array([rod[x].min(), rod[x].max(), rod[y].min(), rod[y].max(), rod[z].min(), rod[z].max()])))
    ex = (-ps, ps)
    ey = (-ps, ps)
    ez = (-ps, ps)
    
    fig = figure(0)
    fig.clf()
    fxy = subplot(231)
    fxz = subplot(232)
    fyz = subplot(233)
    frho = subplot(234)
    frod = subplot(235)
 
    fxy.cla(); fxz.cla(); fyz.cla()
    if Max is not None:
        fxy.hexbin(rod[x].flatten(), rod[y].flatten(), gridsize=gs, extent=ex+ey, cmap=cm, vmax=Max)
        fxz.hexbin(rod[x].flatten(), rod[z].flatten(), gridsize=gs, extent=ex+ez, cmap=cm, vmax=Max)
        fyz.hexbin(rod[y].flatten(), rod[z].flatten(), gridsize=gs, extent=ey+ez, cmap=cm, vmax=Max)
    else:
        fxy.hexbin(rod[x].flatten(), rod[y].flatten(), gridsize=gs, extent=ex+ey, cmap=cm)
        fxz.hexbin(rod[x].flatten(), rod[z].flatten(), gridsize=gs, extent=ex+ez, cmap=cm)
        fyz.hexbin(rod[y].flatten(), rod[z].flatten(), gridsize=gs, extent=ey+ez, cmap=cm)


    frod.imshow(crod)
    frho.imshow(rho, vmin=rho.min(), vmax=rho.max())

#==========================================
#for ts in range(0, 1000, 10):
#    plot_timeslice("/b/plasticity/gcd2d1024/gcd2d1024_s0_debug_r1.plas", 1024, 2, ts)
#    savefig("slices%04d.png" % ts)
#    clf()
#    print ts

#===========================================
#file = "/b/plasticity/gcd2d1024/gcd2d1024_s0_debug_r1l.plas"
#t,s = FieldInitializer.LoadStateRaw(file, 1024, 2, 0)
#rod = s.CalculateRotationRodrigues()
#a = hexbin(rod['x'].flatten(), rod['y'].flatten())
#Max = a.get_array().max()
#
#for ts in arange(0, 20, 0.5):
#    plot_timeslice(file, 1024, 2, ts, Max)
#    savefig("xray_slices%04d.png" % int(ts/0.5))
#    clf()
#    print ts

file = "/b/plasticity/gcd2d1024/gcd2d1024_s0_debug_r1l.plas"
t,s = FieldInitializer.LoadStateRaw(file, 1024, 2, 0)
rod = s.CalculateRotationRodrigues()
a = hexbin(rod['x'].flatten(), rod['y'].flatten())
Max = a.get_array().max()

for ts in arange(0, 20, 0.5):
    plot_timeslice(file, 1024, 2, ts, Max)
    savefig("xray_slices%04d.png" % int(ts/0.5))
    clf()
    print ts
