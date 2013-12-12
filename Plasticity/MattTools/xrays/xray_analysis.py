import pylab
from pylab import *
from scipy import *
from Constants import *
from matplotlib.widgets import RectangleSelector
import scipy.ndimage as nd

import PlasticitySystem
import FieldInitializer
import OrientationField

filename = "/b/plasticity/template/gcd2d1024_s0_debug_r1l.plas"
N = 1024
dim = 2
time = 15


state = None
ps = 0.05
rod = None
rho = None
crod = None
rx = None
ry = None
rz = None
point = array([0,0])

fig = figure(figsize=(0.7*5.12*3, 0.7*6.12*2))
fxy = subplot(231)
fxz = subplot(232)
fyz = subplot(233)
frho = subplot(234)
frod = subplot(235)
fmask = subplot(236)
fpoint = None

def line_select_callback_xy(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    global rx, ry
    rx=sort((x1,x2))
    ry=sort((y1,y2)) 
    update_plot()

def line_select_callback_xz(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    global rx, rz
    rx=sort((x1,x2))
    rz=sort((y1,y2))
    update_plot()

def line_select_callback_yz(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    global ry, rz
    ry=sort((x1,x2))
    rz=sort((y1,y2))
    update_plot()

def toggle_selector_xy(event):
    global rx, ry, rz, time
    dt = 1 #0.05
    movie_dt = 20
    if event.key in ['Q', 'q']:
        rx = None
        ry = None
        update_plot()
    if event.key in ['=', '+']:
        time = time + dt
        get_components()
        update_plot()
    if event.key in ['-', '_']:
        time = time - dt
        if time < 0:
            time = 0
        get_components()
        update_plot()
    if event.key in ['m']:
        ts = time
        for time in arange(ts, ts+movie_dt, dt):
            get_components()
            update_plot()
            save_fig((time-ts)/dt)
    if event.key in ['t']:
        print "Enter the time"
        time = integer(raw_input())
        get_components()
        update_plot()

def toggle_selector_xz(event):
    global rx, ry, rz
    if event.key in ['Q', 'q']:
        rx = None
        rz = None
        update_plot()

def toggle_selector_yz(event):
    global rx, ry, rz
    if event.key in ['Q', 'q']:
        ry = None
        rz = None
        update_plot()

def onclick(event):
    global point
    if event.inaxes == fmask or event.inaxes == frod:
        point=(event.xdata, event.ydata)
        update_lower_plot()

toggle_selector_xy.RS = RectangleSelector(fxy,line_select_callback_xy,drawtype='box',useblit=True, button=[1,3])
toggle_selector_xz.RS = RectangleSelector(fxz,line_select_callback_xz,drawtype='box',useblit=True, button=[1,3])
toggle_selector_yz.RS = RectangleSelector(fyz,line_select_callback_yz,drawtype='box',useblit=True, button=[1,3])

connect('key_press_event', toggle_selector_xy)
connect('key_press_event', toggle_selector_xz)
connect('key_press_event', toggle_selector_yz)

fig.canvas.mpl_connect("button_press_event", onclick)
show()

def get_components():
    global rod, rho, crod, ps, state, time, filename, N, dim
    t, state = FieldInitializer.LoadStateRaw(filename, N, dim, time)
    print "loading time", time, "as", t
    rod = state.CalculateRotationRodrigues()
    rho = state.CalculateRhoFourier().modulus()
    if len(state.gridShape) == 3:
        rho = rho[:,:,0]
        rod[x] = rod[x][:,:,0]
        rod[y] = rod[y][:,:,0]
        rod[z] = rod[z][:,:,0]
    crod = OrientationField.RodriguesToUnambiguousColor(rod[x], rod[y], rod[z])
    ps = max(abs(array([rod[x].min(), rod[x].max(), rod[y].min(), rod[y].max(), rod[z].min(), rod[z].max()])))
"""
def get_components():
    global rod, rho, crod, ps, state
    rod = {}
    rho = zeros((N,N))
    rho[:900,:] = 0.5
    rho[:900,:900] += 0.5
    rod[x] = rho.copy()
    rod[y] = rho.copy()
    rod[z] = rho.copy()
    crod = zeros((N,N,3))
"""
def save_fig(post):
    print "Saving", post
    savefig("xray_plot_%04d.png" % post)        

def update_plot():
    global rod, rho, crod, rx, ry, rz, ps, point, fpoint
    gs = 100
    cm = pylab.cm.hot_r
    
    mrodx = ones_like(rod[x])
    mrody = ones_like(rod[y])
    mrodz = ones_like(rod[z])
    
    if rx is not None:
        mrodx = (rod[x] > rx[0]) * (rod[x] < rx[1])
    if ry is not None:
        mrody = (rod[y] > ry[0]) * (rod[y] < ry[1])
    if rz is not None:
        mrodz = (rod[z] > rz[0]) * (rod[z] < rz[1])
    
    ex = (-ps, ps)
    ey = (-ps, ps)
    ez = (-ps, ps)
    if rx is not None:
        ex = tuple(rx)
    if ry is not None:
        ey = tuple(ry)
    if rz is not None:
        ez = tuple(rz)

    fxy.cla(); fxz.cla(); fyz.cla()
    exy = array([array(ex),array(ey)])
    exz = array([array(ex),array(ez)])
    eyz = array([array(ey),array(ez)])
    hxy = histogram2d(rod[x].flatten(), rod[y].flatten(), bins=gs, range=exy)
    hxz = histogram2d(rod[x].flatten(), rod[z].flatten(), bins=gs, range=exz)
    hyz = histogram2d(rod[y].flatten(), rod[z].flatten(), bins=gs, range=eyz)
    fxy.imshow(rot90(hxy[0]), cmap=cm, extent=ex+ey); 
    fxz.imshow(rot90(hxz[0]), cmap=cm)#, extent=ex+ez); 
    fyz.imshow(rot90(hyz[0]), cmap=cm)#, extent=ey+ez); 
    fpoint = None

    mask1 = ones_like(rod[x]) * mrodx * mrody * mrodz
    mask = (nd.laplace(mask1) > 1e-6) == False
    
    tcrod = ones_like(crod)
    trho = rho * mask
    tcrod[:,:,0] = crod[:,:,0] * mask
    tcrod[:,:,1] = crod[:,:,1] * mask
    tcrod[:,:,2] = crod[:,:,2] * mask

    frod.imshow(rot90(rod[x]), vmin=rod[x].min(), vmax=rod[x].max(), extent=(0,N,0,N)); 
    frho.imshow(rot90(trho), vmin=trho.min(), vmax=trho.max(), extent=(0,N,0,N)); 
    fmask.imshow(rot90(mask1), vmin=mask1.min(), vmax=mask1.max(), cmap=pylab.cm.gray, extent=(0,N,0,N))

    title("Incoherent X-Ray Patterns")
    fxy.set_title("Scattering intensity x-y", fontsize=20)
    fxz.set_title("Scattering intensity x-z", fontsize=20)
    fyz.set_title("Scattering intensity y-z", fontsize=20)
    frho.set_title("Dislocation Density", fontsize=20)
    frod.set_title("Misorientation", fontsize=20)
    fmask.set_title("Selection mask", fontsize=20)
 
    fxy.set_xticks([]); fxy.set_yticks([])
    fxz.set_xticks([]); fxz.set_yticks([])
    fyz.set_xticks([]); fyz.set_yticks([])
    frod.set_xticks([]); frod.set_yticks([])    
    frho.set_xticks([]); frho.set_yticks([])
    fmask.set_xticks([]); fmask.set_yticks([]);

    subplots_adjust(0.,0.,1.,0.95,0.01,0.05)
    show()
    print "done"


def update_lower_plot():
    global point, fpoint
    if point is not None:
        p = (rod[x][point], rod[y][point], rod[z][point])
        if fpoint is None:
            fpoint = fxy.plot(p[0], p[1], 'ro').pop()
        fpoint.set_xdata(array([p[0]]))
        fpoint.set_ydata(array([p[1]]))
   
    draw()


get_components()
update_plot()
