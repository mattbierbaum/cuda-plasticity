import pylab
from pylab import *
from scipy import *
from Constants import *
from matplotlib.widgets import RectangleSelector
import scipy.ndimage as nd

import PlasticitySystem
import FieldInitializer
import OrientationField

def InteractivePlotter(object):
    def __init__(state):
        self.ps = 0.05
        self.rod = None
        self.rho = None
        self.crod = None
        self.rx = None
        self.ry = None
        self.rz = None

        self.fig = figure()
        self.fxy = subplot(231)
        self.fxz = subplot(232)
        self.fyz = subplot(233)
        self.frho = subplot(234)
        self.frod = subplot(235)

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
            global rx, ry, rz
            if event.key in ['Q', 'q']:
                rx = None
                ry = None
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
        
        toggle_selector_xy.RS = RectangleSelector(fxy,line_select_callback_xy,drawtype='box',useblit=True, button=[1,3])
        toggle_selector_xz.RS = RectangleSelector(fxz,line_select_callback_xz,drawtype='box',useblit=True, button=[1,3])
        toggle_selector_yz.RS = RectangleSelector(fyz,line_select_callback_yz,drawtype='box',useblit=True, button=[1,3])
        
        connect('key_press_event', toggle_selector_xy)
        connect('key_press_event', toggle_selector_xz)
        connect('key_press_event', toggle_selector_yz)
        
        show()

        self.rod = state.CalculateRotationRodrigues()
        self.rho = state.CalculateRhoFourier().modulus()
        if len(state.gridShape) == 3:
            self.rho = self.rho[:,:,0]
            self.rod[x] = self.rod[x][:,:,0]
            self.rod[y] = self.rod[y][:,:,0]
            self.rod[z] = self.rod[z][:,:,0]
        self.crod = OrientationField.RodriguesToUnambiguousColor(self.rod[x], self.rod[y], self.rod[z])
        self.ps = max(abs(array([self.rod[x].min(), self.rod[x].max(), self.rod[y].min(), self.rod[y].max(), self.rod[z].min(), self.rod[z].max()])))
        

def update_plot():
    global rod, rho, crod, rx, ry, rz, ps
    gs = 200
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
    
    """
    trodx_xy = (mrodx * mrody * rod[x]).flatten()
    trody_xy = (mrodx * mrody * rod[y]).flatten()
    trodx_xz = (mrodx * mrodz * rod[x]).flatten()
    trodz_xz = (mrodx * mrodz * rod[z]).flatten()
    trody_yz = (mrody * mrodz * rod[y]).flatten()
    trodz_yz = (mrody * mrodz * rod[z]).flatten()

    fxy.hexbin(trodx_xy[trodx_xy.nonzero()], trody_xy[trodx_xy.nonzero()], gridsize=gs, extent=(-ps,ps,-ps,ps), cmap=cm)
    fxz.hexbin(trodx_xz[trodx_xz.nonzero()], trodz_xz[trodx_xz.nonzero()], gridsize=gs, extent=(-ps,ps,-ps,ps), cmap=cm)
    fyz.hexbin(trody_yz[trody_yz.nonzero()], trodz_yz[trody_yz.nonzero()], gridsize=gs, extent=(-ps,ps,-ps,ps), cmap=cm)
    """
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
    fxy.hexbin(rod[x].flatten(), rod[y].flatten(), gridsize=gs, extent=ex+ey, cmap=cm)
    fxz.hexbin(rod[x].flatten(), rod[z].flatten(), gridsize=gs, extent=ex+ez, cmap=cm)
    fyz.hexbin(rod[y].flatten(), rod[z].flatten(), gridsize=gs, extent=ey+ez, cmap=cm)

    mask = ones_like(rod[x]) * mrodx * mrody * mrodz
    mask = (nd.laplace(mask) > 1e-6) == False
    
    tcrod = ones_like(crod)
    trho = rho * mask
    tcrod[:,:,0] = crod[:,:,0] * mask
    tcrod[:,:,1] = crod[:,:,1] * mask
    tcrod[:,:,2] = crod[:,:,2] * mask
    frod.imshow(tcrod)
    frho.imshow(trho, vmin=trho.min(), vmax=trho.max())
    show()
