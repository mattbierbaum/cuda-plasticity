import numpy as np
import scipy as sp
import pylab as pl
import matplotlib as mpl
from Plasticity import TarFile
from collections import defaultdict
import PIL

FF = 1

srange111 = [-0.71,-0.17,-0.71,-0.17]

# create some standard tensors
delta = np.eye(3, dtype='int')
levi  = np.array([[[-(i-j)*(j-k)*(k-i)/2 for i in range(3)] for j in range(3)] for k in range(3)])

# make sure we made delta, levi-civita correctly
assert (np.einsum('imn,jmn->ij', levi, levi) == 2*delta).all()
assert (np.einsum('ijk,imn->jkmn', levi, levi) == \
        (np.einsum('jm,kn->jkmn', delta, delta) - np.einsum('jn,km->jkmn', delta, delta))).all()


mu,nu = 0.5,0.3
lamb = 2.*mu*nu/(1.-2.*nu)
def ExternalStrain(sigma,primaryStrain):
    x,y,z = 'x','y','z'
    strains = {x:primaryStrain[0],y:primaryStrain[1],z:primaryStrain[2]}
    strain_trace = strains[x]+strains[y]+strains[z]
    for i in [x,y,z]:
        sigma[i,i] += strains[i] #lamb*strain_trace + 2.*mu*strains[i]
    return sigma

def CalculateTrueStrain(sig, rate, t, initial=np.array([0,0,0])):
    return ExternalStrain(sig, initial + rate*t)

#==========================================================
# these functions apply to the lattice vectors 
# before casting them across hkl's to form real Bragg peaks
#==========================================================
def generateHKLs(largest=4):
    import itertools
    return ( np.array(x) for x in itertools.product(xrange(-largest,largest+1), repeat=3))

def generateLatticeVectors(lattice='fcc', a0=3.62, shape=(1,)):
    if lattice=='fcc':
        a = a0/2 * np.array([[0,1,1],[1,0,1],[1,1,0]]).T 
    else:
        a = a0* np.sqrt(2) * np.eye(3)

    return np.squeeze( np.ones(shape)[..., np.newaxis, np.newaxis] * a )

def strainLattice(a, eps):
    return np.einsum('jk,XYik->XYij', delta,a) + np.einsum('XYjk,XYik->XYij', eps, a)

def rotateLatticeVectorByRod(v, rod):
    theta = np.sqrt(np.einsum('XYi,XYi->XY', rod, rod))
    axis  = np.einsum('XYi,XY->XYi',rod, 1/theta)
    return np.einsum('XYij,XY->XYij', v,np.cos(theta)) + \
           np.einsum('jkl,XYk,XYil,XY->XYij', levi, axis, v, np.sin(theta)) + \
           np.einsum('XYj,XYk,XYik,XY->XYij', axis, axis, v, (1-np.cos(theta)))

def latticeToReciprocal(a):
    acubed = 1/np.einsum('XYi,XYi->XY', a[...,0,:], np.einsum('ijk,XYj,XYk->XYi', levi, a[...,1,:], a[...,2,:]))
    return 2*np.pi * np.einsum('ilk,jmn,XYlm,XYkn,XY->XYij', levi*(1+levi)/2, levi, a, a, acubed)

#=========================================================
# here, we create hkl's and project to screen
#=========================================================
"""def optimizeAngle(g, k0):
    g_energy = np.zeros(g.shape)
    cos = -(np.einsum('XYi,XYi->XY', g, g) - 2*k0**2) / (2*k0**2)
    g_energy[...,0] = g[...,0]
    g_energy[...,1] = np.sqrt(k0**2*(1-cos**2) - g[...,0]**2)
    g_energy[...,2] = k0*cos
    return g_energy
"""
def optimizeAngle(g, k0, branch=1):
    gout = np.zeros(g.shape)
    gSq  = np.einsum('XYi,XYi->XY', g, g)
    gpSq = g[...,1]*g[...,1] + g[...,2]*g[...,2]
    det  = 4*(k0*k0/gSq)*gpSq/gSq - 1

    cos = 0.5*gSq/gpSq/k0 * (g[...,2] + np.sign(branch)*g[...,1]*np.sqrt(det))
    sin = np.sqrt(1 - cos**2)

    gout[...,0] = g[...,0]
    gout[...,1] = cos*g[...,1] + sin*g[...,2]
    gout[...,2] = cos*g[...,2] - sin*g[...,1] + k0
    return gout

def projectVectorToPlane(g, L, k0, **kwargs):
    q = optimizeAngle(g, k0, **kwargs)
    return ((q[...,2,np.newaxis]>0)*q[...,:]*L/q[...,2,np.newaxis])[...,:-1] 

def binScreenPixels(qx, qy, screen, l, mask=None, refs=None, srange=None):
    #Keeps track of the screen pixels that is of length with a certain number of bins
    srange = srange or [-l,l,-l,l]
    binx = (screen.shape[0]*(qx - srange[0])/(srange[1]-srange[0])).astype('int')
    biny = (screen.shape[1]*(qy - srange[2])/(srange[3]-srange[2])).astype('int')
    if refs is not None:
        for x in range(0,qx.shape[0]):
            for y in range(0,qy.shape[1]):
                if mask == None or mask[y,x] == 1:
                    refs[(binx[x,y], biny[x,y])].append((x,y))

    for x in xrange(0,qx.shape[0]):
        for y in xrange(0,qx.shape[0]):
            if mask == None or mask[y,x] == 1:
                if biny[x,y] > 0 and biny[x,y] < screen.shape[0] and binx[x,y]>0 and binx[x,y] < screen.shape[0]:
                    screen[biny[x,y],binx[x,y]] += 1

def plotXrays(rod, eps, L=4, lattice='sc', a0=3.62, k0=58, length=1.0, hkls=[np.array([1,1,1])], track=False, srange=None, mask=None):

    print "Calculating reciprocal lattice"
    a = generateLatticeVectors(lattice, a0, rod.shape[:-1])
    ap = strainLattice(a, eps)
    ap = rotateLatticeVectorByRod(ap, rod)
    gp = latticeToReciprocal(ap)

    screen = np.zeros((2048/FF,2048/FF), dtype='int')
    refs = defaultdict(list) if track else None

    for hkl in hkls:
        print "Starting ", hkl
        g = np.einsum('i,XYij->XYj', hkl, gp)
        q = projectVectorToPlane(g, L, k0, branch=-1)
        binScreenPixels(q[...,0], q[...,1], screen, length, mask, refs, srange)
        if not track:
            binScreenPixels(q[...,0], -q[...,1], screen, length, mask, refs, srange)
    return screen, refs


def sampleData(filename="/media/scratch/plasticity/lvp2d1024_s0_d.tar", t=None):
    t,s = TarFile.LoadTarState(filename, t)
    rod = s.CalculateRotationRodrigues()
    rho = s.CalculateRhoFourier().modulus()
    #eps = s.CalculateStrainField() #CalculateTrueStrain(s.CalculateStrainField(), rate=-0.05*np.array([-0.5,1.0,-0.5]), t=t)
    eps = CalculateTrueStrain(s.CalculateStrainField(), rate=0.05*np.array([-0.5,1.0,-0.5]), t=t)
    rod = np.array([rod['x'], rod['y'], rod['z']]).T
    eps = np.array([
        [eps['x','x'], eps['x','y'], eps['x','z']],
        [eps['y','x'], eps['y','y'], eps['y','z']],
        [eps['z','x'], eps['z','y'], eps['z','z']]
        ]).T
    return rod, eps, rho

def movie():
    skip = 2
    for t in np.arange(0,20, skip):
        rod, eps, rho = sampleData(filename="/media/scratch/plasticity/lvp2d1024_s2_d0.tar", t=t)
        s,q = plotXrays(rod, eps, hkls=[np.array([4,0,0])], srange=[-0.36,-0.08,0.24,0.46], track=True)
        pl.imshow(s, interpolation='nearest', origin='lower')
        pl.xticks([])
        pl.yticks([])
        pl.savefig("/media/scratch/plasticity/movie%02d.png" % (t/skip))

## quick notes
# array to PIL image: trho = (rho/rho.max()*256).astype('int8'); z = PIL.Image.fromarray(trho, mode='L'); z.save(...)
# PIL image to array; a = np.array(PIL.Image.open(...).getdata(), np.uint8).reshape(img.size[1], img.size[0]);

def plotWallCorrelation(rod, eps, rho, sr=None, rhomask=None, walls=None):
    rhop = rho[:]
    walls = walls or rhop > 3 
    notws = walls != 1

    sr = sr or [-0.36,-0.08,0.24,0.46]
    swalls,qwalls = plotXrays(rod, eps, hkls=[np.array([1,1,1])], srange=sr, track=True, mask=walls)
    snotws,qnotws = plotXrays(rod, eps, hkls=[np.array([1,1,1])], srange=sr, track=True, mask=notws)
    colors = [np.array([1,0,0,1]), np.array([0,0,1,1])]
   
    if rhomask is not None:
        tswalls, tqwalls = np.zeros(swalls.shape,dtype='int'), defaultdict(list)
        for (x,y) in qwalls.keys():
            for (i,j) in qwalls[(x,y)]:
                if rhomask[j,i] != 0:
                    tswalls[y,x] += 1
                    tqwalls[(x,y)].append((i,j))
        tsnotws, tqnotws = np.zeros(snotws.shape,dtype='int'), defaultdict(list)
        for (x,y) in qnotws.keys():
            for (i,j) in qnotws[(x,y)]:
                if rhomask[j,i] != 0:
                    tsnotws[y,x] += 1
                    tqnotws[(x,y)].append((i,j))
        swalls,qwalls = tswalls, tqwalls
        snotws,qnotws = tsnotws, tqnotws

    def torealspace_radius(x,y,sr,shape):
        px,py = float(x)/shape[0]*(sr[1]-sr[0])+sr[0], float(y)/shape[1]*(sr[3]-sr[2])+sr[2]
        ir = np.sqrt(px**2+py**2)
        return ir

    wallhist = []
    notwhist = []
    maskwalls = np.zeros(rho.shape)
    masknotws = np.zeros(rho.shape)
    for x in range(swalls.shape[0]):
        for y in range(swalls.shape[1]):
            if qwalls.has_key((x,y)):
                wallhist.extend([torealspace_radius(x,y,sr,swalls.shape)]*swalls[y,x])
                for tt in qwalls[(x,y)]:
                    maskwalls[tt[1],tt[0]] = 1
            if qnotws.has_key((x,y)):
                notwhist.extend([torealspace_radius(x,y,sr,swalls.shape)]*snotws[y,x])
                for tt in qnotws[(x,y)]:
                    masknotws[tt[1],tt[0]] = 1

    pl.figure()
    pl.hist(wallhist, bins=100, histtype='step', label="Cell walls")
    pl.hist(notwhist, bins=100, histtype='step', label="Cell interiors")
    pl.hist(wallhist+notwhist, bins=100, histtype='step', label="Combined")
    pl.xticks([])
    pl.yticks([])
    pl.ylabel("Counts", fontsize=24)
    pl.xlabel(r"$q_r$", fontsize=24)
    pl.legend(loc='upper right')
    #pl.semilogy()

    fig   = pl.figure(figsize=(0.7*5.12*2, 0.7*6.12*1))
    fxray = pl.subplot(121)
    freal = pl.subplot(122)
    trho = mpl.cm.gray_r(mpl.colors.Normalize()(rho**0.5))
    swalls = swalls.astype('float')
    snotws = snotws.astype('float')
    img = nd.gaussian_filter((swalls/swalls.sum() - snotws/snotws.sum())/(swalls/swalls.sum() + snotws/snotws.sum() + 1), sigma=1.3)

    fxray.imshow(img, vmax=img.max(), vmin=-img.max(), interpolation='nearest', origin='lower', cmap=mpl.cm.RdBu_r)
    freal.imshow(trho, interpolation='nearest', origin='lower')
    
    outline = (abs(nd.laplace(maskwalls)) > 0)[...,np.newaxis]*colors[0][np.newaxis,np.newaxis,...]
    filler  = np.ones(maskwalls.shape)[...,np.newaxis]*colors[0][np.newaxis,np.newaxis,...] 
    filler[:,:,3] = 0.7*maskwalls
    freal.imshow(outline, interpolation='nearest', origin='lower', alpha=1.0)
    freal.imshow(filler, interpolation='nearest', origin='lower', alpha=0.4)

    outline = (abs(nd.laplace(masknotws)) > 0)[...,np.newaxis]*colors[1][np.newaxis,np.newaxis,...]
    filler  = np.ones(masknotws.shape)[...,np.newaxis]*colors[1][np.newaxis,np.newaxis,...] 
    filler[:,:,3] = 0.7*masknotws
    freal.imshow(outline, interpolation='nearest', origin='lower', alpha=1.0)
    freal.imshow(filler, interpolation='nearest', origin='lower', alpha=0.4)

    freal.set_xticks([]); freal.set_yticks([])
    fxray.set_xticks([]); fxray.set_yticks([])
    fig.show()
    pl.subplots_adjust(0.,0.,1.,1.,0.01,0.05)
    pl.draw()
    #return wallhist+notwhist

#========================================================
# interactive peak picker
#========================================================
import scipy.ndimage as nd
from matplotlib.widgets import RectangleSelector

ip_rx, ip_ry = [0],[0]
ip_refs, ip_xray, ip_rho = 0,0,0

def ip_toggle_selector(event):
    global rx, ry
    if event.key in ['Q', 'q']:
        rx = None
        ry = None
        update_plot()
    if event.key in ['a']:
        global ip_rx, ip_ry
        ip_rx.append((1550/FF,2040/FF))
        ip_ry.append((1550/FF,2040/FF))
        update_plot()

def line_select_callback(eclick, erelease):
    x1, y1 = int(eclick.xdata),   int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    global ip_rx, ip_ry
    ip_rx[-1]=np.sort((x1,x2))
    ip_ry[-1]=np.sort((y1,y2))
    print ip_rx, ip_ry
    update_plot()

def update_plot():
    global ip_rx, ip_ry, ip_refs, ip_xray, ip_rho
    color_options = [np.array([1,0,0,1]), np.array([0.65,0,1,1]), np.array([0,0.8,1,1]), 
            np.array([1,0.6,0,1]),  np.array([0,1,0,1]), np.array([1,1,0,1]), np.array([1,0,1,1])]
    trho = ip_rho
    maskr = []
    maskx = []
    colors = []
    for rx, ry in zip(ip_rx, ip_ry):
        maskx.append(np.zeros(ip_xray.shape))
        maskr.append(np.ones(ip_xray.shape))
        c = color_options[(len(maskx)-1)%len(color_options)]
        colors.append(c)

        maskx[-1][ry[0]:ry[1],rx[0]] = 1
        maskx[-1][ry[0]:ry[1],rx[1]] = 1
        maskx[-1][ry[1],rx[0]:rx[1]] = 1
        maskx[-1][ry[0],rx[0]:rx[1]] = 1

        maskr[-1] = np.zeros(ip_rho.shape[:2])
        for x in range(rx[0], rx[1]):
            for y in range(ry[0], ry[1]):
                if ip_refs.has_key((x,y)):
                    for tt in ip_refs[(x,y)]:
                        maskr[-1][tt[1],tt[0]] = 1

    ip_fig = pl.gcf()
    ip_fxray = pl.subplot(121)
    ip_freal = pl.subplot(122)

    ip_fxray.cla()
    ip_freal.cla()
    trho = mpl.cm.gray_r(mpl.colors.Normalize()(ip_rho**0.5))
    txry = mpl.cm.jet(mpl.colors.Normalize()(ip_xray**0.5))

    ip_freal.imshow(trho, interpolation='nearest', origin='lower')
    ip_fxray.imshow(txry, interpolation='nearest', origin='lower')
    
    for mask, color in zip(maskr, colors):
        outline = (abs(nd.laplace(mask)) > 0)[...,np.newaxis]*color[np.newaxis,np.newaxis,...]
        filler  = np.ones(mask.shape)[...,np.newaxis]*color[np.newaxis,np.newaxis,...] 
        filler[:,:,3] = 0.7*mask
        ip_freal.imshow(outline, interpolation='nearest', origin='lower', alpha=1.0)
        ip_freal.imshow(filler, interpolation='nearest', origin='lower', alpha=0.4)

    for mask,color in zip(maskx, colors):
        outline = mask[...,np.newaxis]*color[np.newaxis,np.newaxis,...]
        ip_fxray.imshow(outline, interpolation='nearest', origin='lower', alpha=1.0)

    ip_freal.set_xticks([]); ip_freal.set_yticks([])
    ip_fxray.set_xticks([]); ip_fxray.set_yticks([])
    ip_fig.show()
    pl.subplots_adjust(0.,0.,1.,1.,0.01,0.05)
    pl.draw()

def createInteractivePlot(screen, refs, rho):
    global ip_refs, ip_xray, ip_rho, ip_rx, ip_ry
    ip_refs  = refs
    ip_xray  = screen
    ip_rho   = rho 

    ip_rx, ip_ry = [0],[0]
    ip_rx[-1] = (1550/FF,2040/FF)
    ip_ry[-1] = (1550/FF,2040/FF)

    ip_fig   = pl.figure(figsize=(0.7*5.12*2, 0.7*6.12*1))
    ip_fxray = pl.subplot(121)
    ip_freal = pl.subplot(122)
    ip_toggle_selector.RS = RectangleSelector(ip_fxray,line_select_callback,drawtype='box',useblit=True, button=[1,3])
    pl.connect('key_press_event', ip_toggle_selector)
    update_plot()

def aexam_data():
    rod, eps, rho = sampleData(filename="/media/scratch/plasticity/lvp2d1024_s0_d0.tar", t=0)
    s,q = plotXrays(rod, eps, hkls=[np.array([2,0,0])], srange=[[-0.17,-0.13],[0.125,0.165]], track=True)
    
    rod, eps, rho = sampleData(filename="/media/scratch/plasticity/lvp2d1024_s2_d0.tar", t=0)
    s,q = plotXrays(rod, eps, hkls=[np.array([4,0,0])], srange=[-0.36,-0.08,0.24,0.46], track=True)

    rod, eps, rho = sampleData(filename="/media/scratch/plasticity/lvp2d1024_s2_d0.tar", t=100)
    s,q = plotXrays(rod, eps, hkls=[np.array([4,0,0])], srange=[-0.044,-0.0051,0.074,0.112], track=True)

    rod, eps, rho = sampleData(filename="/media/scratch/plasticity/lvp2d1024_s2_d0.tar", t=100)
    a = np.array(PIL.Image.open("strainedrho1.bmp").getdata(), np.uint8).reshape(1024,1024);
    a = a == 255
    plotWallCorrelation(rod, eps, rho, [0.110,0.289,0.177,0.336], rhomask=a)

    rod, eps, rho = sampleData(filename="/media/scratch/plasticity/lvp2d1024_s2_d0.tar", t=200)
    a = np.array(PIL.Image.open("strainedrho_t200.bmp").getdata(), np.uint8).reshape(1024,1024);
    a = a == 255 
    plotWallCorrelation(rod, eps, rho, [0.0146,0.1045,0.0488,0.125], rhomask=a)

#if ip_peak_select:
#    selection = ip_xray[ip_rx[0]:ip_rx[1], ip_ry[0]:ip_ry[1]]
#    mean =  selection.mean()
#    maskf = ip_xray > mean/2
#   
#    pl.figure()
#    pl.imshow(maskf)
#    # find the peak and translate to real corrdinates
#    top = np.unravel_index(np.argmax(selection), selection.shape)
#    top = (top[0]+ip_rx[0], top[1]+ip_ry[1])
#    markers = np.zeros(ip_xray.shape).astype('int')
#    markers[top] = 1
#    # find the watershed of the peak
#    maskf = nd.watershed_ift((ip_xray*maskf>0).astype('uint8'), markers)
#def plotScreen(screen, fig=None, vmax=None):
#    fig = fig or pl.figure() 
#    vmax = vmax or screen[0].max()
#    pl.imshow(screen[0], vmin=0, vmax=vmax, interpolation='nearest', origin='lower', extent=screen[1])
#    #pl.xticks([])
#    #pl.yticks([])
#
#def screenCoordsToIndices(screen, q):
#    screens, srange = screen
#    x = int(screens.shape[0]*(q[0] - srange[0][0])/(srange[0][1]-srange[0][0]))
#    y = int(screens.shape[1]*(q[1] - srange[1][0])/(srange[1][1]-srange[1][0]))
#    return (x,y)
#txry_walls = ((swalls)**0.5)[...,np.newaxis]*colors[0][np.newaxis,np.newaxis,...]#*(swalls.max()/snotws.max())**0.5
#txry_notws = ((snotws)**0.5)[...,np.newaxis]*colors[0][np.newaxis,np.newaxis,...]
#txry_walls = ((swalls/swalls.max())**0.5)[...,np.newaxis]*colors[0][np.newaxis,np.newaxis,...]#*(swalls.max()/snotws.max())**0.5
#txry_notws = ((snotws/snotws.max())**0.5)[...,np.newaxis]*colors[0][np.newaxis,np.newaxis,...]
#img = (txry_wall - txry_notws)/(txry_walls + txry_notws + 1)

