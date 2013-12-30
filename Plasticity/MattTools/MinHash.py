import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib

def SegmentationSLIC(filename='/media/scratch/plasticity/lvp2d1024_s0_d.tar', time=None):
    t,s = TarFile.LoadTarState(filename, time=time)
    rod = s.CalculateRotationRodrigues()
    img = img_as_float(RodriguesToUnambiguousColor(rod['x'], rod['y'], rod['z'],maxRange=None,centerR=None).astype('int8'))
    
    segments_slic = slic(img, ratio=0.001, n_segments=2500, sigma=2)
    print("Slic number of segments: %d" % len(np.unique(segments_slic)))
    
    from functools import partial
    mark_boundaries = partial(mark_boundaries, color=[0,0,0])
    
    fig = plt.figure()
    plt.imshow(mark_boundaries(img, segments_slic))
    plt.show()

xs = np.fromfile('rodx.bin').reshape((128,128))
ys = np.fromfile('rody.bin').reshape((128,128))
zs = np.fromfile('rodz.bin').reshape((128,128))

cols = np.zeros((128,128,3))
full = np.zeros((128,128,3))
full[:,:,0] = xs
full[:,:,1] = ys
full[:,:,2] = zs

cmap = matplotlib.colors.ListedColormap ( np.random.rand(257,3))

q = np.packbits((full.dot(np.random.randn(3,4)) > 0).astype('int'),-1).squeeze()
plt.imshow(q,cmap=cmap,interpolation='none')

foo = np.zeros((128,128))
for i in xrange(100): 
    q = np.packbits((full.dot(np.random.randn(3,4)) > 0).astype('int'),-1).squeeze()
    g = (np.array(np.gradient(q))**2).sum(0) > 0
    foo += g;

cols[:,:,0] = xs
cols[:,:,1] = ys
cols[:,:,2] = zs
cols = cols - cols.min(0).min(0)
cols = cols / cols.max(0).max(0)

