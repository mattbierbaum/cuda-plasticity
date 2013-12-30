import matplotlib.pyplot as plt
import numpy as np

from skimage.data import lena
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

from Plasticity.FieldInitializers import FieldInitializer
from Plasticity import TarFile
from Plasticity.Observers.OrientationField import RodriguesToUnambiguousColor

import scipy.ndimage as nd
from skimage.filter import canny

def SegmentationSLIC_run_2d(rod):
    img = img_as_float(RodriguesToUnambiguousColor(rod['x'], rod['y'], rod['z'],maxRange=None,centerR=None).astype('uint8'))
    
    segments_slic = slic(img, ratio=0.001, n_segments=80, sigma=0)
    print("Slic number of segments: %d" % len(np.unique(segments_slic)))
   
    return segments_slic

def SegmentationSLIC_run_3d(rod):
    img = img_as_float(RodriguesToUnambiguousColor(rod['x'], rod['y'], rod['z'],maxRange=None,centerR=None).astype('uint8'))
    
    segments_slic = slic(img, ratio=0.001, n_segments=2500, sigma=0)
    print("Slic number of segments: %d" % len(np.unique(segments_slic)))
   
    return segments_slic

def SegmentationSLIC(filename='/media/scratch/plasticity/lvp2d1024_s0_d.tar', time=None):
    t,s = TarFile.LoadTarState(filename, time=time)
    rod = s.CalculateRotationRodrigues()
    img = img_as_float(RodriguesToUnambiguousColor(rod['x'], rod['y'], rod['z'],maxRange=None,centerR=None).astype('uint8'))
    segments_slic = felzenszwalb(img, scale=300, sigma=0.0, min_size=10)

    #segments_slic = slic(img, ratio=0.001, n_segments=2500, sigma=0)
    print("Slic number of segments: %d" % len(np.unique(segments_slic)))
    
    fig = plt.figure()
    plt.imshow(mark_boundaries(img, segments_slic, color=[0,0,0], outline_color=None))
    plt.show()

def SegmentationFelz_run_2d(rod):
    img = img_as_float(RodriguesToUnambiguousColor(rod['x'], rod['y'], rod['z'],maxRange=None,centerR=None).astype('uint8'))
    segments_slic = felzenszwalb(img, scale=100, sigma=0.0, min_size=10)
    print("Slic number of segments: %d" % len(np.unique(segments_slic)))
    return segments_slic

def SegmentationFelzenszwalb(filename='/media/scratch/plasticity/lvp2d1024_s0_d.tar', time=None):
    t,s = TarFile.LoadTarState(filename, time=time)
    rod = s.CalculateRotationRodrigues()
    img = img_as_float(RodriguesToUnambiguousColor(rod['x'], rod['y'], rod['z'],maxRange=None,centerR=None).astype('uint8'))
    
    segments_slic = felzenszwalb(img, scale=100, sigma=0.0, min_size=10)
    print("Slic number of segments: %d" % len(np.unique(segments_slic)))
    
    fig = plt.figure()
    plt.imshow(mark_boundaries(img, segments_slic, color=[0,0,0], outline_color=None))
    plt.show()


