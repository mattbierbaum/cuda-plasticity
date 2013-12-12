from scipy import *
from pylab import *

use_image = False
N, R = 2000, 200

if use_image:
    image = imread("in.png")
else:
    image = zeros((N,N,3))

W,L,C = image.shape
Z = fromfunction(lambda x,y: (1.0*x/W-0.5) + 1.0j*(1.0*y/L-0.5), (W,L))

###=============================================
def getpixels_image(q):
    x = (W*real(q)).astype('int') % W
    y = (L*imag(q)).astype('int') % L
    return image[x,y]

def getpixels_grid(q):
    x = (W*real(q)).astype('int') % W
    y = (L*imag(q)).astype('int') % L
    row = (x%(N/R) == 0).astype('int')
    col = (y%(N/R) == 0).astype('int')
    return (row + col - row*col + 1) %2

if use_image:
    getpixels = getpixels_image
else:
    getpixels = getpixels_grid

###==============================================
def mobius_transform():
    a = 1.0-0.5j
    b = 0.5+0.5j
    c = 1.0-0.2j
    d = b*c / a
    q = (-.1j*Z+0.2j) / ((0.9-1.j)*Z -0.01j)
    return getpixels(100*q/abs(q).max())

def poles():
    q = 1.5 / ((Z - 0.2111j)*(Z + 0.2111j))
    return getpixels(100*q/abs(q).max())

def log_map():
    q = exp(1/(Z+1e-6)) #W/5*sqrt(Z) 
    return getpixels(q)

###================================================
# go ahead and show all of our hard work
###================================================
gray()

out1 = mobius_transform()
figure(); imshow(out1, interpolation='nearest')

out2 = poles()
figure(); imshow(out2, interpolation='nearest')

out3 = log_map()
figure(); imshow(out3, interpolation='nearest')

show()














"""
shape = (N,N)
image = arange(0,N*N).reshape((shape))
col = (image  %(N/R)==0).astype('int')
row = (image/N%(N/R)==0).astype('int')
image = (col + row - row*col + 1) % 2
"""
