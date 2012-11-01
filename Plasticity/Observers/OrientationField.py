from numpy import sqrt, sin, cos, power, zeros, array, fromfunction, pi, abs
from Plasticity import NumericalMethods

ME = NumericalMethods.ME 

def curve_nicely(x, d=8.0):
    return x

def Ordered3Arrays(a,b,c):
    ab = (a>b).astype(float)
    ac = (a>c).astype(float)
    bc = (b>c).astype(float)
    maxabc = (ab*ac*a + (1-ab)*bc*b + (1-ac)*(1-bc)*c)
    minabc = ((1-ab)*(1-ac)*a + ab*(1-bc)*b + ac*bc*c)
    return maxabc, a+b+c-maxabc-minabc, minabc 
            
def RodriguesFieldTo100PoleFigure(x,y,z):
    """
    This function uses rodrigues vector field x, y, z component to
    find out where (100) pole would be at.

    Returns rotated vector of (100): vx, vy, vz
    """
    """
    make angles and unit vectors
    """
    theta = sqrt(x**2+y**2+z**2)+ME
    ux = x/theta
    uy = y/theta
    uz = z/theta
    """
    find rotated (100)
    """
    cost = cos(theta)
    sint = sin(theta)
    vx = ux*ux*(1-cost) + cost
    vy = uy*ux*(1-cost) + uz*sint
    vz = uz*ux*(1-cost) - uy*sint
    """
    Now make them all positive and order properly
    """
    vx = abs(vx)
    vy = abs(vy)
    vz = abs(vz)
    vx, vy, vz = Ordered3Arrays(vx, vy, vz)
    return vx, vy, vz


def CubicPoleFigureRGB(x,y,z):
    """
    This function transforms (100) pole vectors into a set of rgb
    vectors to use for plotting EBSD-like colorcoded rotation field.
    
    This function also assumes (100) poles, with cubic symmetry.
    """
    x, y, z = Ordered3Arrays(abs(x), abs(y), abs(z))
    norm = sqrt(x*x+y*y+z*z)
    distr = x/norm
    distr -= distr.min()
    distr /= distr.max()
    distg = (x+y)/norm/sqrt(2.)
    distg -= distg.min()
    distg /= distg.max()
    distb = (x+y+z)/norm/sqrt(3.)
    distb -= distb.min()
    distb /= distb.max()
    
    r = curve_nicely(x-y)
    g = curve_nicely(y-z)
    b = curve_nicely(sqrt(y*z))
    
    sumpower = 4.
    sum = power(r**sumpower+g**sumpower+b**sumpower, 1/sumpower)
    r /= sum
    g /= sum
    b /= sum
    
    rgb = zeros(list(x.shape)+[3])
    if len(x.shape) == 1:
        rgb[:,0] = r
        rgb[:,1] = g
        rgb[:,2] = b
    elif len(x.shape) == 2:
        rgb[:,:,0] = r
        rgb[:,:,1] = g
        rgb[:,:,2] = b
    return rgb

def RodriguesTo100RGB(rodrigues, scale=None):
    """
    This function is an interface for CubicPoleFigureRGB and
    RodriguesFieldTo100PoleFigure.

    scale is used to define a full rotation.
    by default, 2pi is assumed to be a full rotation.
    """
    rodx = rodrigues['x'] + 0.
    rody = rodrigues['y'] + 0.
    rodz = rodrigues['z'] + 0.
    if scale is not None:
        rodx *= 2.*pi / scale
        rody *= 2.*pi / scale
        rodz *= 2.*pi / scale

    vx, vy, vz = RodriguesFieldTo100PoleFigure(rodx, rody, rodz)
    rgb = CubicPoleFigureRGB(vx, vy, vz)
    return rgb


def RodriguesToUnambiguousColor(rodx,rody,rodz,maxRange=None,centerR=None):
    """
    All rodrigues vectors concentrate around the center vector 'centerR', and are confined
    in the 3D box associated with 'maxRange'. Every rodrigues vector is indicated by an unique
    RGB color index. (128,128,128) is assumed to be the center. 
    """
    rodmin = array([rodx.min(), rody.min(), rodz.min()])
    rodmax = array([rodx.max(), rody.max(), rodz.max()])
    if maxRange is None:
        maxRange = (rodmax-rodmin).max()
    if centerR is None:
        centerR = (rodmin+rodmax)/2.
    colormap = zeros(list(rodx.shape)+[3],float)
    colormap[...,0] = 255.*(0.5+(rodx-centerR[0])/maxRange)
    colormap[...,1] = 255.*(0.5+(rody-centerR[1])/maxRange)
    colormap[...,2] = 255.*(0.5+(rodz-centerR[2])/maxRange)
    return colormap.astype(int)

      
if __name__=="__main__":
    N = 100
    y = fromfunction(lambda x,y: (x-N/2)/N*2., [N,N])
    z = fromfunction(lambda x,y: (y-N/2)/N*2., [N,N])
    x = zeros([N,N]) + 1.

    rgb = CubicPoleFigureRGB(x,y,z)
    import pylab
    pylab.figure()
    pylab.imshow(rgb)
    pylab.colorbar()
    pylab.show()
    
