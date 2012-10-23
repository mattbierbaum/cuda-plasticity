from numpy import fromfunction, arctan, sin, pi, sqrt ,cos, fabs, random, arcsin, array

from Constants import *

def ListPlanes(d,D,L,k):
    """
    wave length
    """
    lamb = 2*pi/k
    maxtheta = arctan(D/sqrt(2)/L)
    nmax = int(2*d*sin(maxtheta)/lamb)
    assert nmax>0

    N = nmax*2+1
    nsize = [N, N, N]
    h = fromfunction(lambda a,b,c: (a-nmax), nsize).astype(int)
    k = fromfunction(lambda a,b,c: (b-nmax), nsize).astype(int)
    l = fromfunction(lambda a,b,c: (c-nmax), nsize).astype(int)

    """
    For cubic symmetry
    interplanar spacing calculated
    """
    ds = d/sqrt(h**2+k**2+l**2)
 
    hkl = {} 
    thetas = arcsin(lamb/2/ds)

    """
    Check for allowed hkls and put in a list
    """
    for i in range(N):
        for j in range(N):
            for m in range(N):
                if thetas[i,j,m] < maxtheta and (i!=0 or j!=0 or m!=0):
                    ijms = [h[i,j,m],k[i,j,m],l[i,j,m]]
                    """
                    Sort to get rid of duplicates?
                    ijms.sort()
                    ijms.reverse()
                    """
                    hkl[tuple(ijms)] = 1.
    return hkl.keys()

def RodriguesToSpots(rodrigues, strain=None, scale=None, threshold=0.02):
    """
    This function turns rodrigues vector field into X-ray diffraction
    spots, assuming several properties of the crystal.

    However, this does not take strain into account.
    This will be implemented in a separte, more complicated function.

    Update: This now can use strain in addition. Strain scales the same
    as the rodrigues vector, but this may need analytical check.

    Everything is based on small deformation approximation.
    """

    """
    lattice spacing, x-ray wavevector, distance screen, screen size
    is defined.
    FIXME - this needs to be properly adjusted
    """
    a = 1     
    k0 = 15
    D = 1
    L = 1

    """
    Assume that lab and sample have same directions, for now.
    Beam incident along z direction, and the screen is further apart
    in z direction
    """
    rodx = rodrigues[x]
    rody = rodrigues[y]
    rodz = rodrigues[z]
    theta = sqrt(rodx**2+rody**2+rodz**2)+2.2e-16
    ux = rodx/theta
    uy = rody/theta
    uz = rodz/theta
    if scale is not None:
        theta *= (2.*pi)/scale

    hkls = ListPlanes(a,D,L,k0)
    xs = []
    ys = []
    if strain is not None:
        if scale is not None:
            oldstrain = strain
            strain = {}
            for index in oldstrain:
                strain[index] = oldstrain[index] * (2.*pi)/scale

        b1 = 2*pi/a * array([1+2.*strain[z,y], -strain[x,y], -strain[x,z]])
        b2 = 2*pi/a * array([-strain[y,x], 1+2.*strain[z,x], -strain[y,z]])
        b3 = 2*pi/a * array([-strain[z,x], -strain[z,y], 1+2.*strain[x,y]])
        volchange = 1.+strain[x,x]+strain[y,y]+strain[z,z]
        b1 /= volchange
        b2 /= volchange
        b3 /= volchange
        
    for h,k,l in hkls:
        """
        reciprocal vector
        """
        if strain is not None: 
            G = b1*h + b2*k + b3*l
            Gx = G[0]
            Gy = G[1]
            Gz = G[2]
        else:
            Gx = h*2*pi/a
            Gy = k*2*pi/a
            Gz = l*2*pi/a
        """
        rotate G to lab direction

        based on rodrigues rotation formula from wikipedia
        """
        u_proj_G = (ux*Gx+uy*Gy+uz*Gz)
        Gx_lab = Gx*cos(theta)+(uy*Gz-uz*Gy)*sin(theta)+u_proj_G*ux*(1-cos(theta))
        Gy_lab = Gy*cos(theta)+(uz*Gx-ux*Gz)*sin(theta)+u_proj_G*uy*(1-cos(theta))
        Gz_lab = Gz*cos(theta)+(ux*Gy-uy*Gx)*sin(theta)+u_proj_G*uz*(1-cos(theta))

        """
        calculate k' and check condition

        k' = k0 + G_lab
        """
        kx_p = Gx_lab
        ky_p = Gy_lab
        kz_p = Gz_lab + k0
    
        k_p_diff = fabs(sqrt(kx_p*kx_p+ky_p*ky_p+kz_p*kz_p) - k0)/k0
        """
        plot a point
        """
        check_thres = (k_p_diff < threshold)
        xpos = kx_p*(float(D)/k0/L)
        ypos = ky_p*(float(D)/k0/L)

        xpos = xpos.flatten()
        ypos = ypos.flatten()
        indices = check_thres.flatten().nonzero()

        xs += list(xpos[indices])
        ys += list(ypos[indices])
    return xs, ys

def  XYPlotToImage(xs,ys,N=64,value=0.1):
    xs = array(xs)
    ys = array(ys)
    data = zeros((2*N,2*N),float)
    x = (N+xs*N).astype(int)
    y = (N+ys*N).astype(int)
    for i,j in zip (x,y):
        data[i,j] += value
    return data 

def RordriguesToImagePlot(rodrigues, strain=None, scale=None, threshold=0.0001, N=64, value=0.1):
    xs,ys = RodriguesToSpots(rodrigues,strain,scale,threshold)
    data = XYPlotToImage(xs,ys,N,value)
    return data 


if __name__=="__main__":
    rod1 = {}
    rod2 = {}
    N = 100
    gridShape = tuple([N,N])
    rod1[x] = random.random(gridShape)*pi
    rod1[y] = random.random(gridShape)*pi
    rod1[z] = random.random(gridShape)*pi

    xs, ys = RodriguesToSpots(rod1, threshold=0.001)
    import pylab
    pylab.figure()
    pylab.subplot(321)
    pylab.plot(xs, ys, 'b+')
    pylab.title("Random orientation")
    pylab.xticks([])
    pylab.yticks([])
    
    rod2[x] = random.random(gridShape)*0.02+1.0
    rod2[y] = random.random(gridShape)*0.02
    rod2[z] = random.random(gridShape)*0.02

    xs, ys = RodriguesToSpots(rod2, threshold=0.001)
    pylab.subplot(322)
    pylab.plot(xs, ys, 'b+')
    pylab.title("Nearly single orientation")
    pylab.xticks([])
    pylab.yticks([])

    def getrandomstrain(gridShape, maxstrain=0.01):
        strain = {}
        strain[x,x] = random.random(gridShape)*maxstrain
        strain[y,y] = random.random(gridShape)*maxstrain
        strain[z,z] = random.random(gridShape)*maxstrain
        strain[x,y] = strain[y,x] = random.random(gridShape)*maxstrain
        strain[y,z] = strain[z,y] = random.random(gridShape)*maxstrain
        strain[x,z] = strain[z,x] = random.random(gridShape)*maxstrain
        return strain

    strain = getrandomstrain(gridShape)
    xs, ys = RodriguesToSpots(rod1, strain=strain, threshold=0.001)
    pylab.subplot(323)
    pylab.plot(xs, ys, 'b+')
    pylab.title("Random with small strain")
    pylab.xticks([])
    pylab.yticks([])

    xs, ys = RodriguesToSpots(rod2, strain=strain, threshold=0.001)
    pylab.subplot(324)
    pylab.plot(xs, ys, 'b+')
    pylab.title("Single with small strain")
    pylab.xticks([])
    pylab.yticks([])

    strain = getrandomstrain(gridShape, maxstrain=0.1)
    xs, ys = RodriguesToSpots(rod1, strain=strain, threshold=0.001)
    pylab.subplot(325)
    pylab.plot(xs, ys, 'b+')
    pylab.title("Random with large strain")
    pylab.xticks([])
    pylab.yticks([])

    xs, ys = RodriguesToSpots(rod2, strain=strain, threshold=0.001)
    pylab.subplot(326)
    pylab.plot(xs, ys, 'b+')
    pylab.title("Single with large strain")
    pylab.xticks([])
    pylab.yticks([])

    pylab.show()
    pylab.figure()
    pylab.imshow(XYPlotToImage(xs,ys),vmin=0.,vmax=1.5)
    pylab.show()
