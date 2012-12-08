import numpy as np
import pylab as pl
import time
import scipy.weave as weave
import scipy.optimize as opt
import powerlaw

def BoxCounting(field, N, cutoff, dim=2, box=2, majority=1):
    areas = []
    boxes = []
    snaps = []
 
    code = """
        int ns = n/box;

        int iloops = (dim>0)?ns:1;
        int jloops = (dim>1)?ns:1;
        int kloops = (dim>2)?ns:1;

        int iiloops = (dim>0)?box:1;
        int jjloops = (dim>1)?box:1;
        int kkloops = (dim>2)?box:1;

        for (int i=0; i<iloops; i++){
        for (int j=0; j<jloops; j++){
        for (int k=0; k<kloops; k++){
            int index = i + ns*j + ns*ns*k;
            b_out[index] = 0;

            for (int ii=0; ii<iiloops; ii++){
            for (int jj=0; jj<jjloops; jj++){
            for (int kk=0; kk<kkloops; kk++){
                b_out[index] += b_in[i*box+ii + 
                                    (j*box+jj)*ns*box + 
                                    (k*box*box+kk)*ns*box*ns*box ];
            } } }

            if (b_out[index] >= majority)
                b_out[index] = 1;
            else
                b_out[index] = 0;
        } } }
    """

    n = N
    b_in = (field>cutoff).copy().flatten()

    areas.append(b_in.sum()) 
    boxes.append(1)
    snaps.append(b_in.reshape(n, n).copy())
     
    while n > 1:
        newn = n/box
        b_out = np.zeros(newn * newn)

        err = weave.inline(code, ['n', 'box', 'dim', 'majority', 'b_in', 'b_out'], extra_compile_args=["-O3"]); 

        areas.append(b_out.sum()) 
        boxes.append(N/newn)
        snaps.append(b_out.reshape((newn,newn)).copy())

        b_in = b_out
        n = newn

    return np.array(boxes), np.array(areas), np.array(snaps)

def twopowers_values(p,bs,bc):
    alpha, C, x_bend = p
    x_bend = 2**x_bend
    return (bs<x_bend)*C*(bs**alpha) + (bs>=x_bend)*C*(x_bend**alpha)*((bs/x_bend)**(-2))

def twopowers(p,bs,bc):
    result = bc - twopowers_values(p,bs,bc)
    return result

def ExponentFromBoxCounting(boxsize, boxcount, dim=2, cutoff=4):
    #p = np.polyfit(np.log(boxsize[boxcount.nonzero()][:cutoff]), np.log(boxcount[boxcount.nonzero()][:cutoff]), 1)
    p = opt.leastsq(twopowers, [-2,(boxcount[0])/np.power(boxsize[0],-2),3], args=(np.array(boxsize),np.array(boxcount))) 
    #print p[0]
    return p[0]


def Run(map, N, dim):
    start = time.time()
    json  = TarFile.LoadTarJSON(filename)
    state = TarFile.LoadTarState(filename, time=time)
    map = state.CalculateRhoFourier().modulus()

    exp = []
    cut = []
    for i in np.arange(1.0,30,5.0):
        b, a, s = BoxCounting(map, N=N, dim=dim, cutoff=i)
        poly = ExponentFromBoxCounting(b, a, cutoff=2)
    
        import pylab
        pylab.loglog(b,a)
        pylab.show()
        """
        data = []
        for (x,n) in zip(b,a):
            data = data + ([x]*n)
        data = np.array(data)
    
        fit = powerlaw.Fit(data, discrete=True, xmin=b[0], xmax=b[-1])
        print fit.power_law.alpha
        #print fit.truncated_power_law.alpha
        """
    
        dimension = poly[0]
        cut.append(i)
        exp.append(dimension)
        print i, dimension
    
        import pylab
        pylab.loglog(b,twopowers_values(poly,b,a),'bo-',label='fit')
        pylab.loglog(b,a,'rx',label='data')
        pylab.show()

    end = time.time()
    print "total time: ", end - start
    print "dimension = ", dimension
    
    x = np.arange(b[0], b[-1], 1.0)
    y = poly[1] * (x ** poly[0])
    
    pl.plot(cut, exp, 'o-')
    pl.show()
    


