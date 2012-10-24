from Plasticity.PlasticityStates import RhoState
from Plasticity.Fields import Fields

from numpy import fft,pi
import numpy
import scipy.weave as W

from Plasticity.Constants import *
from Plasticity import NumericalMethods

def RandomRodriguesVectorFieldGenerator(gridShape, numberOfCells, inplane=True, seed=0):
    """
    Generate random rodrigues vector field with nucleation & growth method.
    """ 
    field = Fields.VectorField(gridShape)
     
    dimension = len(gridShape)
    if dimension == 1:
        pass
    elif dimension == 2:
        code = """
        int x[numberOfCells], y[numberOfCells];
        /* Generate seeds */
        //srand(time(0));
        if (seed==0) {
            srand(time(0));
        } else {
            srand(seed);
        }
        for(int i=0; i<numberOfCells; i++) {
            while(1) {
                int j=0;
                x[i] = (int)((double)rand() / ((double)RAND_MAX + (double)1) * nx)%nx;
                y[i] = (int)((double)rand() / ((double)RAND_MAX + (double)1) * ny)%ny;
                for(j=0; j<i; j++) {
                    if ((x[i]==x[j])&&(y[i]==y[j])) {
                        break;
                    }
                }
                if (j==i)
                    break;
            }
        }

        /* Compare distance to each seed and mark with the number */
        double defaultmindist = ((double)nx+ny)*((double)nx+ny);
        for(int i=0; i<nx; i++) {
            for(int j=0; j<ny; j++) {
                double mindist;
                mindist = defaultmindist;
                for(int t=0; t<numberOfCells; t++) {
                    int xtmp = abs(i-x[t]);
                    int ytmp = abs(j-y[t]);
                    if (xtmp > nx/2) {
                        xtmp = nx-xtmp;
                    }
                    if (ytmp > ny/2) {
                        ytmp = ny-ytmp;
                    }
                    double dist = xtmp*xtmp+ytmp*ytmp;
                    if (dist < mindist) {
                        *(result+i*ny+j) = t+1;
                        mindist = dist;
                    }
                }
            }
        }  
        """ 
        nx = gridShape[0]
        ny = gridShape[1]
        result = numpy.zeros(gridShape)
        W.inline(code, ['nx','ny','result','numberOfCells','seed'], extra_compile_args=["-w"])
        if inplane is True:
            codeAssign = """
            for(int i=0; i<nx; i++) {
                for(int j=0; j<ny; j++) {
                    *(fieldz+i*ny+j) = *(values+(int)*(result+i*ny+j)-1);
                }
            }
            """
            fieldz = numpy.zeros(gridShape)        
            if seed != 0:
                numpy.random.seed(seed)
            values = numpy.random.random(numberOfCells)*2*numpy.pi
    
            W.inline(codeAssign, ['nx','ny','result','fieldz','values',], extra_compile_args=["-w"])
            field[z] = fieldz
        else:
            codeAssign = """
            for(int i=0; i<nx; i++) {
                for(int j=0; j<ny; j++) {
                    *(fieldx+i*ny+j) = *(valuesx+(int)*(result+i*ny+j)-1);
                    *(fieldy+i*ny+j) = *(valuesy+(int)*(result+i*ny+j)-1);
                    *(fieldz+i*ny+j) = *(valuesz+(int)*(result+i*ny+j)-1);
                }
            }
            """
            if seed != 0:
                numpy.random.seed(seed)
            fieldx = numpy.zeros(gridShape)        
            fieldy = numpy.zeros(gridShape)        
            fieldz = numpy.zeros(gridShape)        
            valuesx = numpy.random.random(numberOfCells)*2*numpy.pi
            valuesy = numpy.random.random(numberOfCells)*2*numpy.pi
            valuesz = numpy.random.random(numberOfCells)*2*numpy.pi
    
            W.inline(codeAssign, ['nx','ny','result','fieldx','fieldy','fieldz','valuesx','valuesy','valuesz'], extra_compile_args=["-w"])
            field[x] = fieldx
            field[y] = fieldy
            field[z] = fieldz
        return field
    elif dimension == 3:
        pass
    else:
        assert False
        pass

def ManyPartRodriguesInitializer(gridShape, angle=pi/36, width=10, parts=4):
    rodrigues = Fields.VectorField(gridShape)
    dimension = len(gridShape)
    zfield = numpy.zeros(gridShape)
    steps = gridShape[1]/parts
    anglesteps = angle/parts

    if dimension == 2:
        for i in range(parts/2):
            zfield[:,steps*i:] += anglesteps
        for i in range(parts/2, parts):
            zfield[:,steps*i:] -= anglesteps
        """
        Make it smooth of width 10
        """
        for i in range(width):
            zfield = 0.25*numpy.roll(zfield,-1,1)+0.5*zfield+0.25*numpy.roll(zfield,1,1)
    
        rodrigues[z] = zfield
    elif dimension == 1:
        zfield[gridShape[0]/2:] = angle2
        """
        Make it smooth of width 10
        """
        for i in range(width):
            zfield = 0.25*numpy.roll(zfield,-1,0)+0.5*zfield+0.25*numpy.roll(zfield,1,0)
    
        rodrigues[z] = zfield
    return rodrigues 

def TwoPartRodriguesInitializer(gridShape, angle=pi/36, width=10):
    rodrigues = Fields.VectorField(gridShape)
    dimension = len(gridShape)
    angle1 = -angle/2.
    angle2 = angle/2.
    zfield = numpy.zeros(gridShape)
    zfield += angle1

    import GridArray
    if dimension == 2:
        zfield[:,gridShape[1]/2:] = angle2
        """
        Make it smooth of width 10
        """
        for i in range(width):
            zfield = 0.25*numpy.roll(zfield,-1,1)+0.5*zfield+0.25*numpy.roll(zfield,1,1)
    
        rodrigues[z] = GridArray.GridArray(zfield)
    elif dimension == 1:
        zfield[gridShape[0]/2:] = angle2
        """
        Make it smooth of width 10
        """
        for i in range(width):
            zfield = 0.25*numpy.roll(zfield,-1,0)+0.5*zfield+0.25*numpy.roll(zfield,1,0)
    
        rodrigues[z] = zfield
    return rodrigues 
 

def Smearing2DFields(gridShape,field,diffusion=10.):
    diffusion *= field.fabs().max()/gridShape[0]
    Kfield = field.FFT()
    X ,Y = gridShape
    kx   = numpy.fromfunction(lambda x,y: numpy.sin(pi*x/X),[X,Y/2+1])
    ky   = numpy.fromfunction(lambda x,y: numpy.sin(pi*y/Y),[X,Y/2+1])        
    kSq  = kx*kx + ky*ky
    for comp in Kfield.components:
        Kfield *= numpy.exp(-diffusion*kSq)
    return Kfield.IFFT()  

    
def InitializeRhoFromRodriguesField(gridShape, rodrigues):
    #rodrigues = Smearing2DFields(gridShape,rodrigues)
    rho = Fields.TensorField(gridShape)
    state = RhoState.RhoState(gridShape, field=rho)
    """
    """
    K = state.sinktools

    Krho = Fields.TensorField(gridShape, kspace=True)
    Komega = rodrigues.FFT()

    sum = 0.
    for m in Komega.components:
        sum += K.k[m]*Komega[m]
    sum *= -1.0j

    for component in Krho.components:
        (i,j) = component
        Krho[component] = 1.j*K.k[j]*Komega[i]
        if (i==j):
            Krho[component] += sum

    """
    Alternatively
    """
    """
    dimension = len(gridShape)
    if dimension == 1:
        dim = [z]
    elif dimension == 2:
        dim = [x,y]
    elif dimension == 3:
        dim = [x,y,z]

    sum = 0.
    for m in rodrigues.components:
        if m in dim:
            sum += NumericalMethods.SymmetricDerivative(rodrigues[m], gridShape, dim.index(m))
    sum *= -1.0

    for component in rho.components:
        (i,j) = component
        if j in dim:
            rho[component] = NumericalMethods.SymmetricDerivative(rodrigues[i], gridShape, dim.index(j))
        if i==j:
            rho[component] += sum        
    """
    """
    Krho_combined = Fields.TensorField(KgridShape)
    Krho_combined[x,x]=K.kz*Krho[x,y]-K.ky*Krho[x,z] 
    Krho_combined[x,y]=K.kz*Krho[y,y]-K.ky*Krho[y,z] 
    Krho_combined[x,z]=K.kz*Krho[z,y]-K.ky*Krho[z,z] 
    Krho_combined[y,x]=K.kx*Krho[x,z]-K.kz*Krho[x,x] 
    Krho_combined[y,y]=K.kx*Krho[y,z]-K.kz*Krho[y,x] 
    Krho_combined[y,z]=K.kx*Krho[z,z]-K.kz*Krho[z,x] 
    Krho_combined[z,x]=K.ky*Krho[x,x]-K.kx*Krho[x,y] 
    Krho_combined[z,y]=K.ky*Krho[y,x]-K.kx*Krho[y,y] 
    Krho_combined[z,z]=K.ky*Krho[z,x]-K.kx*Krho[z,y] 
    for component in Krho_combined.components:
        print component, Krho_combined[component].max(), Krho_combined[component].min()

    print K.k[x].max(), K.k[x].min()
    print Komega[z].max(), Komega[z].min()
    temp = K.ky*(K.k[x]*Komega[z])-K.kx*(K.k[y]*Komega[z])
    print temp.max(), temp.min()
    arg = temp.argmax()
    x1, x2 = (int(arg/KgridShape[1]), arg%KgridShape[1])
    print temp.shape
    print K.ky[x1,x2], K.kx[x1,x2], Komega[z][x1,x2]
    print K.ky[x1,x2]*(K.kx[x1,x2]*Komega[z][x1,x2])-K.kx[x1,x2]*(K.ky[x1,x2]*Komega[z][x1,x2])
    """

    """
    """
    rho = Krho.IFFT()
    state.UpdateOrderParameterField(rho)
    """
    Debug code from here
    """
    """
    Krho2 = rho.FFT()
    print (Krho-Krho2)[z,x]
    print (Krho-Krho2)[z,y]
    print (Krho-Krho2).max()
    import numpy.fft as fft
    for elem in rho.components:
        ft = fft.irfftn(Krho[elem])
        ift = fft.rfftn(ft)
        print numpy.abs(ift-Krho[elem]).max()
        print ift-Krho[elem]
    """
    return state

