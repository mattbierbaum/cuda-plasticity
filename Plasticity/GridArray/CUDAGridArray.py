import numpy
from numpy import fromfunction, pi
import numpy.fft as fft

import GridArray

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.cumath as cumath
import pycuda.curandom as curandom
from pyfft.cuda import Plan
from pycuda.elementwise import ElementwiseKernel
from pycuda.compiler import SourceModule

#from Constants import *

fft_plan_pool = {}
def get_pool(shape):
    shape = tuple(shape)
    if shape not in fft_plan_pool:
        fft_plan_pool[shape] = Plan(shape, dtype=numpy.float64)
    return fft_plan_pool[shape]

def print_mem_usage():
    (free, total) = cuda.mem_get_info()
    print "Free  : %f" % (free/1e9)
#    print "Total : %f" % total

GridArray.print_mem_usage = print_mem_usage

print_mem_usage()
#=========================================
# x direction
rollx1d = ElementwiseKernel(
    "double *x, double *z, int N, int s",
    "z[(i+N+s)%s] = x[i]",
    "rollerx1d")

rollx2d = ElementwiseKernel(
    "double *x, double *z, int N, int s",
    "int j = int(i/s);    \
     int ii = int(i-s*j); \
     z[(ii+N+s)%s + j*s] = x[i]",
    "rollerx2d")

rollx3d = ElementwiseKernel(
    "double *x, double *z, int N, int s",
    "int k = int(i/(s*s));      \
     int j = int((i-s*s*k)/s);  \
     int ii = int(i-s*s*k-s*j); \
     z[(ii+N+s)%s + j*s + k*s*s ] = x[i]",
    "rollerx3d")

#===========================================
# y direction
rolly2d = ElementwiseKernel(
    "double *x, double *z, int N, int s",
    "int j = int(i/s);    \
     int ii = int(i-s*j); \
     z[ii + ((j+N+s)%s)*s] = x[i]",
    "rollery2d")

rolly3d = ElementwiseKernel(
    "double *x, double *z, int N, int s",
    "int k = int(i/(s*s));      \
     int j = int((i-s*s*k)/s);  \
     int ii = int(i-s*s*k-s*j); \
     z[ii + ((j+N+s)%s)*s + k*s*s ] = x[i]",
    "rollery3d")

#============================================
# z direction
rollz3d = ElementwiseKernel(
    "double *x, double *z, int N, int s",
    "int k = int(i/(s*s));      \
     int j = int((i-s*s*k)/s);  \
     int ii = int(i-s*s*k-s*j); \
     z[ii + j*s + ((k+N+s)%s)*s*s ] = x[i]",
    "rollerz3d")


# the ordering of this array is very important
# in order to match the numpy.roll format.
# The extra row is just to show you the upper triangular
# form for now good reason
rollers = [[None, rollx1d, rolly2d, rollz3d], 
           [None, None,    rollx2d, rolly3d],
           [None, None,    None,    rollx3d],
           [None, None,    None,    None   ]]


class CUDAArray:
    def __init__(self, arr, arrC = None):
        self.arr = None
        self.arrC = None
        if arr.__class__ == gpuarray.GPUArray:
            self.arr = arr
        elif arr.__class__ == CUDAArray:
            self.arr = arr.arr
            self.arrC = arr.arrC
        else:
            if arr.dtype == 'complex64' or arr.dtype == 'complex128':
                self.arr = gpuarray.to_gpu(numpy.real(arr).astype('float64'))
                self.arrC = gpuarray.to_gpu(numpy.imag(arr).astype('float64'))
            else:
                self.arr = gpuarray.to_gpu(arr.astype('float64'))
        if arrC is not None:
            if arrC.__class__ == gpuarray.GPUArray:
                self.arrC = arrC
            else:
                self.arrC = gpuarray.to_gpu(arrC.astype('float64'))
        self.plan = get_pool(self.arr.shape)

    def free(self):
        if self.arr is not None:
            del self.arr #self.arr.gpudata.free()
        if self.arrC is not None:
            del self.arrC #self.arrC.gpudata.free()

    def numpy_array(self):
        """
        This function should return the array as a numpy array type.
        
        By default we can return self, since we inherit a numpy array.
        But it should be noted that other implementations should not
        simply return self.
        """
        if self.arrC is not None:
            return self.arr.get()+self.arrC.get()*1.0j
        return self.arr.get()
  
    def fftn(self):
        zeros = gpuarray.zeros_like(self.arr) 
        arr = gpuarray.zeros_like(self.arr) 
        arrC = gpuarray.zeros_like(self.arr) 
        self.plan.execute(self.arr, zeros, data_out_re=arr, data_out_im=arrC)
        return CUDAArray(arr, arrC)
     
    def ifftn(self, shape=None):
        arr = gpuarray.zeros_like(self.arr) 
        arrC = gpuarray.zeros_like(self.arrC) 
        self.plan.execute(self.arr, self.arrC, data_out_re=arr, 
                          data_out_im=arrC, inverse=True)
        return CUDAArray(arr, arrC)

    def rfftn(self):
        # it seems that we can just take half of the original fft
        # in both arr, arrC so that we match what was here originally
        zeros = gpuarray.zeros_like(self.arr) 
        arr = gpuarray.zeros_like(self.arr) 
        arrC = gpuarray.zeros_like(self.arr) 
        self.plan.execute(self.arr, zeros, data_out_re=arr, data_out_im=arrC)
        return CUDAArray(arr, arrC)

    def irfftn(self, shape=None):
        arr = gpuarray.zeros_like(self.arr) 
        arrC = gpuarray.zeros_like(self.arrC) 
        self.plan.execute(self.arr, self.arrC, data_out_re=arr, 
                          data_out_im=arrC, inverse=True)
        return CUDAArray(arr)

    def roll(self, shift, axis):
        #FIXME STILL
        """
        This better be implemented in a more efficient way
        """
        if self.arrC is not None:
            dim = len(self.arr.shape)
            roller = rollers[axis][dim]

            resultR = gpuarray.empty_like(self.arr)
            roller(self.arr, resultR, shift, self.arr.shape[0])

            resultC = gpuarray.empty_like(self.arrC)
            roller(self.arrC, resultC, shift, self.arrC.shape[0])
            return CUDAArray(resultR, resultC)

        else:
            dim = len(self.arr.shape)
            roller = rollers[axis][dim]

            result = gpuarray.empty_like(self.arr)
            roller(self.arr, result, shift, self.arr.shape[0])
            return CUDAArray(result)

    def __ilshift__(self, other):
        # FIXME make this a set value with operator
        """
        Using an lshift operator as the assignment operator can do several
        good, such as in place assigning for faster evaluation of certain
        things. But, it is syntactically different from the assignment operator
        and you have to first define the value to use lshift operator, thus
        making things more complicated.

        This also involves changing a lot of existing structures. Which is
        not all intuitive.
        """
        if not isinstance(other, CUDAArray):
            return NotImplemented
        else:
            # CUDAArray returns other, since it is the fastest way to handle
            return other

    """
    Functions used
    """
    def sqrt(self):
        return CUDAArray(cumath.sqrt(self.arr))

    def log(self):
        return CUDAArray(cumath.log(self.arr))

    def abs(self):
        return CUDAArray(cumath.fabs(self.arr))

    def fabs(self):
        return CUDAArray(cumath.fabs(self.arr))

    def exp(self):
        return CUDAArray(cumath.exp(self.arr))

    def power(self,x):
        return CUDAArray(self.arr.__pow__(x))

    """
    These do not need to be overloaded. However,
    numpy sum/max/min has richer options than we require.
    """
    def average(self):
        return self.sum()/self.arr.size
 
    def sum(self):
        return gpuarray.sum(self.arr).get().sum()

    """
    custom code inserted in pycuda to support these two
    """
    def max(self):
        return gpuarray.max(self.arr).get().max()

    def min(self):
        return gpuarray.min(self.arr).get().min()

    """
    static methods
    """
    def empty(shape, dtype='float64'):
        if dtype == 'complex128' or dtype == 'complex64' or dtype == 'complex':
            return CUDAArray(numpy.zeros(shape, dtype=dtype))
        return CUDAArray(gpuarray.empty(shape, dtype=dtype))
    empty = staticmethod(empty)
 
    def zeros(shape, dtype='float64'):
        if dtype == 'complex128' or dtype == 'complex64' or dtype == 'complex':
            return CUDAArray(numpy.zeros(shape, dtype=dtype))
        return CUDAArray(gpuarray.zeros(shape, dtype=dtype))
    zeros = staticmethod(zeros)

    def zeros_like(self):
        if self.arrC is not None:
            return CUDAArray(gpuarray.zeros_like(self.arr),gpuarray.zeros_like(self.arrC))
        return CUDAArray(gpuarray.zeros_like(self.arr))

    """
    default operations
    """
    def __add__(self, other):
        if other.__class__ == CUDAArray:
            if self.arrC is not None:
                if other.arrC is not None:
                    return CUDAArray(self.arr+other.arr, self.arrC+other.arrC)
                else:
                    return CUDAArray(self.arr+other.arr, self.arrC)
            else:
                if other.arrC is not None:
                    return CUDAArray(self.arr+other.arr, other.arrC)
                else:
                    return CUDAArray(self.arr+other.arr)
        if type(other) == complex:
            if self.arrC is not None:
                return CUDAArray(self.arr+other.real, self.arrC+other.imag)
            else:
                # FIXME
                return CUDAArray(self.arr+other.real, gpuarray.zeros_like(self.arr)+other.imag)
        else:
            if self.arrC is not None:
                return CUDAArray(self.arr+other, self.arrC)
            else:
                return CUDAArray(self.arr+other)
    __radd__ = __add__

    def __mul__(self, other):
        if other.__class__ == CUDAArray:
            if self.arrC is not None:
                if other.arrC is not None:
                    return CUDAArray(self.arr*other.arr-self.arrC*other.arrC, self.arr*other.arrC+self.arrC*other.arr)
                else:
                    return CUDAArray(self.arr*other.arr, self.arrC*other.arr)
            else:
                if other.arrC is not None:
                    return CUDAArray(self.arr*other.arr, self.arr*other.arrC)
                else:
                    return CUDAArray(self.arr*other.arr)
        if type(other) == complex:
            if self.arrC is not None:
                return CUDAArray(self.arr*other.real-self.arrC*other.imag, self.arr*other.imag+self.arrC*other.real)
            else:
                return CUDAArray(self.arr*other.real, self.arr*other.imag)
        else:
            if self.arrC is not None:
                return CUDAArray(self.arr*other, self.arrC*other)
            else:
                return CUDAArray(self.arr*other)
    __rmul__ = __mul__

    def __sub__(self, other):
        if other.__class__ == CUDAArray:
            if self.arrC is not None:
                if other.arrC is not None:
                    return CUDAArray(self.arr-other.arr, self.arrC-other.arrC)
                else:
                    return CUDAArray(self.arr-other.arr, self.arrC)
            else:
                if other.arrC is not None:
                    return CUDAArray(self.arr-other.arr, -other.arrC)
                else:
                    return CUDAArray(self.arr-other.arr)
        if type(other) == complex:
            if self.arrC is not None:
                return CUDAArray(self.arr-other.real, self.arrC-other.imag)
            else:
                # FIXME
                return CUDAArray(self.arr-other.real, gpuarray.zeros_like(self.arr)-other.imag)
        else:
            if self.arrC is not None:
                return CUDAArray(self.arr-other, self.arrC)
            else:
                return CUDAArray(self.arr-other)

    def __rsub__(self, other):
        if other.__class__ == CUDAArray:
            if self.arrC is not None:
                if other.arrC is not None:
                    return CUDAArray(other.arr-self.arr, other.arrC-self.arrC)
                else:
                    return CUDAArray(other.arr-self.arr, -self.arrC)
            else:
                if other.arrC is not None:
                    return CUDAArray(other.arr-self.arr, other.arrC)
                else:
                    return CUDAArray(other.arr-self.arr)
        if type(other) == complex:
            if self.arrC is not None:
                return CUDAArray(other.arr-self.real, other.arrC-self.imag)
            else:
                # FIXME
                return CUDAArray(other.arr-self.real, other.imag-gpuarray.zeros_like(self.arr))
        else:
            if self.arrC is not None:
                return CUDAArray(other-self.arr, -self.arrC)
            else:
                return CUDAArray(other-self.arr)

    def __div__(self, other):
        if other.__class__ == CUDAArray:
            if self.arrC is not None:
                if other.arrC is not None:
                    mag = other.arr*other.arr+other.arrC*other.arrC
                    return CUDAArray((self.arr*other.arr+self.arrC*other.arrC)/mag, (self.arrC*other.arr-self.arr*other.arrC)/mag)
                else:
                    return CUDAArray(self.arr/other.arr, self.arrC/other.arr)
            else:
                if other.arrC is not None:
                    mag = other.arr*other.arr+other.arrC*other.arrC
                    return CUDAArray(self.arr*other.arr/mag, -self.arr*other.arrC/mag)
                else:
                    return CUDAArray(self.arr/other.arr)
        if type(other) == complex:
            if self.arrC is not None:
                mag = other.real*other.real+other.imag*other.imag
                return CUDAArray((self.arr*other.real+self.arrC*other.imag)/mag, (self.arrC*other.real-self.arr*other.imag)/mag)
            else:
                mag = other.real*other.real+other.imag*other.imag
                return CUDAArray(self.arr*other.real/mag, -self.arr*other.imag/mag)
        else:
            if self.arrC is not None:
                return CUDAArray(self.arr/other, self.arrC/other)
            else:
                return CUDAArray(self.arr/other)

    def __rdiv__(self, other):
        if other.__class__ == CUDAArray:
            if self.arrC is not None:
                mag = self.arr*self.arr+self.arrC*self.arrC
                if other.arrC is not None:
                    return CUDAArray((self.arr*other.arr+self.arrC*other.arrC)/mag, (self.arr*other.arrC-self.arrC*other.arr)/mag)
                else:
                    return CUDAArray(self.arr*other.arr/mag, self.arr*other.arrC/mag)
            else:
                if other.arrC is not None:
                    return CUDAArray(other.arr/self.arr, other.arrC/self.arr)
                else:
                    return CUDAArray(other.arr/self.arr)
        if type(other) == complex:
            if self.arrC is not None:
                mag = self.arr*self.arr+self.arrC*self.arrC
                return CUDAArray((self.arr*other.real+self.arrC*other.imag)/mag, (self.arr*other.imag-self.arrC*other.real)/mag)
            else:
                return CUDAArray(self.arr*other.real/mag, self.arr*other.imag/mag)
        else:
            if self.arrC is not None:
                mag = self.arr*self.arr+self.arrC*self.arrC
                return CUDAArray(other*self.arr/mag, other*self.arrC/mag)
            else:
                return CUDAArray(other/self.arr)

    def __iadd__(self, other):
        return self.__add__(other)
        if other.__class__ == CUDAArray:
            self.arr += other.arr
        else:
            self.arr += other
        return self

    def __isub__(self, other):
        return self.__sub__(other)
        if other.__class__ == CUDAArray:
            self.arr -= other.arr
        else:
            self.arr -= other
        return self

    def __imul__(self, other):
        return self.__mul__(other)
        if other.__class__ == CUDAArray:
            self.arr *= other.arr
        else:
            self.arr *= other
        return self

    def __idiv__(self, other):
        return self.__div__(other)
        if other.__class__ == CUDAArray:
            self.arr /= other.arr
        else:
            self.arr /= other
        return self

    def __neg__(self):
        if self.arrC is not None:
            return CUDAArray(-self.arr, -self.arrC)
        return CUDAArray(-self.arr)

    def __setitem__(self, index, value):
        if self.arrC is not None:
            if type(value) == complex:
                self.arr[index] = value.real
                self.arrC[index] = value.imag
            else:
                self.arr[index] = value
                self.arrC[index] = 0
        else:
            self.arr[index] = value

    def __getitem__(self, index):
        if self.arrC is not None:
            return self.arrC[index]*1.0j+self.arrC[index]
        return self.arr[index]

    def __lt__(self,other):
	if other.__class__ == CUDAArray:
            other = other.arr
	else:
	    other = gpuarray.empty_like(self.arr).fill(other)
        return CUDAArray(self.arr.__lt__(other))
    def __le__(self,other):
        if other.__class__ == CUDAArray:
            other = other.arr
        return CUDAArray(self.arr.__le__(other))
    def __eq__(self,other):
        if other.__class__ == CUDAArray:
            other = other.arr
        return CUDAArray(self.arr.__eq__(other))
    def __ne__(self,other):
        if other.__class__ == CUDAArray:
            other = other.arr
        return CUDAArray(self.arr.__ne__(other))
    def __ge__(self,other):
        if other.__class__ == CUDAArray:
            other = other.arr
        return CUDAArray(self.arr.__ge__(other))
    def __gt__(self,other):
        if other.__class__ == CUDAArray:
            other = other.arr
            return CUDAArray(self.arr.__gt__(other))
        else:
            """
            compare = gpuarray.empty_like(self.arr)
	    compare.fill(other)
   	    return CUDAArray(self.arr.__gt__(compare))
            """
	    
	    gt = ElementwiseKernel(
                            "double *x, double a, double *z",
                            "z[i] = (x[i]>a)",
                            "gt_value")
            result = gpuarray.empty_like(self.arr)
            gt(self.arr, other, result)
            return CUDAArray(result)
 	    

    def __pow__(self,other):
        return self.power(other)

def random_normal(loc=0.0, scale=1.0, size=None):
    u1 = curandom.rand(size, dtype=numpy.float64)
    u2 = curandom.rand(size, dtype=numpy.float64)
    z1 = cumath.sqrt(-2.*cumath.log(u1))*cumath.cos(2.*numpy.pi*u2)
    return CUDAArray(scale*z1+loc)


GridArray.random_normal = random_normal
GridArray.GridArray = CUDAArray


