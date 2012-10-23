import numpy
import numpy.fft as fft

def print_mem_usage():
    pass

class NumpyArray(numpy.ndarray):
    def __new__(subtype, data, dtype=None, copy=False):
        subarr = numpy.array(data, dtype=dtype, copy=copy)
        
        subarr = subarr.view(subtype)
        
        return subarr

    def symbol(self):
        """
        This returns a symbolic version of the array for lazy evaluation
        """
        return self
    _s = symbol
   
    def evaluate(self):
        """
        This function forces evaluation of any lazily postponed evaluation
        of the array. For the numpy array this does not do anything.
        """
        return self
    _e = evaluate

    def numpy_array(self):
        """
        This function should return the array as a numpy array type.
        
        By default we can return self, since we inherit a numpy array.
        But it should be noted that other implementations should not
        simply return self.
        """
        return self

    def rfftn(self):
        return NumpyArray(fft.rfftn(self))

    def irfftn(self, shape=None):
        return NumpyArray(fft.irfftn(self, shape))

    def roll(self, offset, direction):
        return NumpyArray(numpy.roll(self, offset, direction))

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
        if not isinstance(other, NumpyArray):
            return NotImplemented
        else:
            # NumpyArray returns other, since it is the fastest way to handle
            return other

    """
    Functions used
    """
    def sqrt(self):
        return NumpyArray(numpy.sqrt(self))

    def log(self):
        return NumpyArray(numpy.log(self))

    def abs(self):
        return NumpyArray(numpy.abs(self))

    def fabs(self):
        return NumpyArray(numpy.fabs(self))

    def sign(self):
        return NumpyArray(numpy.sign(self))

    def exp(self):
        return NumpyArray(numpy.exp(self))

    def power(self,x):
        return NumpyArray(numpy.power(self,x))

    """
    These do not need to be overloaded. However,
    numpy sum/max/min has richer options than we require.
    """
    def average(self):
        sum = self.sum()
        return sum/self.size

    def sum(self):
        return numpy.ndarray.sum(self)

    def max(self):
        return numpy.ndarray.max(self)

    def min(self):
        return numpy.ndarray.min(self)

    """
    static methods
    """
    def empty(shape, dtype='float64'):
        return NumpyArray(numpy.empty(shape, dtype=dtype))
    empty = staticmethod(empty)
 
    def zeros(shape, dtype='float64'):
        return NumpyArray(numpy.zeros(shape, dtype=dtype))
    zeros = staticmethod(zeros)


GridArray = NumpyArray
random_normal = numpy.random.normal
