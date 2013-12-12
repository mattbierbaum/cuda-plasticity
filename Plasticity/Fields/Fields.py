from Plasticity.GridArray import GridArray
from Plasticity.GridArray import FourierSpaceTools

def NoneAllocator(*args, **kwargs):
    return None

class Field(object):
    """
    Implementation of multiplying rhoTilde by k^4:
      for elem in sigmaTilde.components:
        sigmaTilde[elem] = ksqsq[elem] * rhoTilde[elem]
    Implementation of J += sigma . rho:
      for elem in J.components:
        for dir in {'x','y','z'}:
          J[elem] += sigma[elem[0],dir] * rho[dir,elem[1]]
    """
    def __init__(self, gridShape, components, kspace=False, allocator=None):
        """ Initialize Field Class """
        self.gridShape = gridShape
        self.components = components 
        self.components.sort()
        self.data = {}        
        KgridShape = FourierSpaceTools.FourierSpaceTools(gridShape).gridShape 
        if allocator is None:
            allocator = GridArray.GridArray.zeros
        for component in components:
            if kspace is False:
                self.data[component] = allocator(gridShape)
            else:
                self.data[component] = allocator(KgridShape, dtype='complex')
            #self.data[component] = GridArray.GridArray(gridShape=gridShape)

    def evaluate(self):
        """
        Evaluate all postponed evaluations in every component
        """
        for component in self.components:
            self.data[component].evaluate()
        
    def GridShape(self):
        return self.gridShape

    def GridDimension(self):
        return len(self.GridShape())

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value
        
    def FFT(self, out=None):
        # FIXME it may not work properly
        if out is None:
            # FIXME Do in-place FFT somehow
            out = self.__class__(self.gridShape, self.components, kspace=True, allocator=NoneAllocator)
            for elem in self.components:
                out[elem] = self.data[elem].rfftn()
        else:
            for elem in self.components:
                out[elem] = self.data[elem].rfftn()
        return out

    def IFFT(self, out=None):
        # FIXME it may not work properly
        if out is None:
            # FIXME Do in-place FFT somehow
            out = self.__class__(self.gridShape, self.components, allocator=NoneAllocator)
            for elem in self.components:
                out[elem] = self.data[elem].irfftn(self.gridShape)
        else:
            for elem in self.components:
                out[elem] = self.data[elem].irfftn(self.gridShape)
        return out

    def __mul__(self, other):
        """
        You can only multiply intrinsically with an array of the same grid shape, or a scalar.
        Multiplying with a field needs contracting indices, which we don't deal with at this moment
        """
        result = self.__class__(self.gridShape, self.components, allocator=NoneAllocator)
        if isinstance(other, Field):
            # do the components match? do we need to check?
            for component in self.components:
                result[component] = self[component] * other[component]
        else:
            # this should work for numpy array or scalar other
            for component in self.components:
                result[component] = self[component] * other
        return result

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        """
        You can divide a field by a field of the same type, an array
        of same shape, or a scalar.
        """
        # FIXME all these operations are not complete and has problems
        result = self.__class__(self.gridShape, self.components, allocator=NoneAllocator)
        if isinstance(other, Field):
            # do the components match? do we need to check?
            for component in self.components:
                result[component] = self[component] / other[component]
        else:
            # this should work for numpy array or scalar other
            for component in self.components:
                result[component] = self[component] / other
        return result

    def __rdiv__(self, other):
        """
        rdiv is different from rmul due to the fact that rdiv is not
        commutable. implementation of this needs more discussion as to
        whether we want this to work like this with numpy arrays or scalars.
        """
        if isinstance(other, Field):
            return other.__div__(self)
        else:
            result = self.__class__(self.gridShape, self.components, allocator=NoneAllocator)
            # this should work for numpy array or scalar other
            for component in self.components:
                result[component] = other / self[component] 
            return result

    def __add__(self, other):
        """
        You can only add fields of the same shape(components), add
        an array of same grid shape, or a scalar.
        """
        result = self.__class__(self.gridShape, self.components, allocator=NoneAllocator)
        if not isinstance(other, Field):
            # this should work for numpy array or scalar other
            for component in self.components:
                result[component] = self[component] + other
        else:
            # do the components match? do we need to check?
            for component in self.components:
                result[component] = self[component] + other[component]
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """
        You can only subtract fields of the same shape(components), add
        an array of same grid shape, or a scalar.
        """
        result = self.__class__(self.gridShape, self.components, allocator=NoneAllocator)
        if not isinstance(other, Field):
            # this should work for numpy array or scalar other
            for component in self.components:
                result[component] = self[component] - other
        else:
            # do the components match? do we need to check?
            for component in self.components:
                result[component] = self[component] - other[component]
        return result

    def __rsub__(self, other):
        result = self.__class__(self.gridShape, self.components, allocator=NoneAllocator)
        if not isinstance(other, Field):
            # this should work for numpy array or scalar other
            for component in self.components:
                result[component] = other - self[component] 
        else:
            # do the components match? do we need to check?
            for component in self.components:
                result[component] = other[component] - self[component]
        return result

    def __neg__(self):
        result = self.__class__(self.gridShape, self.components, allocator=NoneAllocator)
        for component in self.components:
            result[component] = -self[component]
        return result

    def numpy_arrays(self):
        result = self.__class__(self.gridShape, self.components, allocator=NoneAllocator)
        for component in self.components:
            result[component] = self[component].numpy_array() 
        return result

    def fabs(self):
        result = self.__class__(self.gridShape, self.components, allocator=NoneAllocator)
        for component in self.components:
            result[component] = self[component].fabs() 
        return result

    def log(self):
        result = self.__class__(self.gridShape, self.components, allocator=NoneAllocator)
        for component in self.components:
            result[component] = self[component].log() 
        return result

    def sqrt(self):
        result = self.__class__(self.gridShape, self.components, allocator=NoneAllocator)
        for component in self.components:
            result[component] = self[component].sqrt() 
        return result

    def max(self):
        arr = []
        for component in self.components:
            arr.append(self[component].max())
        return max(arr)
 
    def min(self):
        arr = []
        for component in self.components:
            arr.append(self[component].min())
        return min(arr)
 
    def sum(self):
        arr = 0. 
        for component in self.components:
            arr += self[component].sum()
        return arr

    def modulus(self):
        arr = 0. 
        for component in self.components:
            arr += self[component]*self[component]
        return arr.sqrt()
 
    """
    def free(self):
        for c in self.components:
            self.data[c].free()
            self.data[c].free()
    """

class TensorField (Field):
    """
    General 2nd rank tensor field with 9 components in 3-D.
    """
    def __init__(self, gridShape, components=None, rank=None, directions=['x','y','z'], directionList=None, kspace=False, allocator=None):
        if components is None:
            if rank is not None:
                components = self.GenerateComponents(rank=rank, \
                                            directions=directions)
            elif directionList is not None:
                components = self.GenerateComponentsFromList(directionList)
            else:
                components = [('x','x'), ('x','y'), ('x','z'), \
                              ('y','x'), ('y','y'), ('y','z'), \
                              ('z','x'), ('z','y'), ('z','z')]
        super(TensorField, self).__init__(gridShape, components, kspace=kspace, allocator=allocator)
        #Field.__init__(self,gridShape, components, kspace=kspace, allocator=allocator)

    def GenerateComponents(self, rank=2, directions=['x','y','z']):
        """
        Generate all possible combinations for general tensor of rank-n
        in dimensions with axis given by directions.

        usage:
        GenerateComponents(rank=1) returns ['x','y','z']
        GenerateComponents(rank=2, directions=['x','y']) returns
                          [('x','x'), ('x','y'), ('y','x'), ('y','y')]
        """
        pool = []
        pool += directions
        for i in range(rank-1):
            newPool = []
            for elem in pool:
                for dir in directions:
                    newElement = tuple(list(elem)+[dir])
                    newPool.append(newElement)
            pool = newPool
        return pool

    def GenerateComponentsFromList(self, dirList):
        """
        Generate all possible combinations for general tensor of rank-n
        with its possible indices given by the list of lists.

        usage:
        GenerateComponentsFromList([['x', 'y'], ['x','y','z']])
        """
        pool = []
        pool += dirList[0]
        for i in range(1, len(dirList)):
            newPool = []
            for elem in pool:   
                for dir in dirList[i]:
                    newElement = tuple(list(elem)+[dir])
                    newPool.append(newElement)
            pool = newPool
        return pool
     
class SymmetricTensorField (TensorField):
    """
    Inherits TensorField since this is a symmetric tensor field.
    """
    def __init__(self, gridShape, components=None, rank=2, directions=None, kspace=False, allocator=None):
        """
        Symmetric Tensor Field is limited to rank 2 tensors only.
        """
        if components is None:
            if directions is not None:
                components = self.SymmetrizeComponents(\
                    self.GenerateComponents(rank=2, directions=directions))
            else:
                # This does not need to be symmetrized
                components = [('x','x'), ('x','y'), ('x','z'), \
                              ('y','y'), ('y','z'), \
                              ('z','z')]
        else:
            components = self.SymmetrizeComponents(components)
        Field.__init__(self,gridShape, components, kspace=kspace, allocator=allocator)

    def __getitem__(self, index):
        """
        __getitem__ for Symmetric Field sorts the index so that they refer to
        the correct data.
        """ 
        l = list(index)
        l.sort()
        return TensorField.__getitem__(self, tuple(l))
       
    def __setitem__(self, index, value):
        """
        __setitem__ for Symmetric Field sorts the index so that they refer to
        the correct data.
        """ 
        l = list(index)
        l.sort()
        TensorField.__setitem__(self, tuple(l), value)
        
    def SymmetrizeComponents(self, components):
        """
        SymmetrizeComponents symmetrizes the components list and returns
        ordered component list with only necessary components
        """
        dict = {}
        arr = []
        for component in components:
            l = list(component)
            l.sort()
            index = tuple(l)
            if index not in dict:
                arr.append(index)  
                dict[index] = 1
        return arr

class TensorScalarMixedField(Field):
    """
    General 9-component tensor field plus an additional scaler field.
    """
    def __init__(self, gridShape,components=None,kspace=False, allocator=None): 
       if components is None:
           components = [('x','x'), ('x','y'), ('x','z'), \
                         ('y','x'), ('y','y'), ('y','z'), \
                         ('z','x'), ('z','y'), ('z','z'), \
                         ('s','s')]
       Field.__init__(self,gridShape, components, kspace=kspace, allocator=allocator)

 
class VectorField (Field):
    """
    3 dimensional Vector field. 
    """
    def __init__(self, gridShape, components=None, kspace=False, allocator=None):
        components = ['x','y','z']
        Field.__init__(self,gridShape, components, kspace=kspace, allocator=allocator)


class ScalarField(Field):
    """
    3 dimensional Vector field. 
    """
    def __init__(self, gridShape, components=None, kspace=False, allocator=None):
        components = ['x']
        Field.__init__(self,gridShape, components, kspace=kspace, allocator=allocator)

class FCCSlipSystemField(Field):
    import scipy
    """
    Generates the Field for FCC Slip system.

    FIXME : Generalize to general slip system fields.
    """
    def __init__(self, gridShape, kspace=False, allocator=None):
        self.slipPlanes = scipy.array([[1,1,1],[1,1,-1],[1,-1,1],[-1,1,1]])
        self.slipDirections = {}
        for plane in self.slipPlanes:
            self.slipDirections[plane] = [scipy.array([i,j,k]) for i in range(-1,2) for j in range(-1,2) for k in range(-1,2) 
                                        if (scipy.dot(scipy.array([i,j,k]),plane)==0 and i*i + j*j + k*k == 2)]
        self.components = [(plane, direction) for plane in self.slipPlanes for directions in self.slipDirections[plane]]
        Field.__init__(self,gridShape, components, kspace=kspace, allocator=allocator)
    
