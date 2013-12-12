from Plasticity.Fields import Fields
from Plasticity import NumericalMethods
from Plasticity.PlasticityStates import PlasticityState
from Plasticity.GridArray import FourierSpaceTools

from Plasticity.Constants import *
import numpy as np

ME = NumericalMethods.ME
fourierTools = {}
fourierSinTools = {}

class SmecticState(object):
    def __init__(self, gridShape, field=None, inherit=None):
        self.gridShape = gridShape
        self.dimension = len(gridShape)
        self.UpdateOrderParameterField(field)
        
        if inherit is not None:
            self.ktools = inherit.ktools
            self.sinktools = inherit.sinktools
        else:
            if gridShape not in fourierTools:
                fourierTools[gridShape] = FourierSpaceTools.FourierSpaceTools(gridShape)
                fourierSinTools[gridShape] = FourierSpaceTools.FourierSpaceTools(gridShape, func=np.sin)
            self.ktools = fourierTools[gridShape]
            self.sinktools = fourierSinTools[gridShape]

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['ktools']
        del odict['sinktools']
        odict['field'] = self.GetOrderParameterField().numpy_arrays()
        return odict 

    def GetOrderParameterField(self):
        return self.field

    def UpdateOrderParameterField(self, newfield):
        if newfield is None:
            self.field = Fields.VectorField(self.gridShape)
        else:
            self.field = newfield

    def enforceBCs(self):
        field = self.GetOrderParameterField()
        for c in field.components:
            temp = field[c][...,:field.gridShape[-1]/2]
            field[c][...,:field.gridShape[-1]/2] = temp
            if c == 'z':
                temp[...,0] = 0
                field[c][...,field.gridShape[-1]/2:] = -np.roll(temp[...,::-1], 1, 2)
            else:
                field[c][...,field.gridShape[-1]/2:] = temp[...,::-1]
        self.UpdateOrderParameterField(field)

    def removeCurl(self): 
        field = self.GetOrderParameterField()
        ffield = self.field.FFT()

        kdotf = np.zeros(self.ktools.kx.shape).astype('complex')
        for c in field.components:
            kdotf += self.ktools.k[c] * ffield[c] / self.ktools.kSq
        kdotf = np.nan_to_num(kdotf)

        for i in field.components:
            ffield[i] = self.ktools.k[i] * kdotf
        field = ffield.IFFT()

        self.UpdateOrderParameterField(field)

    def CalculateDisplacementField(self):
        field = self.GetOrderParameterField()
        ffield = self.field.FFT()

        kdotf = np.zeros(self.ktools.kx.shape).astype('complex')
        for c in field.components:
            kdotf += self.ktools.k[c] * ffield[c] #/ (self.ktools.kSq + ME)

        return np.fft.irfftn(kdotf)

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(self.gridShape, field=self.GetOrderParameterField() * other.GetOrderParameterField(), inherit=self)
        else:
            return self.__class__(self.gridShape, field=self.GetOrderParameterField() * other, inherit=self)
    def __add__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(self.gridShape, field=self.GetOrderParameterField() + other.GetOrderParameterField(), inherit=self)
        else:
            return self.__class__(self.gridShape, field=self.GetOrderParameterField() + other, inherit=self)
    def __div__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(self.gridShape, field=self.GetOrderParameterField() / other.GetOrderParameterField(), inherit=self)
        else:
            return self.__class__(self.gridShape, field=self.GetOrderParameterField() / other, inherit=self)
    def __sub__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(self.gridShape, field=self.GetOrderParameterField() - other.GetOrderParameterField(), inherit=self)
        else:
            return self.__class__(self.gridShape, field=self.GetOrderParameterField() - other, inherit=self)

    def __rmul__(self, other):
        return self.__mul__(other)
    def __radd__(self, other):
        return self.__add__(other)
    def __rdiv__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(self.gridShape, field=other.GetOrderParameterField() / self.GetOrderParameterField(), inherit=self)
        else:
            return self.__class__(self.gridShape, field=other / self.GetOrderParameterField(), inherit=self)
    def __rsub__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(self.gridShape, field=other.GetOrderParameterField() - self.GetOrderParameterField(), inherit=self)
        else:
            return self.__class__(self.gridShape, field=other - self.GetOrderParameterField(), inherit=self)
