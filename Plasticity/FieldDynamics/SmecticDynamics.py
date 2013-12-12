from Plasticity.FieldDynamics.FieldDynamics import *
from Plasticity.FieldDynamics.CentralUpwindHJ import *
from Plasticity.Constants import *
from Plasticity.FieldDynamics.CentralUpwindHJBetaPDynamics3D import *

from Plasticity.PlasticityStates import PlasticityState
from Plasticity.GridArray import GridArray

from Plasticity import NumericalMethods
from Plasticity.Fields import Fields

import numpy

class SmecticDynamics(object): 
    def __init__(self, Dx=None, B=1.0, K=1.0):
        self.B = B
        self.K = K

    def CalculateFlux(self, time, state, CFLCondition=False):
        field = state.GetOrderParameterField()
        ffield = field.FFT()

        rhs = Fields.VectorField(state.gridShape)
        fdivN = Fields.VectorField(state.gridShape, kspace=True)
        fdivdivN = Fields.VectorField(state.gridShape, kspace=True)
        Nsq = numpy.zeros(state.gridShape)

        for c in field.components:
            Nsq += field[c] * field[c]
            fdivN['x'] = -1.j * state.ktools.k[c] * ffield[c]

        for c in field.components:
            fdivdivN[c] = -1.j * state.ktools.k[c] * fdivN['x']
        
        divdivN = fdivdivN.IFFT()
        for c in field.components:
            rhs[c] = self.B*(1-Nsq)*field[c] + 2*self.K*divdivN[c]

        if CFLCondition:
            return rhs, rhs
        return rhs 

