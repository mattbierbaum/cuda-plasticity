from Plasticity.PlasticityStates import PlasticityState
from Plasticity.Fields import Fields
from Plasticity.GridArray import GridArray
from Plasticity.GridArray import FourierSpaceTools

class BurgersState(PlasticityState.PlasticityState):
    def __init__(self, gridShape, field=None, nu=0.3, mu=0.5, inherit=None):
        PlasticityState.PlasticityState.__init__(self,gridShape,field=field,nu=nu,mu=mu,inherit=inherit)
            
    def GetOrderParameterField(self):
        return self.field

    def UpdateOrderParameterField(self,field):
        if field is None:
            self.field = Fields.TensorField(self.gridShape, components=['u','v','w'])
        else:
            self.field = field

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['ktools']
        del odict['sinktools']
        odict['field'] = self.GetOrderParameterField().numpy_arrays()
        return odict

    def __setstate__(self,dict):
        self.__dict__.update(dict)
        if self.gridShape not in PlasticityState.fourierTools:
            PlasticityState.fourierTools[self.gridShape] = FourierSpaceTools.FourierSpaceTools(self.gridShape)
            import numpy
            PlasticityState.fourierSinTools[self.gridShape] = FourierSpaceTools.FourierSpaceTools(self.gridShape, func=numpy.sin)
        self.ktools = PlasticityState.fourierTools[self.gridShape]
        self.sinktools = PlasticityState.fourierSinTools[self.gridShape]
        for component in self.field.components:
            self.field[component] = GridArray.GridArray(self.field[component])
        self.UpdateOrderParameterField(self.field)
         
