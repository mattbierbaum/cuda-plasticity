import pickle
import numpy
import sys

from Plasticity import NumericalMethods

class Observer:
    """Base class for Observers"""
    def __init__(self):
        pass

    def Update(self, time=None, state=None):
        pass

class RecallStateObserver(Observer):
    def __init__(self):
        self.state = None

    def Update(self, time, state):
        self.state = state


class RecordStateObserver(Observer):
    """
    This Observer saves the state time series to a file using pickle.
    """
    def __init__(self, filename=None, mode='w'):
        if filename is not None:
            self.file = open(filename, mode)
        else:
            self.file = sys.stdout
        
    def Update(self, time, state):
        """
        Pickle dump time and state consecutively. Uses protocol 2 to save space. (Binary data)
        """
        #FIXME
        """
        pickle dumping will not take care of different types of arrays we may
        use. We might need to overload the way pickle writes arrays so that
        they are stored in the same way for different implementations.
        """
        pickle.dump(time, self.file, protocol=2)
        pickle.dump(state, self.file, protocol=2)
        """
        flush so that you can see the states updated immediately
        """
        self.file.flush()


class RecordStateObserverRaw(Observer):
    """
    This Observer saves the state time series to a file using pickle.
    """
    def __init__(self, filename=None, mode='wb'):
        if filename is not None:
            self.file = open(filename, mode)
        else:
            self.file = sys.stdout
        
    def Update(self, time, state):
        """
        flush so that you can see the states updated immediately
        """
        self.file.write(numpy.array([time]).tostring())
        field = state.GetOrderParameterField()
        for c in field.components:
            self.file.write(field.data[c].tostring())
        self.file.flush()

class VerboseTimestepObserver(Observer):
    """
    Observer that prints current time on every update
    """
    def Update(self, time, state):
        import sys
        sys.stdout.write("t = %f\r" % time)
        sys.stdout.flush()
        #print "t = %f" % time   

class VerboseTotalEnergyObserver(Observer):
    """
    Observer that prints the total energy on every update
    """
    def Update(self, time, state):
        print "Total Energy = %f"% numpy.sum(state.CalculateElasticEnergy())
        print "t = %f" % time   

class EnergyDownhillObserver(Observer):
    """
    Checks whether energy goes downhill.
    """
    def __init__(self):
        self.PreviousEnergy = None
        self.countTotal = 0
        self.countBad = 0

    def Update(self, time, state):
        Energy = state.CalculateElasticEnergy().sum()
        self.countTotal += 1
        if self.PreviousEnergy is not None and self.PreviousEnergy < Energy:
            self.countBad += 1
        print "%d/%d" % (self.countBad, self.countTotal)
        self.PreviousEnergy = Energy
 
class PlotFieldObserver(Observer):
    """
    Pylab field plotting observer. By default plots the order parameter field of the state.
    """
    def __init__(self, component, filename=None, yaxis=None):
        self.component = component
        self.filename = filename
        self.yaxis = yaxis

    def Update(self, time, state):
        if state.dimension == 1:
            self.Plot1DField(state.GetOrderParameterField()[self.component], time)
        elif state.dimension == 2:
            self.Plot2DField(state.GetOrderParameterField()[self.component], time)

    def Plot1DField(self, field, time):
        import pylab
        pylab.figure()
        pylab.plot(field)
        if self.yaxis is not None:
            axis = pylab.axis()
            pylab.axis([axis[0], axis[1]] + self.yaxis)
        
        if self.filename is not None:
            # FIXME
            """
            Assuming a specific format that the component takes is not general and
            thus should be avoided. Instead we can opt to have a format string for
            the component or even assume that the filename already contains that
            information.
            """
            field.tofile(self.filename+str(time).replace('.','_')+'.dat')
            pylab.savefig(self.filename+'%.3f'%time+'.png')
        else:
            pylab.show()

    def Plot2DField(self, field, time):
        import pylab
        pylab.figure()
        pylab.imshow(field, interpolation='nearest')
        if self.filename is not None:
            field.tofile(self.filename+str(time).replace('.','_')+'.dat')
            pylab.savefig(self.filename+'%.3f'%time+'.png')
        else:
            pylab.show()


class PlotSigmaObserver(PlotFieldObserver):
    """
    Plots the stress calculated by PlasticityState class.
    """
    def Update(self, time, state):
        if state.dimension == 1:
            self.Plot1DField(state.CalculateSigma()[self.component], time)
        elif state.dimension == 2:
            self.Plot2DField(state.CalculateSigma()[self.component], time)
        

class PlotRhoSymmetricObserver(PlotFieldObserver):
    """
    Plots the dislocation density calculated by PlasticityState class.
    """
    def Update(self, time, state):
        if state.dimension == 1:
            self.Plot1DField(state.CalculateRhoSymmetric()[self.component],time)
        elif state.dimension == 2:
            self.Plot2DField(state.CalculateRhoSymmetric()[self.component],time)

class PlotRhoSymmetricMagnitudeObserver(PlotFieldObserver):
    """
    Plots the dislocation density calculated by PlasticityState class.
    """
    def __init__(self, filename=None, yaxis=None):
        self.filename = filename
        self.yaxis = yaxis

    def Update(self, time, state):
        rho = state.CalculateRhoFourier()
        rhoMag = rho.modulus()
        if state.dimension == 1:
            self.Plot1DField(rhoMag,time)
        elif state.dimension == 2:
            self.Plot2DField(rhoMag,time)
        
class PlotEnergyDensityObserver(PlotFieldObserver):
    """
    Plots the energy density calculated by PlasticityState class.
    """
    def __init__(self, filename=None, yaxis=None):
        self.filename = filename
        self.yaxis = yaxis

    def Update(self, time, state):
        Energy = state.CalculateElasticEnergy()
        if state.dimension == 1:
            self.Plot1DField(Energy, time)
        elif state.dimension == 2:
            self.Plot2DField(Energy, time)


class PlotEnergyFluxSymmetricObserver(PlotFieldObserver):
    """
    Plots the energy flux density calculated by PlasticityState class.
    """
    def __init__(self, filename=None, yaxis=None):
        self.filename = filename
        self.yaxis = yaxis

    def Update(self, time, state):
        energy = state.CalculateElasticEnergy()
        if state.dimension == 1:
            flux = NumericalMethods.SymmetricDerivative(energy,state.gridShape,0) 
            self.Plot1DField(flux, time)
        elif state.dimension == 2:
            """
            Don't know what to do yet
            """
            pass

