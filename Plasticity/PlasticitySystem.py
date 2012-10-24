import sys

class PlasticitySystem:
    """
    Base class for Plasticity Simulations
    """
    def __init__(self, gridShape, state, mover, dynamics, observers = []):
        """ 
        Initializes the System, set to default if argument is not supplied 
        """
        self.gridShape = gridShape
        self.state = state
        self.mover = mover
        self.dynamics = dynamics
        self.observers = observers

    def Run(self, startTime=0.0, endTime=10.0, fixedTimeStep=None):
        """ 
        Invokes the mover to run from start time to end time 
        """
        self.mover.Run(self.state, self.observers, self.dynamics, startTime, endTime, fixedTimeStep=fixedTimeStep)

class SimulatedSystem(PlasticitySystem):
    import pickle
    def __init__(self, filename):
        """
        Load from a pickled state
        """
        self.file = open(filename)
        self.queue = []

    def FetchState(self):
        if len(self.queue) > 0:
            time, state = self.queue.pop(0)
        else:    
            time = pickle.load(self.file)
            state = pickle.load(self.file)
        return time, state

    def QueueState(self, time, state):
        self.queue.append([time, state])
        
    def Run(self, startTime=0.0, endTime=10.0, fixedTimeStep=None):
        while True:
            time, state = self.FetchState()
            if time < startTime:
                continue
            if time > endTime:
                self.QueueState(time, state)
                break
            for observer in self.observers:
                observer.Update(time, state)
    
