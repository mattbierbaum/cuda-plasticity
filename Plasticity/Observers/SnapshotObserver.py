import Observer

class SnapshotObserver(Observer.Observer):
    """
    Snapshot control observer

    Calls update functions of the observers on its own list, every once the snapshot observer is called numberOfSteps.
    numberOfSteps is by default 100.
    """
    def __init__(self, numberOfSteps=100, timeStep=None):
        self.observers = []
        self.maxcount = numberOfSteps 
        self.count = 0
        self.timeStep = timeStep
        self.timeStamp = 0.

    def AddObserver(self, observer):
        """
        Add observers to the snapshot observer update list.
        """
        if type(observer) == list:
            self.observers += observer
        else: 
            self.observers.append(observer)

    def Update(self, time=None, state=None):
        if self.timeStep is not None and time is not None:
            # Use time step counts instead of number of step counts
            if self.timeStamp + self.timeStep <= time:
                self.timeStamp += self.timeStep
                for observer in self.observers:
                    observer.Update(time=time, state=state)
        else:
            # Use ordinary number of step counts
            self.count = self.count + 1
            if self.count > self.maxcount:
                self.count = 0
                for observer in self.observers:
                    observer.Update(time=time, state=state)
