import Observer
import sys
import select
import os

from Constants import *

def PlotField(field, component):
    """ 
    Example Function that could be used from Interactive Control Observer
    """
    import pylab
    pylab.plot(field[component])
    pylab.show()

class InteractiveControlObserver(Observer.Observer):
    """
    Interact with command line. This is a proof of concept implementation
    that is able to run any command in python. Some more commands need to be
    implemented for use.

    This currently has problem working with any observer printing to stdout
    (i.e. VerboseTimestepObserver) since it gets all messed up together.

    usage:
    print time
    PlotField(state.betaP, (x,x))
    PlotField(state.CalculateSigma(), (x,x))
    """
    def __init__(self):
        self.buffer = ""

    def CheckData(self):
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

    def ParseData(self):
        while self.CheckData():
            c = os.read(0, 1)
            self.buffer += c
       
    def ProcessData(self, time, state):
        # Split lines from buffer
        lines = self.buffer.splitlines(True)
        linescheck = self.buffer.splitlines()
        if len(lines) > 1 or (len(lines) == 1 and lines[0] != linescheck[0]):
            command = linescheck[0] 
            self.buffer = self.buffer[len(lines[0]):]
            try:
                exec(command)
            except:
                pass

    def Update(self, time, state):
        self.ParseData()
        self.ProcessData(time, state)

