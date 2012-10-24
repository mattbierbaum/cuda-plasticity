#!/usr/local/pub/enthought/bin/python
# shape and size of sim.
cudadir  = "/b/plasticity/cuda-plasticity/"
homedir  = "/b/plasticity/"
dim      = 3
previous = ""
postfix  = "r"
method   = "lvp"

import numpy
import PlasticitySystem
import FieldInitializer
import VacancyState, PlasticityState
from Constants import *
import GridArray
import Fields
import CUDAWrapper as wrap
from optparse import OptionParser
import os, sys, getopt

def convertStateToCUDA(state, filename):
    arr = []
    order = [('x', 'x'), ('x', 'y'), ('x', 'z'),
             ('y', 'x'), ('y', 'y'), ('y', 'z'),
             ('z', 'x'), ('z', 'y'), ('z', 'z'), ('s', 's')] 
    try:  
        for c in order:
            arr.append( state.betaP_V[c].transpose().copy() )
    except:
        for c in state.betaP.components:
            arr.append( state.betaP[c].transpose().copy() )

    arr = numpy.array(arr)
    arr.tofile(filename)

def get_file(dir, file):
    return os.system("cp "+homedir+"/"+dir+"/"+file+" . ")

def put_file(dir, file):
    return os.system("cp "+file+" "+homedir+"/"+dir)

def simulation(device, seed, header=None):
    prefix = method
    directory = prefix+str(len(gridShape))+"d"+str(N)
    oldfile   = prefix+str(len(gridShape))+"d"+str(N)+"_s"+str(seed)+"_"+previous+".tgz"
    filestub  = prefix+str(len(gridShape))+"d"+str(N)+"_s"+str(seed)+"_"+previous+postfix
    currfile  = prefix+str(len(gridShape))+"d"+str(N)+"_s"+str(seed)+"_"+previous+postfix+".tgz"

    get_file(directory, currfile) 
    
    if os.path.isfile(os.path.join(directory, currfile)) == True:
        wrap.tar_extract(currfile, ".plas")
    else:
        get_file(directory, oldfile)

        if previous == "":
            state = FieldInitializer.GaussianRandomInitializer(gridShape, lengthscale,seed,vacancy=None)
        else:
            tt,state = wrap.LoadTarState(oldfile)

    defines += "#define FILE_OUTPUT \""+filename+"\" \n"

    if lics <= 5:
        file_input = filestub + ".ics" 
        convertStateToCUDA(state, file_input)
    else:
        file_input = oldfile
    
    defines += "#define FILE_INPUT \""+file_input+"\" \n" 
    defines += "#define TIME_FINAL "+str(time_end)+" \n"
    defines += "#define TIME_STEP "+str(time_step)+" \n"
 
    headerreal = "generated.h" 
    headerfile = filestub + ".h" 

    for name in [headerreal, headerfile]:
        makefile = open(name, "w")
        makefile.write(defines)
        makefile.close()
    
    exname = filestub + ".exe"
 
    import os
    print os.system("make")
    print os.system("rm -r obj/")
    print os.system("mv ./build/release/plasticity "+exname)
    print os.system("rm -r build/")
    print os.system("rm -r generated.h")
    print os.system("./"+exname+" --device="+str(device))
    print put_file(directory, headerfile) 
    print put_file(directory, filename)  

def main():
    parser = OptionParser()
    parser.add_option("-s", "--seed", dest="seed", action="store", type="int",
                      help="seed to use if generated RIC", default=0)
    parser.add_option("-d", "--device", dest="device", action="store", type="int",
                      help="the card to run on", default=0) 
    parser.add_option("-r", "--rerun", dest="header", action="store", 
                      help="specify a header file to rerun", default=None)

    (options, args) = parser.parse_args()
    seed   = options.seed
    device = options.device
    header = options.header

    simulation(device=device, seed=seed)

if __name__ == "__main__":
    main()


