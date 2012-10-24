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
    unique    = directory+"_s"+str(seed)
    oldstub   = unique+"_"+previous
    oldfile   = unique+"_"+previous+".tgz"
    currstub  = unique+"_"+previous+postfix
    currfile  = unique+"_"+previous+postfix+".tgz"

    get_file(directory, currfile) 
    
    if os.path.isfile(os.path.join(directory, currfile)) == True:
        wrap.tar_extract(currfile, ".plas")
        file_input = currstub+".plas"
    else:
        if previous == "":
            state = FieldInitializer.GaussianRandomInitializer(gridShape, lengthscale,seed,vacancy=None)
        else:
            get_file(directory, oldfile)
            tt,state = wrap.LoadTarState(oldfile)
        file_input = currstub + ".ics" 
        convertStateToCUDA(state, file_input)

    conf = {"blank": "nothing"}
    wrap.conf_size(conf, N, dim)
    wrap.conf_dynamics(conf, method)
    wrap.conf_dynamic_nucleation(conf, postfix.contains("d"))
    wrap.conf_files(conf, file_input, currstub+".plas")

    wrap.conf_load(conf, load_direction, load_rate, load_start)
    wrap.conf_times(conf, time_step, time_end)
    
    wrap.write_json_to_h(conf, currstub+".h")
    exname = currstub + ".exe"
 
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


