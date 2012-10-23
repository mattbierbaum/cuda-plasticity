#!/usr/local/pub/enthought/bin/python
# shape and size of sim.
homedir="/b/plasticity/"
dim = 3

previous = ""
postfix = "r"

method = 0
"""
0 : NewGlideOnly
1 : Upwind
2 : Upwind glide-climb
"""

# if we are just starting, make the ICs
if previous == "":
    ics = 1
# if we just relaxed in any way, go to end of file
elif previous[-1] == "b" or previous[-1] == "r" or previous[-1] == "d": 
    ics = 3
# otherwise load from a specific time
else:
    ics = 5
"""
1 : random gaussian with lengthscale
2 : load old file (state file)
3 : load old file (raw file)
4 : load old file (raw init file)
5 : load old file at time_start (raw file) 
6 : restart old file (raw file)
"""

import numpy
dir = "./"
lengthscale = 0.28
if postfix == "y" or postfix == "y2":
    loaddir = numpy.array([0.0,1.0,0.0]) 
elif postfix == "xy" or postfix == "xy2":
    loaddir = numpy.array([-1.0,1.0,0.0]) 
elif postfix == "yz" :
    loaddir = numpy.array([0.0,1.0,-1.0]) 
else:
    loaddir = numpy.array([-0.5,1.0,-0.5]) 

if method == 2:
    Lambda = 0
else:
    Lambda = 1

if "d" in postfix:
    nucleation = 1
else:
    nucleation = 0

if "0" in previous:
    loadrateprev =  0.05
    if "a" in postfix:
        loadrate =  loadrateprev
    if "b" in postfix:
        loadrate =  0.0
    if "c" in postfix:
        loadrate = -loadrateprev

if "1" in previous:
    loadrateprev =  0.005
    if "a" in postfix:
        loadrate =  loadrateprev
    if "b" in postfix:
        loadrate =  0.0
    if "c" in postfix:
        loadrate = -loadrateprev

load_start = 0.0
for char in previous:
    if char == "0":
        load_start += 4.0
    if char == "1":
        load_start += 4.0 
    if char == "a":
        load_start += 4.0
    if char == "b":
        load_start += 0.0 
    if char == "c":
        load_start -= 4.0

#if "0" in postfix or "y" in postfix or "xy" in postfix:
if postfix == "0" or postfix == "y" or postfix == "xy":
    loadrate = 0.05
    time_start = 0.0
    time_step = 0.05 / loadrate
    time_end  = 5.0 / loadrate

if postfix == "y2":
    loadrate = 0.005
    time_start = 0.0
    time_step = 0.05 / loadrate
    time_end  = 5.0 / loadrate

if postfix == "xy2" or postfix == "yz":
    loadrate = 0.01
    time_start = 0.0
    time_step = 0.05 / loadrate
    time_end  = 5.0 / loadrate

if "1" in postfix:
    loadrate = 0.005
    time_start = 0.0
    time_step = 0.05 / loadrate
    time_end  = 5.0 / loadrate

if "l" in postfix:
    loadrate = 0.005
    time_start = 70.0
    time_step = 0.5  #0.05 / loadrate
    time_end  = 50.0 #5.0 / loadrate
    load_start = 70*loadrate

if "w" in postfix:
    loadrate = 0.005
    time_start = 15.0
    time_step = 0.05  #0.05 / loadrate
    time_end  = 2.0 #5.0 / loadrate
    load_start = 70*loadrate + 15*loadrate
 
if "d" in postfix or "r" in postfix:
    time_start = 0.0
    time_step = 1.0
    time_end = 50.0
 
if "b" in postfix: 
    time_start = 4.0 / loadrateprev
    time_step  = 1.0
    time_end   = 50.0

if "a" in postfix or "c" in postfix: 
    time_start = 4.0  / loadrateprev
    time_step  = 0.05 / loadrateprev
    time_end   = 5.0  / loadrateprev

if dim == 2:
    N = 512 
    gridShape = (N,N)
else:
    N = 64
    gridShape = (N,N,N)

import PlasticitySystem
import FieldInitializer
import VacancyState, PlasticityState
from Constants import *
import GridArray
import WallInitializer
import Fields

import scipy.weave as weave
from optparse import OptionParser
import os, sys, getopt


def identity(a):
    greater = (a>numpy.pi).astype(float)
    return a-2*numpy.pi*greater

def md5_state(state):
    import hashlib
    md5 = hashlib.md5()
    md5.update(str(state.GetOrderParameterField().data))
    return md5.hexdigest()

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

def Relaxation_BetaPV(device, seed, header=None):
    import os
    lics = ics

    defines = ""
    defines += "#define N "+str(N)+" \n"
    defines += "#define CFLsafeFactor 0.5 \n"
    if len(gridShape) == 3:
        defines += "#define DIMENSION3 \n"

    if nucleation == 1:
        defines += "#define DYNAMIC_NUCLEATION \n"

    ##--------------------------------------------------------------------------------------
    if method == 0:
        defines += "#define lambda 0 \n"
        defines += "#define NEWGLIDEONLY \n"
        prefix = "lvp" 

    elif method == 1 or method == 2:
        defines += "#define lambda "+str(Lambda)+" \n"
        if Lambda == 1:
            prefix = "mdp"
        else:
            prefix = "gcd"

    if postfix != "r" and postfix != "d" and postfix != "b":
        defines += "#define LOADING \n"
        defines += "#define LOAD_DEF {{"+str(loaddir[0])+",0.0,0.0},{0.0,"+str(loaddir[1])+",0.0},{0.0,0.0,"+str(loaddir[2])+"}} \n"
        defines += "#define LOADING_RATE "+str(loadrate)+" \n"
        defines += "#define LOAD_START "+str(load_start)+" \n"
    ##-------------------------------------------------------------------------------------

    directory= prefix+str(len(gridShape))+"d"+str(N)
    
    oldfile = prefix+str(len(gridShape))+"d"+str(N)+"_s"+str(seed)+"_debug_"+previous+postfix+".plas"

    get_file(directory, oldfile) #os.system("msscmd cd "+directory+", get "+oldfile)
    if os.path.isfile(oldfile):
        lics = 6
    else:
        oldfile  = prefix+str(len(gridShape))+"d"+str(N)+"_s"+str(seed)+"_debug_"+previous+".plas"

    #if lics != 6:
    #    oldfile  = prefix+str(len(gridShape))+"d"+str(N)+"_s"+str(seed)+"_debug_"+previous+".plas"
    #else:
    #    oldfile  = prefix+str(len(gridShape))+"d"+str(N)+"_s"+str(seed)+"_debug_"+previous+postfix+".plas"
    filestub = prefix+str(len(gridShape))+"d"+str(N)+"_s"+str(seed)+"_debug_"+previous+postfix
    filename = filestub + ".plas"
    file_input = oldfile

    #========================================================================
    if   lics == 1:
        state = FieldInitializer.GaussianRandomInitializer(gridShape, lengthscale,seed,vacancy=None)
    elif lics == 2:
        get_file(directory, oldfile)        #os.system("msscmd cd "+directory+", get "+oldfile)
        tt,state = FieldInitializer.LoadState(oldfile)
    elif lics == 3:
        get_file(directory, oldfile)        #os.system("msscmd cd "+directory+", get "+oldfile)
        tt,state = FieldInitializer.LoadStateRaw(oldfile,N,len(gridShape),israw=False)
    elif lics == 4:
        get_file(directory, oldfile)        #os.system("msscmd cd "+directory+", get "+oldfile)
        tt,state = FieldInitializer.LoadStateRaw(oldfile,N,len(gridShape),israw=True)
    elif lics == 5:
        get_file(directory, oldfile)        #os.system("msscmd cd "+directory+", get "+oldfile)
        tt,state = FieldInitializer.LoadStateRaw(oldfile,N,len(gridShape),israw=False, time=time_start)
    #elif lics == 6:
    #    os.system("msscmd cd "+directory+", get "+oldfile)
    ##========================================================================

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
    print put_file(directory, headerfile) #os.system("msscmd cd "+directory+", put "+headerfile)
    print put_file(directory, filename)   #os.system("msscmd cd "+directory+", put "+filename)

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

    Relaxation_BetaPV(device=device, seed=seed)

if __name__ == "__main__":
    main()


