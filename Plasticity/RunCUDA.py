#!/usr/local/pub/enthought/bin/python
# shape and size of sim.
import simplejson 
import os, re
import numpy
from Plasticity.FieldInitializers import FieldInitializer
from Plasticity.TarFile import * 
import shutil as sh

dct = {"cudadir": "/b/plasticity/cuda-plasticity/", 
       "homedir": "/b/plasticity/", 
       "N":       128,
       "dim":      3, 
       "previous": "", 
       "postfix":  "r", 
       "method":   "lvp"}


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

def get_file(store, dir, file):
    return os.system("cp "+store+"/"+dir+"/"+file+" . ")

def put_file(store, dir, file):
    return os.system("cp "+file+" "+store+"/"+dir)

#==========================================================
# creates json headerigurations for simulations
def header_size(header, N, dim):
    header.update({"N": N})
    header.update({"CFLsafeFactor": 0.5})
    if dim == 3:
        header.update({"DIMENSION3":""})

def header_load(header, direction, rate, start=0.0):
    header.update({"LOADING": ""})
    header.update({"LOAD_DEF": direction.tolist()})
    header.update({"LOADING_RATE": rate})
    header.update({"LOAD_START": start})

def header_dynamics(header, dyn, **kwargs):
    if dyn == "gcd":
        header.update({"lambda": 0})
    if dyn == "mdp":
        header.update({"lambda": 1})
    if dyn == "lvp":
        header.update({"lambda": 0})
        header.update({"NEWGLIDEONLY": ""})
   
    #FIXME - both of these are incorrect
    if dyn == "slip":
        header.update({"lambda": 0})
        header.update(kwargs)
    if dyn == "vac":
        header.update({"VACANCIES": ""})
    
def header_dynamic_nucleation(header, dn=True):
    if dn == True:
        header.update({"DYNAMIC_NUCLEATION": 0})

def header_files(header, infile, outfile):
    header.update({"FILE_INPUT": infile})
    header.update({"FILE_OUTPUT": outfile})

def header_times(header, step, end):
    header.update({"TIME_STEP": step})
    header.update({"TIME_FINAL": end})

def header_fromfile(filename):
    tar = tar_open(filename)
    header = read_json(tar_getfile(tar, ".json"))
    tar.close()
    return header

#===============================================================
# actual simulation function
def simulation(dct):
    #homedir, cudadir, N, dim, previous, postfix, method, device, seed, header=None):
    
    prefix = method
    gridShape = (N,)*dim
    lengthscale = 0.2

    directory = prefix+str(len(gridShape))+"d"+str(N)
    unique    = directory+"_s"+str(seed)
    oldstub   = unique+"_"+previous
    oldfile   = unique+"_"+previous+".tar"
    currstub  = unique+"_"+previous+postfix
    currfile  = unique+"_"+previous+postfix+".tar"

    file_output= currstub+".plas"
    if os.path.isfile(currfile) == False:
        get_file(homedir, directory, currfile) 
    if os.path.isfile(currfile) == True:
        tar_extract(tar_open(currfile), ".plas")
        file_input = currstub+".plas"
    else:
        if previous == "":
            state = FieldInitializer.GaussianRandomInitializer(gridShape, lengthscale,seed,vacancy=None)
        else:
            get_file(homedir, directory, oldfile)
            tt,state = LoadTarState(oldfile)
        file_input = currstub + ".ics" 
        convertStateToCUDA(state, file_input)

    header = {"blank": "nothing"}
    header_size(header, N, dim)
    header_dynamics(header, method)
    header_dynamic_nucleation(header, postfix.find("d")>=0)
    header_files(header, file_input, currstub+".plas")

    #header_load(header, load_direction, load_rate, load_start)
    header_times(header, 1.0, 0.1)# time_step, time_end)
 
    headername = currstub+".h" 
    jsonname   = currstub+".json"
    write_json(header, jsonname)
    write_header(header, headername)
    exname = currstub + ".exe"

    here = os.getcwd()
    os.chdir(cudadir) 
    ret = os.system("make HEADER="+here+"/"+headername)
    os.system("rm -r obj/")
    os.system("mv ./build/release/plasticity "+here+"/"+exname)
    os.system("rm -r build/")

    if ret != 0:
        raise RuntimeError("Make failed!")

    os.chdir(here)
    ret = os.system("./"+exname+" --device="+str(device))

    if ret != 0:
        raise RuntimeError("Executable did not finish")

    if not os.path.isdir(currstub):
        os.mkdir(currstub)
    if os.path.isfile(exname):
        sh.move(exname, currstub)
    if os.path.isfile(jsonname):
        sh.move(jsonname, currstub)
    if os.path.isfile(headername):
        sh.move(headername, currstub)
    if os.path.isfile(file_input):
        sh.move(file_input, currstub)
    if os.path.isfile(file_output):
        sh.move(file_output, currstub)
    os.system("tar -cvf "+currfile+" "+currstub)
    os.system("rm -rf "+currstub)
    put_file(homedir, directory, currfile)  


from optparse import OptionParser

def main():
    parser = OptionParser()
    parser.add_option("-s", "--seed", dest="seed", action="store", type="int",
                      help="seed to use if generated RIC", default=0)
    parser.add_option("-d", "--device", dest="device", action="store", type="int",
                      help="the card to run on", default=0) 

    (options, args) = parser.parse_args()
    seed   = options.seed
    device = options.device

    dct.update({"device": options.device})
    dct.update({"seed":   options.seed})
    simulation(dct)

if __name__ == "__main__":
    main()


