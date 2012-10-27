#!/usr/local/pub/enthought/bin/python
# shape and size of sim.
import os, re
import numpy
from Plasticity.FieldInitializers import FieldInitializer
from Plasticity.TarFile import * 
import shutil as sh

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

#==========================================================
# creates json headerigurations for simulations
def write_header(dct, headername):
    headerfile = open_if_not(headername, "w")
    for key,val in dct.iteritems():
        headerfile.write("#define "+key+" "+list_to_crepr(val)+"\n")

def read_header(header):
    define_statement = re.compile("\#define\s([a-zA-Z0-9_\(\),]*)\s?(.*)$")

    dct = {}
    headerfile = open_if_not(header, "r")
    lines = headerfile.readlines()
    headerfile.close()

    for l in lines:
        try:
            var, val = define_statement.match(l.strip()).groups()   
            dct[var] = crepr_to_list(val)
        except:
            print "Not a valid line: ", l.strip()

    return dct

def list_to_crepr(arr):
    return repr(arr).replace("[","{").replace("]","}").replace("\'", "\"")

def crepr_to_list(arr):
    return arr.replace("{","[").replace("}","]").replace("\"","\'") 

def header_size(header, N, dim):
    header.update({"N": N})
    header.update({"CFLsafeFactor": 0.5})
    if dim == 3:
        header.update({"DIMENSION3":""})

def header_load(header, direction, rate, start=0.0):
    header.update({"LOADING": ""})
    if isinstance(direction, numpy.ndarray):
        header.update({"LOAD_DEF": direction.tolist()})
    else:
        header.update({"LOAD_DEF": direction})
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

def local_get(store, dir, file):
    return os.system("cp "+store+"/"+dir+"/"+file+" . ")

def local_put(store, dir, file):
    return os.system("cp "+file+" "+store+"/"+dir)

from Plasticity.Configure import *
#===============================================================
# actual simulation function
def simulation(dct, get_file=local_get, put_file=local_put):
    conf = Configuration(dct)
    homedir = conf.homedir
    cudadir = conf.cudadir
    N       = conf.N
    dim     = conf.dim
    previous= conf.previous
    postfix = conf.postfix
    method  = conf.method
    device  = conf.device
    seed    = conf.seed
    hash    = conf.hash

    prefix = method
    gridShape = (N,)*dim
    lengthscale = 0.2

    directory = prefix+str(len(gridShape))+"d"+str(N)
    unique    = directory+"_s"+str(seed)
    oldstub   = unique+"_"+previous
    oldfile   = unique+"_"+previous+".tar"
    currstub  = unique+"_"+previous+postfix
    currfile  = unique+"_"+previous+postfix+".tar"

    # we are going to load it
    if postfix != "r" and postfix != "d" and postfix != "":
        # this is the first load that determines the series
        if postfix == "0" or postfix == "1":
            load_direction = conf.load_direction
            load_rate = conf.load_rate
            time_start = 50.0 
            time_step = conf.time_step
            time_end = conf.time_end
            load_start = 0.0
        else:
            old = LoadTarJSON(oldfile)
            load_direction = old.load_direction
            load_rate = old.load_rate
            time_start = conf.time_start
            time_step = conf.time_step
            time_end = conf.time_end
            load_start = time_start*load_rate + old.load_start
    else:
        time_step = 1.0
        time_end = 50.0

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
            if os.path.isfile(oldfile) == False:
                get_file(homedir, directory, oldfile)
            tt,state = LoadTarState(oldfile, time=time_start)
        file_input = currstub + ".ics" 
        convertStateToCUDA(state, file_input)
    
    confname = currstub+".conf"
    write_json(conf, currstub+".conf")

    # ====================================================
    # begin non-standard things now
    header = {"blank": "nothing"}
    header_size(header, N, dim)
    header_dynamics(header, method)
    header_dynamic_nucleation(header, postfix.find("d")>=0)
    header_files(header, file_input, currstub+".plas")

    if postfix != "r" and postfix != "d" and postfix != "":
        header_load(header, load_direction, load_rate, load_start)
    header_times(header, time_step, time_end)
 
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

    # =======================================================
    # begin standard things again
    if os.path.isfile(confname):
        sh.move(confname, currstub)
    if os.path.isfile(file_input):
        sh.move(file_input, currstub)
    if os.path.isfile(file_output):
        sh.move(file_output, currstub)
    os.system("tar -cvf "+currfile+" "+currstub)
    os.system("rm -rf "+currstub)
    put_file(homedir, directory, currfile)  



