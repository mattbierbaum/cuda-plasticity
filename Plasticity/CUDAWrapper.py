import simplejson 
import tarfile
import os, re
import numpy
from Plasticity.FieldInitializers import FieldInitializer
import shutil as sh

#=====================================================
# small helper functions
def get_extension(filename):
    return os.path.splitext(filename)[1]

def open_if_not(filename, mode):
    if not isinstance(filename, str) or hasattr(filename, "read"):
        return filename
    else:
        return open(filename, mode)

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

def write_json(dct, filename):
    fi = open(filename, "w")
    fi.write(simplejson.dumps(dct))
    fi.close()

def read_json(jsonfile):
    return simplejson.loads(open_if_not(jsonfile, 'r').read())

def js_N(dct):
    return int(dct["N"])

def js_dim(dct):
    if "DIMENSION3" in dct.keys():
        return 3
    return 2

def tar_getmem(tar, extension):
    if not isinstance(tar, tarfile.TarFile):
        tar = tarfile.open(tar)
    for mem in tar.getmembers():
        if mem.isfile() and get_extension(mem.name) == extension:
            return mem

def tar_getfile(tar, extension):
    return tar.extractfile(tar_getmem(tar,extension))

def tar_extract(tar, extension):
    return tar.extract(tar_getmem(tar,extension))

def list_to_crepr(arr):
    return repr(arr).replace("[","{").replace("]","}").replace("\'", "\"")

def crepr_to_list(arr):
    return arr.replace("{","[").replace("}","]").replace("\"","\'") 


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
# creates json configurations for simulations
def conf_size(conf, N, dim):
    conf.update({"N": N})
    conf.update({"CFLsafeFactor": 0.5})
    if dim == 3:
        conf.update({"DIMENSION3":""})

def conf_load(conf, direction, rate, start=0.0):
    conf.update({"LOADING": ""})
    conf.update({"LOAD_DEF": direction.tolist()})
    conf.update({"LOADING_RATE": rate})
    conf.update({"LOAD_START": start})

def conf_dynamics(conf, dyn, **kwargs):
    if dyn == "gcd":
        conf.update({"lambda": 0})
    if dyn == "mdp":
        conf.update({"lambda": 1})
    if dyn == "lvp":
        conf.update({"lambda": 0})
        conf.update({"NEWGLIDEONLY": ""})
   
    #FIXME - both of these are incorrect
    if dyn == "slip":
        conf.update({"lambda": 0})
        conf.update(kwargs)
    if dyn == "vac":
        conf.update({"VACANCIES": ""})
    
def conf_dynamic_nucleation(conf, dn=True):
    if dn == True:
        conf.update({"DYNAMIC_NUCLEATION": ""})

def conf_files(conf, infile, outfile):
    conf.update({"FILE_INPUT": infile})
    conf.update({"FILE_OUTPUT": outfile})

def conf_times(conf, step, end):
    conf.update({"TIME_STEP": step})
    conf.update({"TIME_FINAL": end})

def conf_fromfile(filename):
    tar = tarfile.open(filename)
    conf = read_json(tar_getfile(tar, ".json"))
    tar.close()
    return conf

#===========================================================
# these are the interesting functions
def LoadTarState(file, time=None):
    tar = tarfile.open(file)
    dct = read_json(tar_getfile(tar, ".json"))
    t,s = FieldInitializer.LoadStateRaw(tar_getfile(tar, ".plas"), js_N(dct), js_dim(dct), time=time)
    tar.close()
    return t,s

def LoadTarJSON(file):
    tar = tarfile.open(file)
    dct = read_json(tar_getfile(tar, ".json"))
    tar.close()
    return dct    

def LoadTarICS(file):
    tar = tarfile.open(file)
    dct = read_json(tar_getfile(tar, ".json"))
    t,s = FieldInitializer.LoadStateRaw(tar_getfile(tar, ".ics"), js_N(dct), js_dim(dct), hastimes=False)
    tar.close()
    return t,s 

def simulation(homedir, cudadir, N, dim, previous, postfix, method, device, seed, header=None):
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
        tar_extract(tarfile.open(currfile), ".plas")
        file_input = currstub+".plas"
    else:
        if previous == "":
            state = FieldInitializer.GaussianRandomInitializer(gridShape, lengthscale,seed,vacancy=None)
        else:
            get_file(homedir, directory, oldfile)
            tt,state = LoadTarState(oldfile)
        file_input = currstub + ".ics" 
        convertStateToCUDA(state, file_input)

    conf = {"blank": "nothing"}
    conf_size(conf, N, dim)
    conf_dynamics(conf, method)
    conf_dynamic_nucleation(conf, postfix.find("d")>=0)
    conf_files(conf, file_input, currstub+".plas")

    #conf_load(conf, load_direction, load_rate, load_start)
    conf_times(conf, 1.0, 0.1)# time_step, time_end)
 
    headername = currstub+".h" 
    jsonname   = currstub+".json"
    write_json(conf, jsonname)
    write_header(conf, headername)
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

