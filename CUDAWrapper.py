import simplejson 
import tarfile
import os, re

#=====================================================
# small helper functions
def get_extension(filename):
    return os.path.splitext(filename)[1]

def open_if_not(filename, mode):
    if not isinstance(filename, str) or hasattr(filename, "read"):
        return filename
    else
        return open(filename, mode)

def write_json_to_h(dct, headername):
    dct = simplejson.loads(json)
    headerfile = open_if_not(headername, "w")
    for key,val in dct.iteritems():
        headerfile.write("#define "+key+" "+list_to_crepr(val))

def read_header(header):
    define_statement = re.compile("\#define\s([a-zA-Z0-9_\(\),]*)\s?(.*)$")

    dct = {}
    headerfile = open_if_not(header, "r")
    lines = headerfile.readlines()
    headerfile.close()

    for l in lines:
        try:
            var, val = define_statement.match(l.strip()).groups()   
            dct[var] = val
        except:
            print "Not a valid line: ", l.strip()

    return dct

def read_json(jsonfile):
    return simplejson.loads(crepr_to_list(open_if_not(jsonfile).read()))

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
    return tar.extract(tar_getmem(extension))

def list_to_crepr(arr):
    return repr(arr).replace("[","{").replace("]","}")

def crepr_to_list(arr):
    return arr.replace("{","[").replace("}","]") 

#==========================================================
# creates json configurations for simulations
def conf_size(conf, N, dim):
    conf = {"N": N}
    conf.update({"CFLsafeFactor": 0.5})
    if dim == 3:
        conf.update({"DIMENSION3":""})
    return conf

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
        conf.update({"NEWGLIDEONLY", ""})
   
    #FIXME - both of these are incorrect
    if dyn == "slip":
        conf.update({"lambda": 0})
        conf.update(kwargs)
    if dyn == "vac":
        conf.update({"VACANCIES": ""})
    
def conf_dynamic_nucleation(conf, dn=True):
    if dn == True:
        conf.update({"DYNAMIC_NUCLEATION", ""})

def conf_fromfile(filename):
    tar = tarfile.open(filename)
    conf = read_json(tar_getfile(tar, ".json"))
    tar.close()
    return conf

#===========================================================
# these are the interesting functions
def LoadTarState(file, time=None):
    filename, extenstion = os.path.splitext(file)
    tar = tarfile.open(file)
    dct = read_json(tar_getfile(tar, ".json"))
    t,s = FieldInitializer.LoadStateRaw(tar_getfile(tar, ".plas"), js_N(dct), js_dim(dct), time=time)
    tar.close()
    return t,s


