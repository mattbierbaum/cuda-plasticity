import simplejson 
import tarfile
import os, re

#=====================================================
# small helper functions
def array_to_carray(arr):
    return repr(arr).replace("array", "").replace("(","").replace(")","").replace("[","{").replace("]","}").replace("\n","").replace(" ","") 

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
        headerfile.write("#define "+key+" "+val)

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
    return simplejson.loads(open_if_not(jsonfile))

def js_N(dct):
    return int(dct["N"])

def js_dim(dct):
    if "DIMENSION3" in dct.keys():
        return 3
    return 2

def tar_getfile(tar, extension):
    if not isinstance(tar, tarfile.TarFile):
        tar = tarfile.open(tar)
    for mem in tar.getmembers():
        if mem.isfile() and get_extension(mem.name) == extension:
            return tar.extractfile(mem)


#==========================================================
# creates json configurations for simulations
class CUDAConfiguration(object):
    def __init__(N, dim):
        self.dct = {"N": N}
        self.dct.update({"CFLsafeFactor": 0.5})
        if dim == 3:
            self.dct.update({"DIMENSION3":""})

    def load(direction, rate, start=0.0):
        self.dct.update({"LOADING": ""})
        self.dct.update({"LOAD_DEF": array_to_carray(direction)})
        self.dct.update({"LOADING_RATE": rate})
        self.dct.update({"LOAD_START": start})

    def dynamics(dyn):
        if dyn == "gcd":
            self.dct.update({"lambda": 0})
        if dyn == "mdp":
            self.dct.update({"lambda": 1})
        if dyn == "lvp":
            self.dct.update({"lambda": 0})
            self.dct.update({"NEWGLIDEONLY", ""})
       
    def dynamic_nucleation(dn=True):
        if dn == True:
            self.dct.update({"DYNAMIC_NUCLEATION", ""})



#===========================================================
# these are the interesting functions
def LoadState(file, time=None):
    filename, extenstion = os.path.splitext(file)
    tar = tarfile.open(file)
    dct = read_json(tar_getfile(tar, ".json"))
    t,s = FieldInitializer.LoadStateRaw(tar_getfile(tar, ".plas"), js_N(dct), js_dim(dct), time=time)
    tar.close()
    return t,s


