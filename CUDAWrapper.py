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

def write_json_to_h(json, headername):
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


#===========================================================
# these are the interesting functions
def LoadState(file, time=None):
    filename, extenstion = os.path.splitext(file)
    tar = tarfile.open(file)
    dct = read_json(tar_getfile(tar, ".json"))
    t,s = FieldInitializer.LoadStateRaw(tar_getfile(tar, ".plas"), js_N(dct), js_dim(dct), time=time)
    tar.close()
    return t,s



