import tarfile
import simplejson
import os


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


def tar_open(filename):
    return tarfile.open(filename)

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


#===========================================================
# these are the interesting functions
def LoadTarState(file, time=None):
    tar = tarfile.open(file)
    dct = read_json(tar_getfile(tar, ".json"))
    t,s = FieldInitializer.LoadStateRaw(tar_getfile(tar, ".plas"), int(dct["N"]), int(dct["dim"]), time=time)
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
    t,s = FieldInitializer.LoadStateRaw(tar_getfile(tar, ".ics"), int(dct["N"]), int(dct["dim"]), hastimes=False)
    tar.close()
    return t,s 

