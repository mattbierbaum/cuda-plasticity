import tarfile
import simplejson
import os
from Plasticity.Configure import *
from Plasticity.FieldInitializers import FieldInitializer

#=====================================================
# small helper functions
def get_extension(filename):
    return os.path.splitext(filename)[1]

def open_if_not(filename, mode):
    if not isinstance(filename, str) or hasattr(filename, "read"):
        return filename
    else:
        return open(filename, mode)

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


#===========================================================
# these are the interesting functions
def LoadTarState(file, time=None):
    tar = tarfile.open(file)
    dct = read_json(tar_getfile(tar, ".conf"))
    t,s = FieldInitializer.LoadStateRaw(tar_getfile(tar, ".plas"), int(dct["N"]), int(dct["dim"]), time=time)
    tar.close()
    return t,s

def LoadTarJSON(file):
    tar = tarfile.open(file)
    dct = read_json(tar_getfile(tar, ".conf"))
    tar.close()
    return dct    

def LoadTarICS(file):
    tar = tarfile.open(file)
    dct = read_json(tar_getfile(tar, ".conf"))
    t,s = FieldInitializer.LoadStateRaw(tar_getfile(tar, ".ics"), int(dct["N"]), int(dct["dim"]), hastimes=False)
    tar.close()
    return t,s 

