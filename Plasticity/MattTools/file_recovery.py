from Plasticity.FieldInitializers import FieldInitializer
from Plasticity.Constants import *
import numpy
import shutil

def CheckRecovered(filename, N, dim):
    filer = filename+'.recover'
    tr, sr = FieldInitializer.LoadStateRaw(filer, N, dim)
    t, s = FieldInitializer.LoadStateRaw(filename, N, dim, time=tr)

    allclose = True
    for i in [x,y,z]:
        for j in [x,y,z]:
            allclose *= numpy.allclose(sr.betaP[i,j], s.betaP[i,j])
    return allclose

def RecoverPLASFile(filename, N, dim, dtype='float64', hastimes=True):
    """
    Recover a plas file from corruption due to file sync issues
    """
    if dtype.__class__ != numpy.dtype:
        dtype = numpy.dtype(dtype)
    elemsize = dtype.itemsize
    datasize = 9*(N**dim)
    
    file = open(filename, 'rb')
    fileo = open(filename+'.recover', 'w')

    file.seek(0,2)
    file_size = int(file.tell() / (elemsize*(datasize+1)))
    cnt = file_size
    file.seek(0,0)

    file.seek(0*(elemsize*(datasize+1)),0)
    tprev = numpy.fromstring(file.read(elemsize), dtype=dtype)
    dataprev = numpy.fromstring(file.read(elemsize*datasize), dtype=dtype)

    for i in range(1,cnt):
        file.seek(i*(elemsize*(datasize+1)),0)
        t = numpy.fromstring(file.read(elemsize), dtype=dtype)
        data = numpy.fromstring(file.read(elemsize*datasize), dtype=dtype)
        if t<tprev:
            break
        fileo.write(tprev.tostring())
        fileo.write(dataprev.tostring())
        tprev = t
        dataprev = data

    fileo.close()
    file.close()

    allclose = CheckRecovered(filename, N, dim)
    print "All close:", allclose

    shutil.move(filename, filename+".backup")
    shutil.move(filename+'.recover', filename)
