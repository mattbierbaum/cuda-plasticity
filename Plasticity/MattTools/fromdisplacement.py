from Plasticity import TarFile
from Plasticity.Constants import *
from Plasticity.NumericalMethods import ME
from Plasticity.Fields import Fields
from Plasticity.GridArray.GridArray import GridArray
from Plasticity.FieldInitializers import FieldInitializer
from numpy.linalg import *
import numpy 
import itertools
cind = [x,y,z]
iind = [0,1,2] 

ux = numpy.fromfile("/media/scratch/plasticity/recon/ux/interval4_trial3_shifted.dat").reshape(16,16,16)
uy = numpy.fromfile("/media/scratch/plasticity/recon/uy/interval4_trial2_shifted.dat").reshape(16,16,16)
uz = numpy.fromfile("/media/scratch/plasticity/recon/uz/interval4_trial3_shifted.dat").reshape(16,16,16)

t,s = TarFile.LoadTarState("/media/scratch/plasticity/lvp3d16_s0_d.tar")
ufield = s.CalculateDisplacementField()
real_ux = ufield['x']
real_uy = ufield['y']
real_uz = ufield['z']
real_rho = s.CalculateRhoFourier()
beta = s.GetOrderParameterField()

N = ux.shape[0]
shape = tuple([i*1 for i in ux.shape])
state = FieldInitializer.GaussianRandomInitializer(shape)
u = Fields.VectorField(shape, kspace=False)

kv = state.sinktools.k
#ksq = state.ktools.kSq

u['x'] = GridArray(ux)
u['y'] = GridArray(uy)
u['z'] = GridArray(uz)
 
out = Fields.TensorField(shape, kspace=False)

uf = u.FFT()
rho = Fields.TensorField(state.gridShape, kspace=False)
for i in [x,y,z]:
    for j in [x,y,z]:
        out[i,j] = GridArray(1.j * kv[i] * uf[j]).irfftn()

state.UpdateOrderParameterField(out)
#    return state, u
