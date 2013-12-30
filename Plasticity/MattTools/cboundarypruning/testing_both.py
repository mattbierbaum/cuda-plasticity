import boundarypruningutil
from Plasticity.TarFile import LoadTarState
#t,s = LoadTarState("/media/scratch/plasticity/lvp2d128_s0_d.tar")
t,s = LoadTarState("/media/scratch/plasticity/lvp2d1024_s0_d.tar")
#t,s = LoadTarState("/media/scratch/plasticity/lvp3d128_s4_d.tar")
rod = s.CalculateRotationRodrigues()
mis, grain, bd, ind = boundarypruningutil.Run(1024, 2, rod, "something", verbose=1, Image=True, J=3e-6, PtoA=1.5)
#mis, grain, bd, ind = boundarypruningutil.Run(128, 3, rod, "something", verbose=1, Image=True, J=8e-7, PtoA=1.2)
#import BoundaryPruningMethod
#BoundaryPruningMethod.Run(128, s, J=8e-05, PtoA=1.5, dir='./')
