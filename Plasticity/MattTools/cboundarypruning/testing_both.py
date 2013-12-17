import boundarypruningutil
from Plasticity.TarFile import LoadTarState
t,s = LoadTarState("/media/scratch/plasticity/lvp2d128_s0_d.tar")
rod = s.CalculateRotationRodrigues()
mis, grain, bd = boundarypruningutil.Run(128, 2, rod, "something", verbose=1, Image=True) #, J=1e-3, PtoA=8.6)
#import BoundaryPruningMethod
#BoundaryPruningMethod.Run(128, s, J=8e-05, PtoA=1.5, dir='./')
