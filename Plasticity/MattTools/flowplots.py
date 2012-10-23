import PlasticitySystem, FieldInitializer
from PlottingToolsForMatt import *
from scipy import *
from pylab import *
import cPickle as pickle
import sys

rates = [0.0025, 0.005, 0.01, 0.015]
seeds = [11, 12, 13, 20, 21, 22, 23]
sizes = [32, 64]

ns, rs, strains, stresses = [], [], [], []

for N in sizes:
    print "N: ",N," ",
	
    for rate in rates:
	print rate," ",
	prefix  = "UNI_zz_S_"
	postfix = "_rate_"+str(rate).replace('.','_')+"_CU_2D"+str(N)+'_betaP.save'

	a, b  = StatisticsRuns(prefix, postfix, seeds, rate)
	pickle.dump(a, open("strain"+str(rate)+"_"+str(N)+".pickle", 'w'))
	pickle.dump(b, open("stress"+str(rate)+"_"+str(N)+".pickle", 'w'))

	ns.append(N)
	rs.append(rate)
	strains.append(a)
	stresses.append(b)
	
	sys.stdout.flush()

    print "\n"
