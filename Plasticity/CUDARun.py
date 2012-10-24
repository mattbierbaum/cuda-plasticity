#!/usr/local/pub/enthought/bin/python
# shape and size of sim.
cudadir  = "/b/plasticity/cuda-plasticity/"
homedir  = "/b/plasticity/"
dim      = 3
previous = ""
postfix  = "r"
method   = "lvp"

import sys
sys.path.append(cudadir)
sys.path.append(cudadir+"Plasticity")
sys.path.append(cudadir+"Plasticity/Fields")
sys.path.append(cudadir+"Plasticity/GridArray")
sys.path.append(cudadir+"Plasticity/PlasticityStates")
sys.path.append(cudadir+"Plasticity/FieldInitializers")
import CUDAWrapper as wrap
from optparse import OptionParser

def main():
    parser = OptionParser()
    parser.add_option("-s", "--seed", dest="seed", action="store", type="int",
                      help="seed to use if generated RIC", default=0)
    parser.add_option("-d", "--device", dest="device", action="store", type="int",
                      help="the card to run on", default=0) 

    (options, args) = parser.parse_args()
    seed   = options.seed
    device = options.device

    wrap.simulation(homedir, cudadir, dim, previous, postfix, method, device=device, seed=seed)

if __name__ == "__main__":
    main()


