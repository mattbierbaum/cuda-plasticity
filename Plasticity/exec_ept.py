from optparse import OptionParser
import Plasticity.RunCUDA as run

dct = {"cudadir": "/home/m-bierbaum/projects/cuda-plasticity/", 
       "homedir": "/media/scratch/tmp/",
       "N":       256,
       "dim":      2, 
       "previous": "", 
       "postfix":  "r", 
       "method":   "lvp"}

def main():
    parser = OptionParser()
    parser.add_option("-s", "--seed", dest="seed", action="store", type="int",
                      help="seed to use if generated RIC", default=0)
    parser.add_option("-d", "--device", dest="device", action="store", type="int",
                      help="the card to run on", default=0) 

    (options, args) = parser.parse_args()
    seed   = options.seed
    device = options.device

    dct.update({"device": options.device})
    dct.update({"seed":   options.seed})
    run.simulation(dct)

if __name__ == "__main__":
    main()


