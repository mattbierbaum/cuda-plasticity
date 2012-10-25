import os
import simplejson
from subprocess import check_output

class Configuration(dict):
    def __init__(self, dc):
        super(Configuration, self).__init__(dc)

        curdir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        self["hash"] = check_output("git log -n 1 | grep commit | sed s/commit\ //", shell=True)[:10]
        os.chdir(curdir) 
        
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

def write_json(dct, filename):
    fi = open(filename, "w")
    fi.write(simplejson.dumps(dct))
    fi.close()

def read_json(jsonfile):
    return simplejson.loads(open_if_not(jsonfile, 'r').read())

def open_if_not(filename, mode):
    if not isinstance(filename, str) or hasattr(filename, "read"):
        return filename
    else:
        return open(filename, mode)

