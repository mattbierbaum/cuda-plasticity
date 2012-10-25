import simplejson

class Configuration(dict):
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

