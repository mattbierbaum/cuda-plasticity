import simplejson 
import os, re

def write_json_to_h(json):
    dct = simplejson.loads(json)
    headername = dct["EXTENDED_HEADER"]
    filename, extension = os.path.splitext(headername)
    
    with headerfile as open(headername, "w"):
        for key,val in dct.iteritems():
            headerfile.write("#define "+key+" "+val)

def read_h_to_json(headername):
    define_statement = re.compile("\#define\s([a-zA-Z0-9_]*)\s(.*)$")

    dct = {}
    headerfile = open(headername, "r")
    lines = headerfile.readlines()
    headerfile.close()

    for l in lines:
        var, val = define_statement.match(l.strip()).groups()   
        dct[var] = val

    return dct
