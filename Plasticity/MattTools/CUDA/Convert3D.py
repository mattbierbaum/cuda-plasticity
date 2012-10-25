import PlasticitySystem
import PlasticityState
import FieldInitializer
import Observer
import numpy

import sys
filename = sys.argv[1]

N = int(sys.argv[2])
ts = []
states = []
elemsize = 8
file = open("%s.plas" % filename, 'rb')
file.seek(0,2)
file_size = file.tell() / elemsize / (N*N*N*9+1)
print "Total %d time steps" % file_size
cnt = file_size

rec = Observer.RecordStateObserver("%s.save" % filename)

file.seek(0, 0)

for i in range(cnt):
    t = numpy.fromstring(file.read(elemsize), dtype='float64')
    data = numpy.fromstring(file.read(elemsize*N*N*N*9), dtype='float64').reshape(3,3,N,N,N)
    gridShape = tuple([N,N,N])
    state = PlasticityState.PlasticityState(gridShape)
    dict = {('x','x') : (0,0), ('x','y') : (0,1), ('x','z') : (0,2),\
            ('y','x') : (1,0), ('y','y') : (1,1), ('y','z') : (1,2),\
            ('z','x') : (2,0), ('z','y') : (2,1), ('z','z') : (2,2)}
    field = state.GetOrderParameterField()
    for component in field.components:
        field[component] = numpy.copy(data[dict[component]].transpose([2,1,0]))
    state = FieldInitializer.ReformatState(state)
    rec.Update(t, state)
    print "Process t = ", t
