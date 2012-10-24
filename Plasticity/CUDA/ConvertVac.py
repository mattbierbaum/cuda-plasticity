import PlasticitySystem
import VacancyState
import PlasticityState
import FieldInitializer
import Observer
import numpy

import sys
filename = sys.argv[1]

N = int(sys.argv[2])
alpha = float(sys.argv[3])
ts = []
states = []
elemsize = 8
file = open("%s.plas" % filename, 'rb')
file.seek(0,2)
#file_size = file.tell() / elemsize / (N*N*N*10+1)
file_size = file.tell() / elemsize / (N*N*10+1)
print "Total %d time steps" % file_size
cnt = file_size

rec = Observer.RecordStateObserver("%s.save" % filename)

file.seek(0, 0)

for i in range(cnt):
    t = numpy.fromstring(file.read(elemsize), dtype='float64')
    #data = numpy.fromstring(file.read(elemsize*N*N*N*10), dtype='float64').reshape(10,N,N,N)
    data = numpy.fromstring(file.read(elemsize*N*N*10), dtype='float64').reshape(10,N,N)
    #gridShape = tuple([N,N,N])
    gridShape = tuple([N,N])
    state = VacancyState.VacancyState(gridShape, alpha=alpha)
    dict = {('x','x') : 0*3+0, ('x','y') : 0*3+1, ('x','z') : 0*3+2,\
            ('y','x') : 1*3+0, ('y','y') : 1*3+1, ('y','z') : 1*3+2,\
            ('z','x') : 2*3+0, ('z','y') : 2*3+1, ('z','z') : 2*3+2,\
            ('s','s') : 3*3+0}
    
    field = state.GetOrderParameterField()
    for component in field.components:
        #field[component] = numpy.copy(data[dict[component]].transpose([2,1,0]))
        field[component] = numpy.copy(data[dict[component]].transpose())
    state = FieldInitializer.ReformatState(state)
    rec.Update(t, state)
    print "Process t = ", t
