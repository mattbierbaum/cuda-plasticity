%module boundarypruning 
%{
#define SWIG_FILE_WITH_INIT
%}

%include "numpy.i"
%init %{
import_array();
%}

%{
#include "boundary_pruning_wrapper.h"
%}
%apply (int DIM1, double *IN_ARRAY1) {(int NN, double *omega)}
%apply (int DIM1, double *IN_ARRAY1) {(int N1, double *misorientations)}
%apply (int DIM1, int *IN_ARRAY1) {(int N2, int *grainsizes)}
%apply (int DIM1, int *IN_ARRAY1) {(int N3, int *bdlengths)}
%apply (int DIM1, int *IN_ARRAY1) {(int N4, int *indexmap)}
%typemap(in,numinputs=0) int *OutValue(int temp) {
    $1 = &temp;
}
%typemap(argout) int *OutValue {
    PyObject *o, *o2, *o3;
    o = PyInt_FromLong(*$1);
    if ((!$result) || ($result == Py_None)) {
        $result = o;
    } else {
        if (!PyTuple_Check($result)) {
            PyObject *o2 = $result;
            $result = PyTuple_New(1);
            PyTuple_SetItem($result,0,o2);
        }
        o3 = PyTuple_New(1);
        PyTuple_SetItem(o3,0,o);
        o2 = $result;
        $result = PySequence_Concat(o2,o3);
        Py_DECREF(o2);
        Py_DECREF(o3);
    }
}
%apply int *OutValue {int *grain_count}
%apply int *OutValue {int *bd_count}
%include "boundary_pruning_wrapper.h"

