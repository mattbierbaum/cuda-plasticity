#ifndef __BP_WRAPPER_H__ 
#define __BP_WRAPPER_H__

int boundary_pruning(int n, int dim,
                     int NN, double *omega,           
                     int N1, double *misorientations,
                     int N2, int    *grainsizes,
                     int N3, int    *bdlengths,
                     int N4, int    *indexmap,
                     double J, double pA, int verbose,
                     int *grain_count, int *bd_count);
#endif
