#include <math.h>
#include "boundary_pruning.h"

int boundary_pruning(int n, int dim,
                     int NN, double *omega,           
                     int N1, double *misorientations, 
                     int N2, int    *grainsizes,       
                     int N3, int    *bdlengths,       
                     int N4, int    *indexmap)
{
    //FIXME size error checking
    vector<Boundary> bds = costs.getheap();
    if (N2 <= clusters.size()) { printf("grainsizes not large enough\n"); return -1; }
    if (N1 <= bds.size())      { printf("misorientations not large enough\n"); return -1; }
    if (N3 <= bds.size())      { printf("bdlenghs not big enough\n"); return -1; }
    if (N4 != pow(n,dim))      { printf("indexmap incorrect dimensions\n"); return -1;}

    //hardest thing to do is make the indexmap, 
    //let's do it first, then do error checking
    int siteid = 0;
    for (int i=0; i<clusters.size(); i++){
        Cluster &c = clusters[i];
        int csize = c.sites.size();
    
        // add the grain size dist.
        grainsizes[siteid] = csize;

        // create the index map
        for (int j=0; j<csize; j++){
            int ind = 0;
            for (int k=0; k<dim; k++)
                ind += c.sites[j][k] * (int)pow(n, k);
            indexmap[ind] = siteid;
        }
        siteid++;
    }

    for (int i=0; i<bds.size(); i++){
        Boundary &b = bds[i];
        misorientations[i] = sqrt(fabs(b.inhomogeneity));
        bdlengths[i] = b.length;
    }

    return 0;
}
