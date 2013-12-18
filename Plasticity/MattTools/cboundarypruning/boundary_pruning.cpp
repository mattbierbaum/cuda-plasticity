#include <vector>
#include <algorithm>
#include <time.h>
#include <math.h>
#include "boundary_pruning.h"
#include "boundary_pruning_wrapper.h"
#include "assist.h"

using namespace std;
void calculate_tension_joint(Cluster *c0, Cluster *c1, double &cx, double &cy, double &sigma);

template <typename T>
class LocationAwareHeap {
    // MAX heap
    public:
        // NOTE: changed to 0-based indexin
        LocationAwareHeap() {}
        ~LocationAwareHeap() {}

        int get_max_pos() {
            return pos_to_hpos.size();
        }

        int get_hpos(int pos) {
            return pos_to_hpos[pos];
        }
            
        int insert(T& element) {
            int pos = pos_to_hpos.size();
            int hpos = heap.size();

            pos_to_hpos.push_back(hpos);
            hpos_to_pos.push_back(pos);
            heap.push_back(element);

            upheap(hpos);
            return pos; 
        }

        void replace( int pos, T& element ) {
            int hpos = pos_to_hpos[pos];
            heap[hpos] = element;
            if (!downheap(hpos))
                upheap(hpos);
        }

        void remove_index( int hpos ) {
            swap(hpos, heap.size()-1);

            pos_to_hpos[hpos_to_pos[heap.size()-1]] = -1;
            heap.pop_back();
            hpos_to_pos.pop_back();

            // Reorder
            if (!downheap(hpos))
                upheap(hpos);
        }     

        void remove( int pos ) {
            int hpos = pos_to_hpos[pos];
            remove_index(hpos);             
        }

        T& operator[] ( const int pos ) {
            return heap[pos_to_hpos[pos]];
        }

        bool empty() {
            return heap.size() < 1;
        }

        T& top() {
            //FIXME
            //if (heap.size < 2) { throw ; } 
            return heap[0];
        }

        int top_index() {
            return hpos_to_pos[0];
        }

        void pop() {
           remove_index(0);
        }

        void swap(int hpos_a, int hpos_b) {
            T tmp   = heap[hpos_a];
            heap[hpos_a] = heap[hpos_b];
            heap[hpos_b] = tmp;
            
            int e   = hpos_to_pos[hpos_a];
            hpos_to_pos[hpos_a] = hpos_to_pos[hpos_b];
            hpos_to_pos[hpos_b] = e;

            pos_to_hpos[hpos_to_pos[hpos_a]] = hpos_a;
            pos_to_hpos[hpos_to_pos[hpos_b]] = hpos_b;
        }

        T& getheap_element(int i){
            return heap[i];
        }

        int getheap_size() {
            return heap.size();
        }

        void print(){
            printf("\n");
            for (int i=0; i<heap.size(); i++)
                printf("%02d ", heap[i]);//.cost);
            printf("\n");
            for (int i=0; i<heap.size(); i++)
                printf("%02d ", hpos_to_pos[i]);
            printf("\n");
            for (int i=0; i<pos_to_hpos.size(); i++)
                printf("%02d ", pos_to_hpos[i]);
            printf("\n");
        }

    private:
        bool downheap(int hpos) {
            bool flag = false;
            while(1) {
                bool comp_left  = (2*hpos+1<heap.size());
                if (comp_left) comp_left = heap[hpos] < heap[2*hpos+1];
                bool comp_right = (2*hpos+2<heap.size());
                if (comp_right) comp_right = heap[hpos] < heap[2*hpos+2];
                if (comp_left && comp_right) {
                    // Pick bigger one to bring up
                    bool comp_left_right = heap[2*hpos+1] < heap[2*hpos+2];
                    if (comp_left_right)
                        comp_left = false;
                    else
                        comp_right = false;
                }
                if (comp_left) {
                    // smaller than child
                    // pull up the left one
                    swap(hpos, 2*hpos+1);
                    hpos = 2*hpos+1;
                    flag = true;
                }
                else if (comp_right) {
                    // smaller than child
                    swap(hpos, 2*hpos+2);
                    hpos = 2*hpos+2;
                    flag = true;
                } else
                    break;
            }
            return flag;
        }

        void upheap(int hpos) {
            // Reorder
            while(1) {
                bool comp_parent = (hpos > 0);
                if (comp_parent) comp_parent = heap[hpos] > heap[(hpos-1)>>1];
                if (comp_parent) {
                    // parent smaller than child
                    swap(hpos, (hpos-1)>>1);
                    hpos = (hpos-1)>>1;
                } else
                    break;
            }
        }

    private:
        vector<T> heap;
        vector<int> pos_to_hpos;
        vector<int> hpos_to_pos;
};

// get the opposite cluster of the tuple of clusters
inline Cluster* cocluster(Boundary &bd, Cluster *c){
    // check validity -- FIXME this breaks things, where it shouldn't
#if 0
    if (bd.get_cluster0() != c && bd.get_cluster1() != c) {
        // ERROR!!!
        printf("ERROR %x %x != %x!!\n", bd.get_cluster0(), bd.get_cluster1(), c);
        float temp = 1/0;
        return NULL;
    }
#endif
    if (bd.get_cluster0() == c)
        return bd.get_cluster1();
    return bd.get_cluster0();
}


// FIXME - if we can avoid using globals
// Quick and dirty way of comparing the clusters by cluster pointer
int global_n = 0;
class LocationAwareHeap<Boundary> *global_costs;
class vector<Cluster> *global_clusters;

bool comp_bd_inhomogeneity( int ibd1, int ibd2 ) {
    Boundary &bd1 = (*global_costs)[ibd1];
    Boundary &bd2 = (*global_costs)[ibd2];
    return (bd1.inhomogeneity < bd2.inhomogeneity);
}

bool comp_bd_clusters( int ibd1, int ibd2 ) {
    Boundary &bd1 = (*global_costs)[ibd1];
    Boundary &bd2 = (*global_costs)[ibd2];
    return (bd1.get_cluster0() <bd2.get_cluster0()) || 
          ((bd1.get_cluster0()==bd2.get_cluster0()) && 
           (bd1.get_cluster1() < bd2.get_cluster1()));
}

bool is_sorted( list<int> &l ) {
    list<int>::iterator iter = l.begin();
    if (iter==l.end())
        return true;
    int pv = *iter;
    iter++;
    while(iter != l.end()) {
        if (comp_bd_clusters(*iter, pv)) {
            return false;
        }
        pv = *iter;
        iter++;
    }
    return true;
}

typedef void(*SetCostFunc)(Boundary &);

void InhomogeneityCost(Boundary &bd) {
    bd.cost = -bd.inhomogeneity;//bd.length;
}

void PerimeterVsAreaCost(Boundary &bd) {
    Cluster *c0 = bd.get_cluster0();
    Cluster *c1 = bd.get_cluster1();

    //double s0 = c0->sites.size();
    //double s1 = c1->sites.size();

    //double j0, j1, j2;
    //calculate_tension_joint(c0, c1, j0, j1, j2);

    //double dj = 0.0;

    // NEW METHOD 1
    //double dj = sqrt(j0*j0 + j1*j1) - (sqrt(c0->centerx*c0->centerx + c0->centery*c0->centery) 
    //                                 + sqrt(c1->centerx*c1->centerx + c1->centery*c1->centery));

    // NEW METHOD 2
    //Cluster *big = (c0->perimeter*c0->sites.size()/c0->sigma > c1->perimeter*c1->sites.size()/c1->sigma) ? c0 : c1;
    //Cluster *big = (c0->sites.size()> c1->sites.size()) ? c0 : c1; // best so far
    //Cluster *big = (c0->perimeter > c1->perimeter) ? c0 : c1;
    //dj = sqrt((j0-big->centerx)*(j0-big->centerx) + (j1-big->centery)*(j1-big->centery)) / bd.length;

    // NEW METHOD 3
    //double dj0 = c0->sites.size() * sqrt((j0-c0->centerx)*(j0-c0->centerx) + (j1-c0->centery)*(j1-c0->centery));
    //double dj1 = c1->sites.size() * sqrt((j0-c1->centerx)*(j0-c1->centerx) + (j1-c1->centery)*(j1-c1->centery));
    //double dj = (dj0 + dj1) / (c0->sites.size() + c1->sites.size());

    // NEW METHOD 4
    //double r0 = (double)s0 / (s0+s1);
    //double r1 = (double)s1 / (s0+s1);
    //dj = sqrt((s0*s0*((c0->centerx - j0)*(c0->centerx - j0)+(c0->centery-j1)*(c0->centery-j1))
    //                + s1*s1*((c1->centerx - j0)*(c1->centerx - j0)+(c1->centery-j1)*(c1->centery-j1)))/((s0+s1)*(s0+s1))) * 1e-4;
    //dj = sqrt((r0*((c0->centerx - j0)*(c0->centerx - j0)+(c0->centery-j1)*(c0->centery-j1))
    //                + r1*((c1->centerx - j0)*(c1->centerx - j0)+(c1->centery-j1)*(c1->centery-j1))))/(s0+s1)/(s0+s1);

    double pa0 = c0->perimeter / (double)c0->sites.size();
    double pa1 = c1->perimeter / (double)c1->sites.size();

    // p/a values are smaller than 4, so this creates a
    // 'lexicographic' ordering
    //bd.cost = pa0*4 + pa1;

#define TUPLE_PARAMETER 32768
    bd.ptoa = MAX(pa0, pa1);
    bd.cost = bd.ptoa*TUPLE_PARAMETER + 2*bd.length - bd.inhomogeneity;//+ dj;
}

void calculate_tension_joint(Cluster *c0, Cluster *c1, double &cx, double &cy, double &sigma){
    int count = 0;
    cx = 0.0; cy = 0.0; sigma = 0.0;
    vector<int>::iterator i0 = c0->sites.begin();
    vector<int>::iterator e0 = c0->sites.end();

    while (i0 != e0){
        cx += *i0 % global_n;
        cy += *i0 / global_n;
        count++;
        i0++;
    }

    i0 = c1->sites.begin();
    e0 = c1->sites.end();

    while (i0 != e0){
        cx += *i0 % global_n;
        cy += *i0 / global_n;
        count++;
        i0++;
    }
    cx /= count;
    cy /= count;

    i0 = c0->sites.begin();
    e0 = c0->sites.end();
    while (i0 != e0){
        sigma += (*i0 % global_n - cx)*(*i0 % global_n - cx);
        sigma += (*i0 / global_n - cx)*(*i0 / global_n - cx);
        i0++;
    }

    i0 = c1->sites.begin();
    e0 = c1->sites.end();
    while (i0 != e0){
        sigma += (*i0 % global_n - cx)*(*i0 % global_n - cx);
        sigma += (*i0 / global_n - cx)*(*i0 / global_n - cx);
        i0++;
    }

    sigma = sqrt(sigma / (count - 1));
}

int stopthisshit = 0;
double T = 1e100;
void delete_boundary(Boundary &bd, int bd_index, LocationAwareHeap<Boundary> &costs, SetCostFunc SetBoundaryCost, bool LocalCostFunc) {
    // Implement merging of clusters
    Cluster *c0 = bd.get_cluster0();
    Cluster *c1 = bd.get_cluster1();

    //double Ss = c0->sites.size() * log( c0->sites.size() ) + c1->sites.size() * log( c1->sites.size() );
    //double Se = (c0->sites.size() + c1->sites.size()) * log( c0->sites.size() + c1->sites.size() );

    //double Es = bd.inhomogeneity;

    // If c0 > c1, swap for simplicity
    //if (c0 > c1)
    // FIXME - this is potentially faster
    if (c0->sites.size() < c1->sites.size())
        swap(c0, c1);
    c0->sites.insert(c0->sites.end(), c1->sites.begin(), c1->sites.end());
    c0->inhomogeneity += c1->inhomogeneity - bd.inhomogeneity;
    c0->perimeter += c1->perimeter - 2*bd.length;

    //calculate_tension_joint(c0, c1, c0->centerx, c0->centery, c0->sigma);

    // Treat boundaries
    list<int>::iterator iter0 = c0->boundary_indices.begin();
    list<int>::iterator end0 = c0->boundary_indices.end();
    list<int>::iterator iter1 = c1->boundary_indices.begin();
    list<int>::iterator end1 = c1->boundary_indices.end();

    // Run while either list is not empty
    while ((iter1 != end1) || (iter0 != end0)) {
        Cluster *tc0 = NULL;
        Cluster *tc1 = NULL;
        if (iter0 != end0) {
            //Skip the boundary that's being removed - woosong
            if (*iter0 == bd_index) {
                iter0 = c0->boundary_indices.erase(iter0);
                end0 = c0->boundary_indices.end();
                continue;
            }
            Boundary &b0 = costs[*iter0];
            tc0 = cocluster(b0, c0);
        }
        if (iter1 != end1) {
            //Skip the boundary that's being removed - woosong
            if (*iter1 == bd_index) {
                iter1 = c1->boundary_indices.erase(iter1);
                end1 = c1->boundary_indices.end();
                continue;
            }
            Boundary &b1 = costs[*iter1];
            tc1 = cocluster(b1, c1);
        }

        if ((tc0 != NULL && tc0 < tc1) || (tc1 == NULL)) {
            // Cost may change for p/a clustering
            Boundary &b0 = costs[*iter0];
            SetBoundaryCost(b0);
            if (!LocalCostFunc) 
                costs.replace(*iter0, b0);
            // add tc0 if tc0 < tc1 or tc1 ended
            iter0++;
        } else if (tc0 == NULL || tc1 < tc0) {
            // add tc1 to iter0 if tc1 < tc0 or tc0 ended
            Boundary &b1 = costs[*iter1];

            // the boundary gets transferred over
            Boundary newbd1(c0, tc1, b1.length, b1.inhomogeneity, b1.cost);
            newbd1.id = b1.id;

            // Cost may change for p/a clustering
            SetBoundaryCost(newbd1);

            b1 = newbd1;

            // Build modified boundary fIXME (change cl0, cl1 properly)
            costs.replace(*iter1, newbd1);

            // add to c0 boundary list (c1 boundary list will be deleted)
            c0->boundary_indices.insert(iter0, *iter1);
            end0 = c0->boundary_indices.end();

            // This needs to go in the correct position for tc1
            // since cluster pointer is changed.
            tc1->boundary_indices.sort( comp_bd_clusters );
            iter1++;
        }
        else {
            // merge !
            // this only happens when tc0==tc1 and both tc0 and tc1
            // not NULL
            Boundary &b0 = costs[*iter0];
            Boundary &b1 = costs[*iter1];

            int len = b0.length + b1.length;
            double imh = (b0.inhomogeneity*b0.length + b1.inhomogeneity*b1.length) / (b0.length + b1.length);

            // tc0 == tc1
            // boundary between (c0 || c1) and (tc0 || tc1)
            Boundary newbd0 = Boundary(c0, tc0, len, imh, -imh);

            if (!LocalCostFunc)
                newbd0.length = b0.length;

            SetBoundaryCost(newbd0);

            if (!LocalCostFunc)
                newbd0.length = b0.length + b1.length;

            // put the new one in place of the original
            costs.replace(*iter0, newbd0); 

            // remove the old boundary
            costs.remove(*iter1);

            // need to remove one from the bounday index list of tc0/tc1
            tc0->boundary_indices.remove(*iter1);

            iter0++;
            iter1++;
        }
    }
    //double Ee = bd.inhomogeneity;
    //if (Ee-Es < T*(Se-Ss))
    //    stopthisshit = 1;

    c1->boundary_indices.clear();
    c1->sites.clear();
    c1->active = false;
}


typedef int (*printptr) (const char *str, ...);
int myprint(const char *str, ...){ return 0;}

//================================================================
//===============================================================
int boundary_pruning(int n, int dim,
                     int NN, double *omega,           
                     int N1, double *misorientations, 
                     int N2, int *grainsizes,       
                     int N3, int *bdlengths,       
                     int N4, int *indexmap,
                     double J, double pA, int verbose,
                     int *grain_count, int *bd_count){
    int size = int(pow(n, dim));
    global_n = n;

    class LocationAwareHeap<Boundary> costs;
    vector<Cluster> clusters(size); 
    global_costs = &costs;
    global_clusters = &clusters;

    printptr print;
    if (verbose == 0)
        print = myprint;
    else
        print = printf;

    // initialize all clusters and boundaries
    // create all boundaries for 1->3 dimensions 
    print("Initializing %iD...\n", dim);
    for (int i=0; i<size; i++) {
        Cluster &cluster = clusters[i];
        cluster.sites.push_back(i);
        cluster.perimeter = 2*dim;
        cluster.inhomogeneity = 0.;
        cluster.id = i;

        if (dim == 2){
            cluster.centerx = i%n;
            cluster.centery = i/n;
            cluster.sigma = 0.0;
        }
    }

    int iloops = (dim>0)?n:1;
    int jloops = (dim>1)?n:1;
    int kloops = (dim>2)?n:1;

    for (int layer=0; layer<dim; layer++) {
        for (int i=0; i<iloops; i++){ 
        for (int j=0; j<jloops; j++){
        for (int k=0; k<kloops; k++){
            int index0 = i + j*n + k*n*n;
            int index1 = (i+(layer==0))%n      + 
                        ((j+(layer==1))%n)*n   + 
                        ((k+(layer==2))%n)*n*n;

            Cluster *c0 = &clusters[index0];
            Cluster *c1 = &clusters[index1];

            double *omega0 = omega+(3*index0);
            double *omega1 = omega+(3*index1);
            double inhomogeneity = pow(*(omega1+0)- *(omega0+0), 2) +
                                   pow(*(omega1+1)- *(omega0+1), 2) + 
                                   pow(*(omega1+2)- *(omega0+2), 2);

            //c0->inhomogeneity = inhomogeneity;

            // Enforce that c0 < c1
            if (c0>c1) { swap(c0, c1); }
            Boundary bd(c0, c1, 1, inhomogeneity, -inhomogeneity);
            bd.id = index0 + size*index1;

            int index = costs.insert(bd);
            c0->boundary_indices.push_back(index);
            c1->boundary_indices.push_back(index);
        } } }
    }

    print("Sorting...\n");
    int count = 0;
    // Put boundaries in order for each cluster
    for(vector<Cluster>::iterator iter = clusters.begin(); iter != clusters.end(); iter++) {
        (*iter).boundary_indices.sort( comp_bd_clusters );
#if 1
        if (!is_sorted((*iter).boundary_indices))
            print("ERROR!!\n");
#endif
    }


    //===============================================================
    // Loop
    //===============================================================
    int step = 0;
    double tcost = costs.top().cost;
    print("Minimizing boundaries...\n");
    while(costs.getheap_size() > 1)
    {
        step++;
        if (step % 50000 == 0)
            print("%e %i\n", tcost, costs.getheap_size());

        // Can't get a reference and pop immediately!!
        // Changed to non-reference - woosong
        Boundary bd = costs.top();
        tcost = bd.cost;

        // this is actually the end condition, we reached our
        // misorientation goal
        if (tcost < -J) break;

        // get the highest energy boundary
        int bd_index = costs.top_index();

        // merge the clusters involved and remove the boundary 
        delete_boundary(bd, bd_index, costs, InhomogeneityCost, true);

        // pop from the costs after all's finished
        costs.remove(bd_index);
    }

    print("Perimeter/Area pruning...\n");
    // First recalcualte costs of all boundaries
    int maxpos = costs.get_max_pos();
    for(int i=0; i<maxpos; i++) {
        if (costs.get_hpos(i) >= 0) {
            // If it still exists
            Boundary &bd = costs[i];
            PerimeterVsAreaCost(bd);
            costs.replace(i, bd);
        }
    }

    double tptoa = 0;

    // Now do pruning
    while(costs.getheap_size() > 1)
    {
        step++;
        if (step % 50000 == 0)
            print("%e %i\n", tptoa, costs.getheap_size());

        // get the highest energy boundary
        Boundary bd = costs.top();
        tcost = bd.cost;
        tptoa = bd.ptoa;

        // our break condition is checked in time
        if (bd.ptoa < pA) break;
        //if (stopthisshit) break;
        //if (tcost < pA*TUPLE_PARAMETER) break;

        int bd_index = costs.top_index();
        delete_boundary(bd, bd_index, costs, PerimeterVsAreaCost, false);
        costs.remove(bd_index);
    }
    print("Saving...\n"); 

    //=================================================================
    // save it now
    //=================================================================
    //FIXME size error checking
    if (N2 <= clusters.size()) 
    { print("grainsizes not large enough\n"); return -1; }
    
    if (N1 <= costs.getheap_size())      
    { print("misorientations not large enough\n"); return -1; }
    
    if (N3 <= costs.getheap_size())      
    { print("bdlenghs not big enough\n"); return -1; }

    if (N4 != pow(n,dim))      
    { print("indexmap incorrect dimensions\n"); return -1;}

    //make the indexmap for the python 
    //code to reinterpret as an image
    int siteid = 0;

    for (int i=0; i<clusters.size(); i++){
        Cluster &c = clusters[i];
        if (c.active){
            int csize = c.sites.size();
    
            // add the grain size dist.
            grainsizes[siteid] = csize;

            // create the index pos_to_hpos
            for (int j=0; j<csize; j++){
                int ind = c.sites[j];
                indexmap[ind] = siteid;
            }
            siteid++;
        }
    }
    *grain_count = siteid;

    // save all of the distributions
    int i;
    for (i=0; i<costs.getheap_size(); i++){
        Boundary &b = costs.getheap_element(i);
        misorientations[i] = sqrt(fabs(b.inhomogeneity/b.length));
        bdlengths[i] = b.length;
    }
    *bd_count = i;

    return 0;
}


///=====================================================
// main
//======================================================
int main(){
    char file[] = "lena.dat";
    int n = 128;
    int d = 2;
    int s = (int)pow(n,d);
    int f = 2;//128;
    int N0=s*3, N1=s*f, N2=s*f, N3=s*f, N4=s;

    //double omega[] = {0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,1,0,0,0,1,0,0,0,1};//ReadDoubleMatrixFile(file, n, n, 0, 1); 
    double *omega = ReadDoubleMatrixFile(file, 3*n, n, 0, 1); 
    double *misorientations = new double[N1];
    int    *grainsizes      = new int[N2];
    int    *bdlengths       = new int[N3];
    int    *indexmap        = new int[N4];
    int grain_count, bd_count;

    boundary_pruning(n,  d,
                     N0, omega,           
                     N1, misorientations, 
                     N2, grainsizes,       
                     N3, bdlengths,       
                     N4, indexmap,
                     180.0, 1.5, 1, &grain_count, &bd_count);

    //free(omega);
    delete [] misorientations;
    delete [] grainsizes;
    delete [] bdlengths;
    delete [] indexmap; 
    return 0;
}


