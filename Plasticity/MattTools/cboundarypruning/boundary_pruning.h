#ifndef __BOUNDARY_PRUNING_H__
#define __BOUNDARY_PRUNING_H__

#include <vector>
#include <list>

#define MAX(x,y) ((x)>(y)? (x) : (y))
#define MIN(x,y) ((x)<(y)? (x) : (y))

using namespace std;

class Boundary {
    public:
        Boundary(class Cluster *cl0, class Cluster *cl1, int length, double inhomogeneity, double cost) : cluster0(cl0), cluster1(cl1), length(length), inhomogeneity(inhomogeneity), cost(cost) {
            if (cluster0 > cluster1) {
                swap(cluster0, cluster1);
            }
            ptoa = 0.0;
            id = 0;
        }

        ~Boundary() {
        }

        friend bool operator> (Boundary &b1, Boundary &b2);
        friend bool operator< (Boundary &b1, Boundary &b2);

        Cluster *get_cluster0() { return cluster0; }
        Cluster *get_cluster1() { return cluster1; }
    private:
        Cluster *cluster0;
        Cluster *cluster1;
    public:
        double cost;
        double ptoa;
        double inhomogeneity;
        int length;
        int id;
};

bool operator> (Boundary &b1, Boundary &b2)
        { return b1.cost > b2.cost;}
bool operator< (Boundary &b1, Boundary &b2)
        { return b1.cost < b2.cost;}

class Cluster {
    public:
        Cluster() {
            active=true;
        }
        /*Cluster(Cluster &copy){

        }*/
        ~Cluster() {

        }

    public:
        // FIXME
        // if we assume bd.cluster0 < bd.cluster1 (some sort of index)
        // and keep boundary_indices sorted on tuple (bd.cluster0, bd.cluster1)
        // it will make merge simple and fast. (for bd.cluster0 < this(=cluster)
        // bd.cluster0 is the other cluster, others, bd.cluster1 is the other
        // one, so it will be have a strict ordering of the other cluster)
        // We don't need searching on these, so a linked list will do?
        list<int> boundary_indices;
        vector<int> sites;
        double inhomogeneity;
        int perimeter;
        bool active;
        int id;

        double centerx, centery;
        double sigma;
};



#endif
