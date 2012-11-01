        // This will consist of three parts
        // (1) cl < c0, c1 : compare bd0.c0 with bd1.c0
        // (2) c0 < cl < c1 : compare bd0.c1 with bd1.c0
        // (3) c0, c1 < cl : compare bd0.c1 with bd1.c1
        /*while(iter0 != end0 || iter1 != end1) {
            Boundary &bd0 = costs[*iter0], &bd1 = costs[*iter1];
            Cluster *cl0 = bd0.get_cluster0(), *cl1 = bd1.get_cluster0();
            if (cl0 == c0) cl0 = bd0.get_cluster1();
            if (cl1 == c1) cl1 = bd1.get_cluster1();
            if (cl0<cl1) {
                // put bd0 in
                costs.replace(*iter0, bd0);
                iter0++;
            } else if (cl0>cl1) {
                // put bd1 in, changing c1 to c0
                Boundary newbd1(c0, c1, length, inhomogeneity, cost);
                // Build modified boundary fIXME (change cl0, cl1 properly)
                costs.replace(*iter1, newbd1);
                // add to c0 boundary list (c1 boundary list will be deleted)
                iter0 = c0->boundary_indices.insert(iter0, *iter1);
                iter0++; // skip newly inserted element
                iter1++; 
            } else {
                // Merge boundaries
                Boundary mergedbd;
                // fIXME Actual Merge operation

                // Remove one boundary
                costs.remove(*iter1);
                costs.replace(*iter0, mergedbd);
            }
        }*/


         //clusters.erase(clusters.find(c1));
//        clusters.erase(std::remove(clusters.begin(), clusters.end(), *c1), clusters.end());
        //c1->set_inactive();




 /*int main(){
    double *b1, *b2;
    int *a, *b, *c;
    boundary_pruning(1,2,2,b1,2,b2,2,c,2,a,2,b);
}*/

//CHECKED ALL
/*
int main(){
    LocationAwareHeap<int> heap;
    int max = 128;//128*128*2;//1024*1024*2;

    struct timespec start;
    clock_gettime(CLOCK_REALTIME, &start);

    for (int i=0; i<max; i++)//max-1; i>=0; i--)  {
        heap.insert(i); 

    heap.print();

    int r1 = 5;
    int r3 = 9;
    heap.replace(2,r1);
    heap.replace(5,r3);
    heap.remove(3);
    heap.replace(1, r1);

    heap.print();

    int len = heap.getheap().size();
    for (int i=1; i<=len; i++){
        printf("%i ", heap.top());
        heap.pop();
    }
    printf("\n");

    struct timespec end;
    clock_gettime(CLOCK_REALTIME, &end);
    printf("time = %f\n", ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)/1e9));
 
    return 0;
}*/



    // build the reconsruction image
/*    int width  = 4;
    int bigsize  = (int)pow(width*n, dim);
    int *bigimap = new int[bigsize];

    int iiloops = (dim>0)?width:1;
    int jjloops = (dim>1)?width:1;
    int kkloops = (dim>2)?width:1;
   
    int big_n = n*width; 
    
    // first build the blown up images
    for (int i=0; i<iloops; i++)
    for (int j=0; j<jloops; j++) 
    for (int k=0; k<kloops; k++) {
        int tm = i+j*n+k*n*n;
        for (int ii=0; ii<iiloops; ii++)
        for (int jj=0; jj<jjloops; jj++)
        for (int kk=0; kk<kkloops; kk++){
            int big_tm = i*width+ii+(j*width+jj)*big_n+(k*width+kk)*big_n*big_n;
            bigimap[big_tm] = indexmap[tm];
            image[3*big_tm+0] = omega[3*tm+0];
            image[3*big_tm+1] = omega[3*tm+1];
            image[3*big_tm+2] = omega[3*tm+2];
        }
    }

    // now take directional gradients and fill in with lines where necessary
    for (int layer=0; layer<dim; layer++){
        for (int i=0; i<iloops*iiloops; i++){
        for (int j=0; j<jloops*jjloops; j++){
        for (int k=0; k<kloops*kkloops; k++){
            int index0 = i + j*big_n + k*big_n*big_n;
            int index1 = (i+(layer==0))%big_n      + 
                        ((j+(layer==1))%big_n)*big_n   + 
                        ((k+(layer==2))%big_n)*big_n*big_n;

            if (bigimap[index0] != bigimap[index1]){
                image[3*index0+0] = 0;
                image[3*index0+1] = 0;
                image[3*index0+2] = 0;
            }            
        } } }
    }

    delete [] indexmap;
    delete [] bigimap;*/

/*
int main(){
    LocationAwareHeap<int> heap;
    int max = 20;//128*128*2;//1024*1024*2;

    struct timespec start;
    clock_gettime(CLOCK_REALTIME, &start);

    for (int i=0; i<max; i++)//max-1; i>=0; i--)  {
        heap.insert(i); 

    heap.print();

    int r1 = 5;
    int r3 = 9;
    heap.replace(2,r1);
    heap.print();

    heap.replace(5,r3);
    heap.print();

    heap.remove(3);
    heap.print();

    heap.replace(1, r1);
  
    heap.print();

    int len = heap.getheap_size();
    for (int i=1; i<=len; i++){
        printf("%i ", heap.top());
        heap.pop();
    }
    printf("\n");

    struct timespec end;
    clock_gettime(CLOCK_REALTIME, &end);
    printf("time = %f\n", ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)/1e9));
 
    return 0;
}
*/
