# include<iostream>
# include<mpi.h>

using namespace std;

double f(double x){
    return (4.0/(1.0+x*x));
}

int main(){
    int myid, num_process;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &num_process);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    int n=1000;
    double weight = 1 / (double)n, sum = 0, x;
    for(int i = myid+1; i <= n; i += num_process){
        x = weight * (i-0.5);
        sum += f(x);
    }
    double pi_myid, pi;
    pi_myid = weight * sum;
    MPI_Reduce(&pi_myid, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pi, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    cout<<"my process_id is "<<myid<<" of "<<num_process<<", the pi is "<<pi<<endl;
    MPI_Finalize();
}