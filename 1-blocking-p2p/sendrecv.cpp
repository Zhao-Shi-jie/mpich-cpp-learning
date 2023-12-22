# include <mpi.h>
# include <iostream>
using namespace std;

int main()
{
    int rank, data[3];
    
    MPI_Init(NULL, NULL);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank == 0) {
	 data[0] = 0;
	 data[1] = 10;
	 data[2] = 20;
	 MPI_Send(data, 3, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else if(rank == 1) {
	 data[0] = -10;
	 data[1] = -20;
	 data[2] = -30;
	 cout<<"Before receving, data: "<<data[0]<<" "<<data[1]<<" "<<data[2]<<endl;
	 MPI_Recv(data, 3, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	 cout<<"After receving, data: "<<data[0]<<" "<<data[1]<<" "<<data[2];
    }
    
    MPI_Finalize();
    return 0;
}    
