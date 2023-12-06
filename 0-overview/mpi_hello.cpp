#include <iostream>
#include <string.h>
#include <mpi.h>
using namespace std;

const int max_string = 100;

int main ()
{
	int comm_sz=0;
	int my_rank=0;
	char greeting[max_string];
	
	MPI_Init(NULL,NULL);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&comm_sz);

	 if(my_rank!=0){
		sprintf(greeting,"Greetings from process %d of %d!",my_rank,comm_sz);
		MPI_Send(greeting,strlen(greeting),MPI_CHAR,0,0,MPI_COMM_WORLD);
	}
	else{
		cout<<"Greetings from process "<<my_rank<<" of "<<comm_sz<<"!"<<endl;
		for(int i=1;i<comm_sz;i++){
			MPI_Recv(greeting,max_string,MPI_CHAR,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			cout<<greeting<<endl;
		}
	}
	
	MPI_Finalize ();
	return 0;
}