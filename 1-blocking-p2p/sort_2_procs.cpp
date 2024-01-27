#include <iostream>
#include <cstring>
#include <mpi.h>
#include<algorithm>

using namespace std;

#define NUM_ELEMENTS 100 

void merge(int *a, int length_a, int *b, int length_b) {
    int *merged = (int *) malloc((length_a + length_b) * sizeof(*a));
    int a_i = 0, b_i = 0;
    for (int i = 0; i < length_a + length_b; i++) {
        if (a_i < length_a && b_i < length_b)
        {
            if (a[a_i] < b[b_i]) {
                merged[i] = a[a_i];
                a_i++;
            } else {
                merged[i] = b[b_i];
                b_i++;
            }
        } else {
            if (a_i < length_a) {
                merged[i] = a[a_i];
                a_i++;
            } else {
                merged[i] = b[b_i++];
            }
        } 
    }
    
    memcpy(a, merged, (length_a + length_b) * sizeof *merged);
    free(merged);
    return;
}

int main()
{
    int rank, size, data[NUM_ELEMENTS];
    MPI_Init(NULL, NULL);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2)
    {
        cout<<"Please run with exactly 2 ranks\n";
        MPI_Finalize();
        return 0;
    }

    int first_half = NUM_ELEMENTS / 2;
    int second_half = NUM_ELEMENTS - first_half;

    srand(0);

    if (rank == 0)
    {
        // prepare original data
        for (int i = 0; i < NUM_ELEMENTS; i++) {
            data[i] = rand() % NUM_ELEMENTS * 2;
        }
        
        // send the second half of the data to rank==1
        MPI_Send(&data[first_half], second_half, MPI_INT, 1, 0, MPI_COMM_WORLD);
        // sort the fisrt half
        sort(data, data+first_half);
        // recieve the sorted second half
        MPI_Recv(&data[first_half], second_half, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // merge the two half
        merge(data, first_half, &data[first_half], second_half);

        cout<<"sorted:\t\t";
        for (int i = 0; i < NUM_ELEMENTS; i++)
        {
            cout<<data[i]<<" ";
        }
    } else if (rank == 1) {
        // recieve the second half data
        MPI_Recv(&data[first_half], second_half, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        sort(&data[first_half], &data[first_half]+second_half);
        MPI_Send(&data[first_half], second_half, MPI_INT, 0, 0, MPI_COMM_WORLD); 
    }

    MPI_Finalize();
    return 0;
}
