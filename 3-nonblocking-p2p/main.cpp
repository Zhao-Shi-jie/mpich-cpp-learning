#include <iostream>
#include <mpi.h>
#include "utils.h"   // for utility functions and BLK_DIM

// 函数声明
void smmb(int argc, char **argv);
void smm(int argc, char **argv);

int main(int argc, char **argv) {
    double t1, t2;

    // 初始化MPI环境
    MPI_Init(&argc, &argv);

    // 同步所有进程
    MPI_Barrier(MPI_COMM_WORLD);

    


    // 主进程打印时间
    // int rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // if (rank == 0) {
    //     std::cout << "[" << rank << "] 阻塞实现时间: " << (t2 - t1) << std::endl;
    // }



    // 执行非阻塞实现
    smm(argc, argv);
    // 执行阻塞实现
    smmb(argc, argv);

    // 主进程打印时间
    // if (rank == 0) {
    //     std::cout << "[" << rank << "] 非阻塞实现时间: " << (t2 - t1) << std::endl;
    // }

    return 0;
}