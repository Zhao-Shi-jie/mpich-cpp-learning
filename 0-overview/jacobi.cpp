#include <iostream>
#include <mpi.h>

constexpr int steps = 10;
constexpr int totalsize = 16;
constexpr int mysize = 4;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // 获取当前进程的rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // 获取进程总数

    int left, right, tag1 = 3, tag2 = 4;
    float A[totalsize][mysize + 2], B[totalsize][mysize + 2];  // 声明两个二维数组A和B
    int begin_col, end_col;
    MPI_Status status;

    // 初始化A数组
    for (int i = 0; i < totalsize; ++i) {
        for (int j = 0; j < mysize + 2; ++j) {
            A[i][j] = 0.0;  // 将A数组初始化为0
        }
    }

    // 设置边界条件
    if (rank == 0) {
        for (int i = 0; i < totalsize; ++i) {
            A[i][1] = 8.0;  // rank为0的进程负责设置左边界
        }
    }

    if (rank == size - 1) {
        for (int i = 0; i < totalsize; ++i) {
            A[i][mysize] = 8.0;  // 最后一个进程负责设置右边界
        }
    }

    for (int i = 0; i < mysize + 2; ++i) {
        A[0][i] = 8.0;  // 所有进程设置上边界
        A[totalsize - 1][i] = 8.0;  // 所有进程设置下边界
    }

    for (int n = 0; n < steps; ++n) {
        left = (rank > 0) ? rank - 1 : MPI_PROC_NULL;  // 左边进程
        right = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;  // 右边进程

        MPI_Sendrecv(&A[1][mysize], totalsize, MPI_FLOAT, right, tag1,
                     &A[1][0], totalsize, MPI_FLOAT, left, tag1, MPI_COMM_WORLD, &status);  // 进程之间交换数据

        MPI_Sendrecv(&A[1][1], totalsize, MPI_FLOAT, left, tag2,
                     &A[1][mysize + 1], totalsize, MPI_FLOAT, right, tag2, MPI_COMM_WORLD, &status);  // 进程之间交换数据

        // 计算每个进程负责的列范围
        begin_col = (rank == 0) ? 2 : 1;
        end_col = (rank == size - 1) ? mysize - 1 : mysize;

        // 执行雅可比迭代的计算
        for (int j = begin_col; j <= end_col; ++j) {
            for (int i = 1; i < totalsize - 1; ++i) {
                B[i][j] = (A[i][j+1] + A[i][j-1] + A[i+1][j] + A[i-1][j]) * 0.25;
            }
        }

        // 将B数组的值更新到A数组
        for (int j = begin_col; j <= end_col; ++j) {
            for (int i = 1; i < totalsize - 1; ++i) {
                A[i][j] = B[i][j];
            }
        }
    }

    // 打印每个进程负责的部分结果
    for (int i = 1; i < totalsize - 1; ++i) {
        for (int j = 1; j < mysize + 1; ++j) {
            std::cout << A[i][j] << " ";
        }
        std::cout << std::endl;
    }

    MPI_Finalize();  // 结束MPI环境
    return 0;
}
