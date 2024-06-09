#include <algorithm> 
#include <cstddef> 
#include <iostream>
#include <vector>
#include <mpi.h>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>

constexpr size_t BLK_DIM = 2;

// 检查本地矩阵是否全为零
// local_mat: 指向本地矩阵的指针
// 返回值: 如果矩阵全为零，则返回true；否则返回false
bool is_zero_local(const double* local_mat) {
    for (size_t i = 0; i < BLK_DIM; ++i) {
        for (size_t j = 0; j < BLK_DIM; ++j) {
            if (local_mat[j + i * BLK_DIM] != 0.0) {
                return false;
            }
        }
    }
    return true;
}

// 执行矩阵的局部双精度浮点数乘法
// local_a: 指向矩阵A的本地部分的指针
// local_b: 指向矩阵B的本地部分的指针
// local_c: 指向矩阵C的本地部分的指针（结果矩阵）
void dgemm(const double* local_a, const double* local_b, double* local_c) {
    std::fill_n(local_c, BLK_DIM * BLK_DIM, 0.0);

    for (size_t i = 0; i < BLK_DIM; ++i) {
        for (size_t j = 0; j < BLK_DIM; ++j) {
            for (size_t k = 0; k < BLK_DIM; ++k) {
                local_c[j + i * BLK_DIM] += local_a[k + i * BLK_DIM] * local_b[j + k * BLK_DIM];
            }
        }
    }
}

// 将全局矩阵的一部分打包到本地矩阵中
// local_mat: 指向本地矩阵的指针
// global_mat: 指向全局矩阵的指针
// mat_dim: 全局矩阵的维度
// global_i: 全局矩阵中子矩阵的行索引
// global_j: 全局矩阵中子矩阵的列索引
void pack_global_to_local(double* local_mat, const double* global_mat, size_t mat_dim, size_t global_i, size_t global_j) {
    size_t offset = global_i * BLK_DIM * mat_dim + global_j * BLK_DIM;

    for (size_t i = 0; i < BLK_DIM; ++i) {
        for (size_t j = 0; j < BLK_DIM; ++j) {
            local_mat[j + i * BLK_DIM] = global_mat[offset + j + i * mat_dim];
        }
    }
}

/*
    将本地矩阵的内容累加到全局矩阵中对应的部分
    global_mat: 指向全局矩阵的指针
    local_mat: 指向本地矩阵的指针
    mat_dim: 全局矩阵的维度
    global_i: 全局矩阵中子矩阵的行索引
    global_j: 全局矩阵中子矩阵的列索引
*/
void add_local_to_global(double* global_mat, const double* local_mat, size_t mat_dim, size_t global_i, size_t global_j) {
    size_t offset = global_i * BLK_DIM * mat_dim + global_j * BLK_DIM;

    for (size_t i = 0; i < BLK_DIM; ++i) {
        for (size_t j = 0; j < BLK_DIM; ++j) {
            global_mat[offset + j + i * mat_dim] += local_mat[j + i * BLK_DIM];
        }
    }
}

void init_mats(int mat_dim, double* mat_a, double* mat_b, double* mat_c) {
    // 设置随机数种子
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // 初始化矩阵A和B为随机值，初始化矩阵C为0
    for (int i = 0; i < mat_dim * mat_dim; ++i) {
        mat_a[i] = static_cast<double>(std::rand()) / RAND_MAX;
        mat_b[i] = static_cast<double>(std::rand()) / RAND_MAX;
        mat_c[i] = 0.0;
    }
}

void print_matrix(const double* mat, int dim) {
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            std::cout << mat[i * dim + j] << " ";
        }
        std::cout << std::endl;
    }
}

void check_mats(const double* mat_a, const double* mat_b, const double* mat_c, int mat_dim) {
    bool correct = true;
    double* mat_check = new double[mat_dim * mat_dim];

    // Compute the product of mat_a and mat_b
    for (int i = 0; i < mat_dim; ++i) {
        for (int j = 0; j < mat_dim; ++j) {
            double sum = 0.0;
            for (int k = 0; k < mat_dim; ++k) {
                sum += mat_a[i * mat_dim + k] * mat_b[k * mat_dim + j];
            }
            mat_check[i * mat_dim + j] = sum;
        }
    }

    // Compare mat_check with mat_c
    for (int i = 0; i < mat_dim * mat_dim; ++i) {
        if (std::fabs(mat_c[i] - mat_check[i]) > 1e-9) {
            correct = false;
            break;
        }
    }

    if (correct) {
        std::cout << "The result is correct." << std::endl;
    } else {
        std::cout << "The result is incorrect." << std::endl;
    }

    // Clean up
    delete[] mat_check;
}

int main(int argc, char **argv) {
    int rank, nprocs;
    int mat_dim = 4, blk_num;
    double *mat_a, *mat_b, *mat_c;
    double *local_a, *local_b, *local_c;

    double t1, t2;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (nprocs == 1) {
        if (rank == 0) {
            std::cout << "nprocs 必须大于1." << std::endl;
        }
        MPI_Finalize();
        exit(0);
    }

    blk_num = mat_dim / BLK_DIM;
    // 初始化矩阵
    if (rank == 0) {
        mat_a = new double[3 * mat_dim * mat_dim];
        mat_b = mat_a + mat_dim * mat_dim;
        mat_c = mat_b + mat_dim * mat_dim;
        init_mats(mat_dim, mat_a, mat_b, mat_c);
        // 打印矩阵A和B
        std::cout << "Matrix A:" << std::endl;
        print_matrix(mat_a, mat_dim);
        std::cout << "Matrix B:" << std::endl;
        print_matrix(mat_b, mat_dim);
    }

    // 分配本地缓冲区
    local_a = new double[BLK_DIM * BLK_DIM];
    local_b = new double[BLK_DIM * BLK_DIM];
    local_c = new double[BLK_DIM * BLK_DIM];


    // 同步所有进程
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();

    int work_id_len = blk_num * blk_num;

    // 主进程的工作
    if (rank == 0) {
        // ... [省略主进程分发A和B块的代码] ...

        for (int work_id = 0; work_id < work_id_len; work_id += nprocs - 1) {
            for (int i = 1; i < nprocs; ++i) {
                int current_work_id = work_id + i - 1;
                if (current_work_id < work_id_len) {
                    int global_i = current_work_id / blk_num;
                    int global_k = current_work_id % blk_num;
                    pack_global_to_local(local_a, mat_a, mat_dim, global_i, global_k);
                    MPI_Send(local_a, BLK_DIM * BLK_DIM, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                    for (int j = 0; j < blk_num; ++j) {
                        pack_global_to_local(local_b, mat_b, mat_dim, global_k, j);
                        MPI_Send(local_b, BLK_DIM * BLK_DIM, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                    }
                }
            }
            // 接收结果
            for (int i = 1; i < nprocs; ++i) {
                int current_work_id = work_id + i - 1;
                if (current_work_id < work_id_len) {
                    int global_i = current_work_id / blk_num;
                    for (int j = 0; j < blk_num; ++j) {
                        MPI_Recv(local_c, BLK_DIM * BLK_DIM, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        add_local_to_global(mat_c, local_c, mat_dim, global_i, j);
                    }
                }
            }
        }
    } else {
        // 工作进程的工作
        while (true) {
            MPI_Status status;
            MPI_Recv(local_a, BLK_DIM * BLK_DIM, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
            // 检查是否有结束信号
            if (status.MPI_TAG == 1) {
                break;
            }
            for (int j = 0; j < blk_num; ++j) {
                MPI_Recv(local_b, BLK_DIM * BLK_DIM, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (!is_zero_local(local_a) && !is_zero_local(local_b)) {
                    dgemm(local_a, local_b, local_c);
                } else {
                    std::fill_n(local_c, BLK_DIM * BLK_DIM, 0.0);
                }
                MPI_Send(local_c, BLK_DIM * BLK_DIM, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
        }
    }

    // 记录结束时间
    t2 = MPI_Wtime();

    // 同步所有进程
    MPI_Barrier(MPI_COMM_WORLD);

    // 主进程检查结果并打印时间
    if (rank == 0) {
        // check_mats函数用于检查结果的正确性
        check_mats(mat_a, mat_b, mat_c, mat_dim);
        std::cout << "[" << rank << "] 时间: " << (t2 - t1) << std::endl;
        // 打印矩阵C
        std::cout << "Matrix C (Result):" << std::endl;
        print_matrix(mat_c, mat_dim);
    }

    // 释放本地缓冲区
    delete[] local_a;
    // 主进程释放全局矩阵
    if (rank == 0) {
        delete[] mat_a;
    }


    MPI_Finalize();
    return 0;
}