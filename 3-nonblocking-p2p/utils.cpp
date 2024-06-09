#include "utils.h"
#include <algorithm> 
#include <cstdlib>   
#include <ctime>    
#include <iostream>
#include <cmath>


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

//计算比较结果正确性 
void check_mats(const double* mat_a, const double* mat_b, const double* mat_c, int mat_dim) {
    bool correct = true;
    double* mat_check = new double[mat_dim * mat_dim];

    for (int i = 0; i < mat_dim; ++i) {
        for (int j = 0; j < mat_dim; ++j) {
            double sum = 0.0;
            for (int k = 0; k < mat_dim; ++k) {
                sum += mat_a[i * mat_dim + k] * mat_b[k * mat_dim + j];
            }
            mat_check[i * mat_dim + j] = sum;
        }
    }

    
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

    // Clean
    delete[] mat_check;
}