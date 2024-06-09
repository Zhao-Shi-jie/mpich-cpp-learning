#ifndef UTILS_H
#define UTILS_H

#include <cstddef>

constexpr size_t BLK_DIM = 2;

// 检查本地矩阵是否全为零
bool is_zero_local(const double* local_mat);

// 执行矩阵的局部双精度浮点数乘法
void dgemm(const double* local_a, const double* local_b, double* local_c);

// 将全局矩阵的一部分打包到本地矩阵中
void pack_global_to_local(double* local_mat, const double* global_mat, size_t mat_dim, size_t global_i, size_t global_j);

// 将本地矩阵的内容累加到全局矩阵中对应的部分
void add_local_to_global(double* global_mat, const double* local_mat, size_t mat_dim, size_t global_i, size_t global_j);

// 初始化矩阵
void init_mats(int mat_dim, double* mat_a, double* mat_b, double* mat_c);

// 打印矩阵
void print_matrix(const double* mat, int dim);

// 检查矩阵计算结果的正确性
void check_mats(const double* mat_a, const double* mat_b, const double* mat_c, int mat_dim);

#endif // UTILS_H