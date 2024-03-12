#include <iostream>
#include <vector>
#include <mpi.h>

constexpr int BLK_DIM = 2; // 假设块的维度是2

// 检查矩阵块是否全为0
bool is_zero_block(const std::vector<double>& mat, int offset, int mat_dim) {
    for (int i = 0; i < BLK_DIM; i++) {
        for (int j = 0; j < BLK_DIM; j++) {
            if (mat[offset + j + i * mat_dim] != 0.0)
                return false;
        }
    }
    return true;
}

// 矩阵块乘法并增加到C中
void dgemm_increment_c(const std::vector<double>& mat_a, const std::vector<double>& mat_b, std::vector<double>& mat_c, 
                       int offset_a, int offset_b, int offset_c, int mat_dim) {
    for (int i = 0; i < BLK_DIM; i++) {
        for (int j = 0; j < BLK_DIM; j++) {
            for (int k = 0; k < BLK_DIM; k++) {
                mat_c[offset_c + j + i * mat_dim] +=
                    mat_a[offset_a + k + i * mat_dim] * mat_b[offset_b + j + k * mat_dim];
            }
        }
    }
}

// 输出矩阵的函数
void print_matrix(const std::vector<double>& mat, int mat_dim, const std::string& name) {
    std::cout << "Matrix " << name << ":" << std::endl;
    for (int i = 0; i < mat_dim; ++i) {
        for (int j = 0; j < mat_dim; ++j) {
            std::cout << mat[i * mat_dim + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // 假设矩阵维度和块的数量是已知的
    int mat_dim = 4; // 整个矩阵的维度
    int blk_num = mat_dim / BLK_DIM; // 块的数量

    // 所有进程共享的矩阵A, B和C
    std::vector<double> mat_a(mat_dim * mat_dim, 0.0);
    std::vector<double> mat_b(mat_dim * mat_dim, 0.0);
    std::vector<double> mat_c(mat_dim * mat_dim, 0.0);
    std::vector<double> gt_c(mat_dim * mat_dim, 0.0);
    // 初始化矩阵A：使用递增的值
    double val = 1.0;
    for (int i = 0; i < mat_dim; ++i) {
        for (int j = 0; j < mat_dim; ++j) {
            mat_a[i * mat_dim + j] = val;
            val += 1.0;
        }
    }

    // 初始化矩阵B：使用递减的值
    val = static_cast<double>(mat_dim * mat_dim);
    for (int i = 0; i < mat_dim; ++i) {
        for (int j = 0; j < mat_dim; ++j) {
            mat_b[i * mat_dim + j] = val;
            val -= 1.0;
        }
    }

    // 根据rank计算每个进程负责的工作ID
    int work_id_len = blk_num * blk_num;
    for (int work_id = rank; work_id < work_id_len; work_id += nprocs) {
        int global_i = work_id / blk_num;
        int global_j = work_id % blk_num;
        
        for (int global_k = 0; global_k < blk_num; global_k++) {
            int offset_a = global_i * BLK_DIM * mat_dim + global_k * BLK_DIM;
            int offset_b = global_k * BLK_DIM * mat_dim + global_j * BLK_DIM;
            int offset_c = global_i * BLK_DIM * mat_dim + global_j * BLK_DIM;

            if (is_zero_block(mat_a, offset_a, mat_dim) || is_zero_block(mat_b, offset_b, mat_dim))
                continue;

            dgemm_increment_c(mat_a, mat_b, mat_c, offset_a, offset_b, offset_c, mat_dim);
        }
    }

    // 使用MPI_Allreduce直接在mat_c上进行原地操作
    MPI_Allreduce(MPI_IN_PLACE, mat_c.data(), mat_c.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < mat_dim; ++i) {
            for (int j = 0; j < mat_dim; ++j) {
                double sum = 0;
                for (int k = 0; k < mat_dim; ++k) {
                    sum += mat_a[i * mat_dim + k] * mat_b[k * mat_dim + j];
                }
                gt_c[i * mat_dim + j] = sum;
            }
        }
        print_matrix(mat_a, mat_dim, "A");
        print_matrix(mat_b, mat_dim, "B");
        print_matrix(mat_c, mat_dim, "C");
        print_matrix(gt_c, mat_dim, "GTC");
    }

    MPI_Finalize();
    return 0;
}
