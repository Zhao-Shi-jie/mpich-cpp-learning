#include <algorithm> 
#include <cstddef>  
#include <iostream> 
#include <mpi.h>  
#include <cstdlib> 
#include <ctime> 
#include <cmath> 
#include "utils.h"   // for utility functions and BLK_DIM

// TODO：优化非阻塞实现

void smm(int argc, char **argv) {
    int rank, nprocs;
    int mat_dim = 64, blk_num;
    double *mat_a, *mat_b, *mat_c;
    double *local_a, *local_b, *local_c;

    double t1, t2;

    // 初始化MPI环境
    // MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (nprocs == 1) {
        if (rank == 0) {
            std::cout << "nprocs 必须大于1." << std::endl;
        }
        MPI_Finalize();
        exit(0);
    }

    // 计算每个维度上的块数
    blk_num = mat_dim / BLK_DIM;

    // 初始化矩阵
    if (rank == 0) {
        mat_a = new double[3 * mat_dim * mat_dim];
        mat_b = mat_a + mat_dim * mat_dim;
        mat_c = mat_b + mat_dim * mat_dim;
        init_mats(mat_dim, mat_a, mat_b, mat_c);
        // 打印矩阵A和B
        // std::cout << "Matrix A:" << std::endl;
        // print_matrix(mat_a, mat_dim);
        // std::cout << "Matrix B:" << std::endl;
        // print_matrix(mat_b, mat_dim);
    }

    // 分配本地缓冲区
    local_a = new double[3 * BLK_DIM * BLK_DIM];
    local_b = local_a + BLK_DIM * BLK_DIM;
    local_c = local_b + BLK_DIM * BLK_DIM;

    // 同步所有进程
    MPI_Barrier(MPI_COMM_WORLD);

    // 记录开始时间
    t1 = MPI_Wtime();

    // 计算工作单元的总数
    int work_id_len = blk_num * blk_num;

    // 主进程的工作
    if (rank == 0) {
        // 分发A和B，并接收结果
        int work_start_id = 0;

        // 遍历所有工作单元
        while (work_start_id < work_id_len) {
            int worker, nworkers;
            int global_j;

            // 在最后一轮迭代中，并非所有工作进程都在工作，因为工作单元的数量不一定均匀
            nworkers = std::min(work_id_len - work_start_id, nprocs - 1);
            MPI_Request *send_requests_a = new MPI_Request[nworkers];
            // 将A的块发送给所有工作进程
            for (worker = 0; worker < nworkers; ++worker) {
                int work_id = work_start_id + worker;
                int global_i = work_id / blk_num;
                int global_k = work_id % blk_num;
                pack_global_to_local(local_a, mat_a, mat_dim, global_i, global_k);
                MPI_Isend(local_a, BLK_DIM * BLK_DIM, MPI_DOUBLE, worker+1, 0, MPI_COMM_WORLD, &send_requests_a[worker]);
                // std::cout << "1发送A" << std::endl;
            }
            // MPI_Waitall(nworkers, send_requests_a, MPI_STATUSES_IGNORE);
            delete[] send_requests_a;

            
            // 发送B的块并接收结果
            for (global_j = 0; global_j < blk_num; ++global_j) {
                MPI_Request *send_requests_b = new MPI_Request[nworkers];
                // 向每个工作进程发送B的一个块
                for (worker = 0; worker < nworkers; ++worker) {
                    int work_id = work_start_id + worker;
                    int global_k = work_id % blk_num;
                    pack_global_to_local(local_b, mat_b, mat_dim, global_k, global_j);
                    MPI_Isend(local_b, BLK_DIM * BLK_DIM, MPI_DOUBLE, worker+1, 0, MPI_COMM_WORLD, &send_requests_b[worker]);
                }
                // 在接收之前需要等待所有发送完成
                //MPI_Waitall(nworkers, send_requests_b, MPI_STATUSES_IGNORE);
                delete[] send_requests_b;
                // std::cout << "0发送B" << std::endl;
                // 工作进程在计算当前子矩阵的同时，主进程与其他工作进程通信

                // 接收所有工作进程的结果
                for (worker = 0; worker < nworkers; ++worker) {
                    int work_id = work_start_id + worker;
                    int global_i = work_id / blk_num;

                    MPI_Recv(local_c, BLK_DIM * BLK_DIM, MPI_DOUBLE, worker+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    add_local_to_global(mat_c, local_c, mat_dim, global_i, global_j);
                    // std::cout << "maser进程添加了global_i和global_j分别是" << global_i<<" "<<global_j<<",来自worker:"<<worker+1<< std::endl;
                }
            }
            work_start_id += nworkers;
        }
    } else {
        // std::cout << rank << std::endl;
        // 为接收预取的A和B块的请求分配空间
        MPI_Request recv_a_req = MPI_REQUEST_NULL, pf_recv_a_req = MPI_REQUEST_NULL;
        std::vector<MPI_Request> pf_b_reqs(blk_num), recv_b_reqs(blk_num);

        // 为预取的A和B块分配空间
        double *pf_local_a = new double[BLK_DIM * BLK_DIM];
        double *pf_local_bs = new double[blk_num * BLK_DIM * BLK_DIM];
        double *local_bs = new double[blk_num * BLK_DIM * BLK_DIM];

        int first_work_id = rank - 1;

        // 预取第一轮迭代使用到的A和B块
        if (first_work_id < work_id_len) {
            MPI_Irecv(pf_local_a, BLK_DIM * BLK_DIM, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &pf_recv_a_req);
            for (int i = 0; i < blk_num; ++i) {
                MPI_Irecv(&pf_local_bs[i * BLK_DIM * BLK_DIM], BLK_DIM * BLK_DIM, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &pf_b_reqs[i]);
            }
        }
        for (int work_id = first_work_id; work_id < work_id_len; work_id += nprocs - 1) {
            MPI_Wait(&pf_recv_a_req, MPI_STATUS_IGNORE);
            // std::cout << rank << "接受A完毕" << std::endl;

            // 对每个B块进行接收、计算和发送结果
            for (int i = 0; i < blk_num; ++i) {
                // 等待接收到当前B块
                MPI_Wait(&pf_b_reqs[i], MPI_STATUS_IGNORE);
                // std::cout << rank << "接受B" << i << "完毕" << std::endl;

                // 执行计算
                // std::cout << "开始计算" << std::endl;
                if (!is_zero_local(pf_local_a) && !is_zero_local(&pf_local_bs[i * BLK_DIM * BLK_DIM])) {
                    dgemm(pf_local_a, &pf_local_bs[i * BLK_DIM * BLK_DIM], local_c);
                } else {
                    std::fill_n(local_c, BLK_DIM * BLK_DIM, 0.0);
                }

                // 使用非阻塞发送将计算结果C发送回主进程
                MPI_Request send_req;
                MPI_Isend(local_c, BLK_DIM * BLK_DIM, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &send_req);
                MPI_Wait(&send_req, MPI_STATUS_IGNORE); // 等待发送完成
            }

            // 为下一轮迭代预取A和B块
            if (work_id + nprocs - 1 < work_id_len) {
                MPI_Irecv(pf_local_a, BLK_DIM * BLK_DIM, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &pf_recv_a_req);
                for (int i = 0; i < blk_num; ++i) {
                    MPI_Irecv(&pf_local_bs[i * BLK_DIM * BLK_DIM], BLK_DIM * BLK_DIM, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &pf_b_reqs[i]);
                }
            }
        }

        // 释放预取缓冲区
        delete[] pf_local_a;
        delete[] pf_local_bs;
        delete[] local_bs;
    }

    // 记录结束时间
    t2 = MPI_Wtime();

    // 同步所有进程
    MPI_Barrier(MPI_COMM_WORLD);

    // 主进程检查结果并打印时间
    if (rank == 0) {
        // check_mats函数用于检查结果的正确性
        check_mats(mat_a, mat_b, mat_c, mat_dim);
        std::cout << "[" << rank << "] 非阻塞实现时间: " << (t2 - t1) << std::endl;
        // std::cout << "[" << rank << "] 时间: " << (t2 - t1) << std::endl;
        // // 打印矩阵C
        // std::cout << "Matrix C (Result):" << std::endl;
        // print_matrix(mat_c, mat_dim);
    }

    // 释放本地缓冲区
    delete[] local_a;
    // 主进程释放全局矩阵
    if (rank == 0) {
        delete[] mat_a;
    }

    // 结束MPI环境
    // MPI_Finalize();
    // return 0;
}