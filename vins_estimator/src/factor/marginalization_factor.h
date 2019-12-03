#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4; //最多四个线程同时处理

struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function;   
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks;  //待优化的数据，以IMU为例，就是[7,9,7,9](即IMU残差的优化变量)
    std::vector<int> drop_set;  //待merge的优化变量id

    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals; //对于IMU就是15X1,对于视觉就是2X1

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

class MarginalizationInfo
{
  public:
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;

    //添加残差块的相关信息
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    //计算残差块的Jacobian,更新parameter_block_data
    void preMarginalize();
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);
    
    //存储了所有的变量的信息
    std::vector<ResidualBlockInfo *> factors;
    int m, n; //m表示将要被merge的变量size，n表示剩下的

    //这里long指的都是内存地址，int或者double表示的就是这个内存指向的数据的size(或是数据，或是变量的信息)
    //内存地址由我们将优化变量的vector传入，进而将首地址强制转换成long得到

    //global size，即<所有变量的内存地址,local_size>
    //存的是优化变量
    std::unordered_map<long, int> parameter_block_size; 
    int sum_block_size;
    std::unordered_map<long, int> parameter_block_idx; //local size，即<待marge的优化变量内存地址，在parameter_block_size中的id>
    //<所有变量的数据地址,数据vector>，也就是我们为了构造残差，所给到的估计值
    //对于IMU而言，就是三个增量，共9维；对于视觉就是两个像素坐标Pi和Pj，共6维
    std::unordered_map<long, double *> parameter_block_data;  

    std::vector<int> keep_block_size; //global size //剩下的变量
    std::vector<int> keep_block_idx;  //local size
    std::vector<double *> keep_block_data;

    Eigen::MatrixXd linearized_jacobians; //边缘化之后恢复出来的jacobain矩阵
    Eigen::VectorXd linearized_residuals; //边缘化后得到的residuals
    const double eps = 1e-8;

};

class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};
