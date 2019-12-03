#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "../utility/utility.h"

class PoseLocalParameterization : public ceres::LocalParameterization
{
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const { return 7; };  //表示参数x的自由度，七个维度，前三维是t:(x,y,z)；后三维是q:(x,y,z,w)
    virtual int LocalSize() const { return 6; };   //表示delta x所在的正切空间的自由度，可以理解为旋转实际上用三个维度就表示了，但是在四元数中是用了四个维度的
};
