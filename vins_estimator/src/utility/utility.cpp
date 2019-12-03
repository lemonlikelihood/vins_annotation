#include "utility.h"

Eigen::Matrix3d Utility::g2R(const Eigen::Vector3d &g)
{
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();
    Eigen::Vector3d ng2{0, 0, 1.0}; //因为VINS中，g的使用都是-g的形式，所以用正的
    //R0 * ng1 = ng2
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix(); //求出了g和(0,0,1)之间的旋转，
    double yaw = Utility::R2ypr(R0).x(); //求出了两个向量在z方向上的转角
    
    //左乘，相对于固定坐标系进行变换；右乘，相对于自身坐标系变换
    //当我们进行了R0的旋转之后，已经保证了g和惯性坐标系的z轴上一致，但是这并不代表着原来的坐标系和惯性坐标系一致，可以想象惯性坐标系绕着z轴旋转任意角度，g依然能够对齐
    //所以我们将原来的坐标系绕着惯性坐标系的z轴旋转，确保不会出现x,y轴不对应的情况
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0; //这里求出的R0相当于在z轴上没有旋转
    //R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
    return R0;
}
