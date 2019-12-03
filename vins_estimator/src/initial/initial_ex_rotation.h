#pragma once 

#include <vector>
#include "../parameters.h"
using namespace std;

#include <opencv2/opencv.hpp>

#include <eigen3/Eigen/Dense>
using namespace Eigen;
#include <ros/console.h>

/* This class help you to calibrate extrinsic rotation between imu and camera when your totally don't konw the extrinsic parameter */
class InitialEXRotation
{
public:
	InitialEXRotation();
    //标定外参矩阵
    bool CalibrationExRotation(vector<pair<Vector3d, Vector3d>> corres, Quaterniond delta_q_imu, Matrix3d &calib_ric_result);
private:
    //求解帧间cam坐标系的旋转矩阵
	Matrix3d solveRelativeR(const vector<pair<Vector3d, Vector3d>> &corres);
    //三角化验证R,t
    double testTriangulation(const vector<cv::Point2f> &l,
                             const vector<cv::Point2f> &r,
                             cv::Mat_<double> R, cv::Mat_<double> t);
    //本质矩阵SVD分解计算4组R,t值
    void decomposeE(cv::Mat E,
                    cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                    cv::Mat_<double> &t1, cv::Mat_<double> &t2);
    int frame_count; //帧计数器

    vector< Matrix3d > Rc; //帧间cam的R，由对极几何得到
    vector< Matrix3d > Rimu; //帧间IMU的R，由IMU预积分得到
    vector< Matrix3d > Rc_g; //每帧IMU相对于起始帧IMU的R
    Matrix3d ric; //cam到IMU的外参
};

