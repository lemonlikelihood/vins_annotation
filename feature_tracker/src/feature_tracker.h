#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
  public:
    FeatureTracker();

    void readImage(const cv::Mat &_img,double _cur_time);

    void setMask();

    void addPoints();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();

    void undistortedPoints();

    cv::Mat mask; //图像掩码
    cv::Mat fisheye_mask; //鱼眼相机mask，去除边缘噪点
    cv::Mat prev_img, cur_img, forw_img; //上一次发布的图像帧，当前帧，光流追踪的下一帧
    vector<cv::Point2f> n_pts; //当我们对光流追踪的特征点处理好后，我们还需要将特征点的数量补齐(利用goodFeaturesToTrack实现，放在n_pts中)
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;  //对应图像的特征点
    vector<cv::Point2f> prev_un_pts, cur_un_pts; //特帧点归一化坐标
    vector<cv::Point2f> pts_velocity; //特帧点在x,y上的运动速度
    vector<int> ids; //能够被追踪到的点的id
    vector<int> track_cnt; //每个被追踪到的点的被追踪到的次数
    map<int, cv::Point2f> cur_un_pts_map;
    map<int, cv::Point2f> prev_un_pts_map;
    camodocal::CameraPtr m_camera;
    double cur_time;
    double prev_time;

    static int n_id; //特帧点id，光流追踪到新的特征点，就给它这个id，然后id递增
};
