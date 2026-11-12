#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include <Eigen/Dense>

/*
 * PoseEstimator 类用于估计相机的位姿（位置和方向），并提供绘制坐标轴的功能
 * 通过给定的相机内参和畸变系数，PoseEstimator
 * 可以根据图像中的标记点和对应的三维点计算相机的旋转向量和平移向量，并在图像上绘制坐标轴以可视化
 */

class PoseEstimator {
 public:
  PoseEstimator(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs);

  // 位姿解算
  bool solve(const std::vector<cv::Point2f>& image_points,
             const std::vector<cv::Point3f>& object_points);

  // 绘制坐标轴
  void drawAxis(cv::Mat& frame, float axis_length);

  // 获取平移向量和旋转向量
  cv::Vec3d getTvec() const { return tvec_; }
  cv::Vec3d getRvec() const { return rvec_; }

 private:
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  cv::Vec3d rvec_ = cv::Vec3d(0,0,0);
  cv::Vec3d tvec_;
  Eigen::Matrix3d R_eigen_;
};