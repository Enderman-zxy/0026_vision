// #include "PoseEstimator.h"

// #include <PoseLib/poselib.h>

// #include <iostream>

// #include "Eigen/Dense"

// PoseEstimator::PoseEstimator(const cv::Mat& camera_matrix,
//                              const cv::Mat& dist_coeffs)
//     : camera_matrix_(camera_matrix.clone()),
//       dist_coeffs_(dist_coeffs.clone()) {}

// // 位姿解算
// bool PoseEstimator::solve(const std::vector<cv::Point2f>& image_points,
//                           const std::vector<cv::Point3f>& object_points) {
//   if (image_points.size() < 4 || object_points.size() < 4) {
//     std::cerr << "Not enough points for pose estimation." << std::endl;
//     return false;
//   }
//   // bool success = cv::solvePnP(object_points, image_points, camera_matrix_,
//   //                             dist_coeffs_, rvec_, tvec_);
//   // return success;

//   double fx = camera_matrix_.at<double>(0, 0);
//   double fy = camera_matrix_.at<double>(1, 1);
//   double cx = camera_matrix_.at<double>(0, 2);
//   double cy = camera_matrix_.at<double>(1, 2);

//   // 2. 构造 PoseLib 需要的点集（必须是 Vector3d）
//   std::vector<Eigen::Vector3d> x;  // 这里是 Vector3d
//   std::vector<Eigen::Vector3d> X;
//   for (size_t i = 0; i < 4; ++i) {
//     // 提取像素坐标
//     double u = image_points[i].x;
//     double v = image_points[i].y;

//     // 核心转换：像素坐标 -> 归一化平面坐标 -> 单位方向向量
//     double xn = (u - cx) / fx;
//     double yn = (v - cy) / fy;

//     Eigen::Vector3d bearing(xn, yn, 1.0);
//     bearing.normalize();  // 必须归一化，让向量长度为 1

//     x.push_back(bearing);
//     X.push_back(Eigen::Vector3d(object_points[i].x, object_points[i].y,
//                                 object_points[i].z));
//   }

//   // 2. 调用 PoseLib 的 P3P 求解器（传入前 3 个点）
//   std::vector<poselib::CameraPose> poses;
//   int num_solutions = poselib::p3p(x, X, &poses);

//   if (num_solutions == 0) {
//     return false;
//   }

//   // 3. P3P 可能会返回 1~4 个解，通常有多个解都能对应上。
//   //
//   简单起见，这里直接取第一个解（如果你的场景中镜头不会倒置，通常第一个就是对的）。
//   // 严谨的做法是用第 4 个点去重投影验算，选误差最小的那个。
//   Eigen::Matrix3d R = poses[0].R();
//   Eigen::Vector3d t = poses[0].t;

//   // 4. 将 Eigen 的旋转矩阵转为 OpenCV 的旋转向量 (rvec)
//   cv::Mat R_cv(3, 3, CV_64F);
//   for (int i = 0; i < 3; ++i)
//     for (int j = 0; j < 3; ++j) R_cv.at<double>(i, j) = R(i, j);

//   // cv::Mat rvec_cv;
//   // cv::Rodrigues(R_cv, rvec_cv);  // 依赖 core 模块，可以放心用

//   Eigen::AngleAxisd angle_axis(R);
//   Eigen::Vector3d rvec = angle_axis.angle() * angle_axis.axis();

//   // 5. 赋值给类成员变量
//   rvec_ = cv::Vec3d(rvec.x(), rvec.y(), rvec.z());
//   tvec_ = cv::Vec3d(t.x(), t.y(), t.z());
// }

// // 绘制坐标轴
// void PoseEstimator::drawAxis(cv::Mat& frame, float axis_length) {
//   std::vector<cv::Point3f> axis_points = {
//       cv::Point3f(0, 0, 0), cv::Point3f(axis_length, 0, 0),
//       cv::Point3f(0, axis_length, 0), cv::Point3f(0, 0, axis_length)};

//   std::vector<cv::Point2f> image_points;

//   double fx = camera_matrix_.at<double>(0, 0);
//   double fy = camera_matrix_.at<double>(1, 1);
//   double cx = camera_matrix_.at<double>(0, 2);
//   double cy = camera_matrix_.at<double>(1, 2);

//   // cv::projectPoints(axis_points, rvec_, tvec_, camera_matrix_,
//   dist_coeffs_,
//   //                   image_points);

//   double theta = cv::norm(rvec_);
//   Eigen::Matrix3d R_eigen = Eigen::Matrix3d::Identity();

//   if (theta > 1e-9) {
//     Eigen::Vector3d axis(rvec_[0] / theta, rvec_[1] / theta, rvec_[2] /
//     theta); Eigen::AngleAxisd angle_axis(theta, axis); R_eigen =
//     angle_axis.toRotationMatrix();
//   }

//   for (const auto& pt : axis_points) {
//     // 1. 将 3D 点从世界坐标系转换到相机坐标系: P_c = R * P_w + t
//     Eigen::Vector3d P_w(pt.x, pt.y, pt.z);
//     Eigen::Vector3d t_eigen(tvec_[0], tvec_[1], tvec_[2]);
//     Eigen::Vector3d P_c = R_eigen * P_w + t_eigen;

//     double X_c = P_c.x();
//     double Y_c = P_c.y();
//     double Z_c = P_c.z();

//     if (Z_c <= 0) {  // 避免除以 0 或点跑到相机后面
//       image_points.push_back(cv::Point2f(-1, -1));
//       continue;
//     }

//     // 2. 针孔相机投影公式
//     double u = fx * (X_c / Z_c) + cx;
//     double v = fy * (Y_c / Z_c) + cy;
//     image_points.push_back(cv::Point2f(u, v));
//   }

//   if (image_points[0].x >= 0) {
//     cv::line(frame, image_points[0], image_points[1], cv::Scalar(0, 0, 255),
//              3);  // X-axis in red
//     // cv::line(frame, image_points[0], image_points[2], cv::Scalar(0, 255,
//     0),
//     //          3);  // Y-axis in green
//     cv::line(frame, image_points[0], image_points[3], cv::Scalar(0, 255, 0),
//              3);  // Z-axis in blue

//     double distance_m = cv::norm(tvec_);
//     double distance_mm = distance_m * 1000.0;  // Convert to millimeters
//     cv::putText(frame, "Distance: " + std::to_string(distance_mm) + " mm",
//                 cv::Point(30, 80), cv::FONT_HERSHEY_SIMPLEX, 0.8,
//                 cv::Scalar(0, 255, 255), 2);
//   }
// }

#include "PoseEstimator.h"

#include <PoseLib/poselib.h>

#include <Eigen/Dense>
#include <iostream>

#include "maix_basic.hpp"

PoseEstimator::PoseEstimator(const cv::Mat& camera_matrix,
                             const cv::Mat& dist_coeffs)
    : camera_matrix_(camera_matrix.clone()),
      dist_coeffs_(dist_coeffs.clone()) {}

bool PoseEstimator::solve(const std::vector<cv::Point2f>& image_points,
                          const std::vector<cv::Point3f>& object_points) {
  if (image_points.size() < 4 || object_points.size() < 4) return false;

  double fx = camera_matrix_.at<double>(0, 0);
  double fy = camera_matrix_.at<double>(1, 1);
  double cx = camera_matrix_.at<double>(0, 2);
  double cy = camera_matrix_.at<double>(1, 2);

  // 构造点集
  std::vector<Eigen::Vector3d> x, X;
  for (size_t i = 0; i < 4; ++i) {
    double xn = (image_points[i].x - cx) / fx;
    double yn = (image_points[i].y - cy) / fy;
    Eigen::Vector3d bearing(xn, yn, 1.0);
    bearing.normalize();
    x.push_back(bearing);
    X.push_back(Eigen::Vector3d(object_points[i].x, object_points[i].y,
                                object_points[i].z));
  }

  // P3P 传入前 3 个点
  std::vector<Eigen::Vector3d> x3(x.begin(), x.begin() + 3);
  std::vector<Eigen::Vector3d> X3(X.begin(), X.begin() + 3);

  std::vector<poselib::CameraPose> poses;
  int num_solutions = poselib::p3p(x3, X3, &poses);

  if (num_solutions == 0 || poses.empty()) return false;

  double min_reproj_err = 1e10;
  int best_idx = -1;

  for (size_t i = 0; i < poses.size(); ++i) {
    Eigen::Matrix3d R = poses[i].R();
    Eigen::Vector3d t = poses[i].t;

    // 用当前解，把第 4 个 3D 点投影到图像上
    Eigen::Vector3d P_c = R * X[3] + t;

    // 必须在相机前方
    if (P_c.z() <= 0) continue;

    // 投影计算像素坐标
    double u_pred = fx * (P_c.x() / P_c.z()) + cx;
    double v_pred = fy * (P_c.y() / P_c.z()) + cy;

    // 计算与真实检测到的第 4 个点的像素误差（平方和）
    double err = std::pow(u_pred - image_points[3].x, 2) +
                 std::pow(v_pred - image_points[3].y, 2);

    // maix::log::info("Solution %d: reproj_err = %f", i, err);
    if (err < min_reproj_err) {
      min_reproj_err = err;
      best_idx = i;
    }
  }

  if (best_idx == -1) return false;

  // 保存最佳解的位姿
  R_eigen_ = poses[best_idx].R();
  Eigen::Vector3d t = poses[best_idx].t;
  tvec_ = cv::Vec3d(t.x(), t.y(), t.z());

  return true;
}

void PoseEstimator::drawAxis(cv::Mat& frame, float axis_length) {
  std::vector<cv::Point3f> axis_points = {
      cv::Point3f(0, 0, 0), cv::Point3f(axis_length, 0, 0),
      cv::Point3f(0, 0, axis_length)  // 只画 X 和 Z 轴就够了
  };

  std::vector<cv::Point2f> image_points;
  double fx = camera_matrix_.at<double>(0, 0);
  double fy = camera_matrix_.at<double>(1, 1);
  double cx = camera_matrix_.at<double>(0, 2);
  double cy = camera_matrix_.at<double>(1, 2);

  Eigen::Vector3d t_eigen(tvec_[0], tvec_[1], tvec_[2]);

  for (const auto& pt : axis_points) {
    Eigen::Vector3d P_w(pt.x, pt.y, pt.z);
    Eigen::Vector3d P_c = R_eigen_ * P_w + t_eigen;  // 直接用 R_eigen_

    if (P_c.z() <= 0) {
      image_points.push_back(cv::Point2f(-1, -1));
      continue;
    }
    double u = fx * (P_c.x() / P_c.z()) + cx;
    double v = fy * (P_c.y() / P_c.z()) + cy;
    image_points.push_back(cv::Point2f(u, v));
  }

  if (image_points[0].x >= 0 && image_points[1].x >= 0) {
    cv::line(frame, image_points[0], image_points[1], cv::Scalar(0, 0, 255),
             3);  // X (红)
    cv::line(frame, image_points[0], image_points[2], cv::Scalar(255, 0, 0),
             3);  // Z (蓝)
  }
}