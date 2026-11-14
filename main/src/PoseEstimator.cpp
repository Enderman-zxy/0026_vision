#include "PoseEstimator.h"

#include <PoseLib/poselib.h>

#include <Eigen/Dense>
#include <iostream>

#include "maix_basic.hpp"

namespace {
// 位姿精化迭代次数（4 个点 6 自由度，通常 3~5 次就收敛）
constexpr int kRefineIters = 8;
// 重投影 RMS 上限（像素）。超过这个值说明四边形拟合本身不可信。
constexpr double kMaxReprojRmsPx = 6.0;
}  // namespace

PoseEstimator::PoseEstimator(const cv::Mat& camera_matrix,
                             const cv::Mat& dist_coeffs,
                             const CameraExtrinsics& extrinsics)
    : camera_matrix_(camera_matrix.clone()),
      dist_coeffs_(dist_coeffs.clone()),
      extrinsics_(extrinsics) {}

// [M1] 像素 -> 归一化坐标，并按 plumb-bob 模型反解畸变（迭代 8 次）。
// 旧实现直接把 (u-cx)/fx 当归一化坐标，dist_coeffs_ 读了从来没用过。
// MaixCDK 的 OpenCV 没有 calib3d（没有 undistortPoints），所以这里手写。
bool PoseEstimator::pixelToNormalized(const cv::Point2f& px,
                                      cv::Point2d& norm) const {
  if (camera_matrix_.empty() || camera_matrix_.rows < 3) return false;
  const double fx = camera_matrix_.at<double>(0, 0);
  const double fy = camera_matrix_.at<double>(1, 1);
  const double cx = camera_matrix_.at<double>(0, 2);
  const double cy = camera_matrix_.at<double>(1, 2);
  if (std::fabs(fx) < 1e-9 || std::fabs(fy) < 1e-9) return false;

  const double xd = (px.x - cx) / fx;
  const double yd = (px.y - cy) / fy;

  if (dist_coeffs_.empty() || dist_coeffs_.total() < 4) {
    norm = cv::Point2d(xd, yd);
    return true;
  }

  const double k1 = dist_coeffs_.at<double>(0, 0);
  const double k2 = dist_coeffs_.at<double>(0, 1);
  const double p1 = dist_coeffs_.at<double>(0, 2);
  const double p2 = dist_coeffs_.at<double>(0, 3);
  const double k3 =
      (dist_coeffs_.total() >= 5) ? dist_coeffs_.at<double>(0, 4) : 0.0;

  double x = xd, y = yd;
  for (int i = 0; i < 8; ++i) {
    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;
    const double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
    if (std::fabs(radial) < 1e-9) return false;
    const double dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    const double dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
    x = (xd - dx) / radial;
    y = (yd - dy) / radial;
    if (!std::isfinite(x) || !std::isfinite(y)) return false;
  }
  norm = cv::Point2d(x, y);
  return true;
}

// 用全部点做 Gauss-Newton 精化：min sum || pi(R*X + t) - p ||^2
bool PoseEstimator::refinePose(const std::vector<cv::Point2d>& obs_norm,
                               const std::vector<Eigen::Vector3d>& obj_pts,
                               Eigen::Matrix3d& R, Eigen::Vector3d& t) const {
  const int n = static_cast<int>(obs_norm.size());
  if (n < 4 || static_cast<int>(obj_pts.size()) != n) return false;

  for (int iter = 0; iter < kRefineIters; ++iter) {
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();

    for (int i = 0; i < n; ++i) {
      const Eigen::Vector3d Pc = R * obj_pts[i] + t;
      if (Pc.z() <= 1e-6) return false;
      const double iz = 1.0 / Pc.z();
      const double iz2 = iz * iz;
      const double u = Pc.x() * iz;
      const double v = Pc.y() * iz;

      Eigen::Matrix<double, 2, 1> r;
      r << u - obs_norm[i].x, v - obs_norm[i].y;

      Eigen::Matrix<double, 2, 3> Jp;
      Jp << iz, 0.0, -Pc.x() * iz2, 0.0, iz, -Pc.y() * iz2;

      Eigen::Matrix3d Pc_hat;
      Pc_hat << 0.0, -Pc.z(), Pc.y(), Pc.z(), 0.0, -Pc.x(), -Pc.y(), Pc.x(),
          0.0;

      Eigen::Matrix<double, 2, 6> J;
      J.block<2, 3>(0, 0) = -Jp * Pc_hat;  // 对旋转增量
      J.block<2, 3>(0, 3) = Jp;            // 对平移增量

      H += J.transpose() * J;
      b += J.transpose() * r;
    }

    // 轻微 LM 阻尼，避免近似平面构型下奇异
    H.diagonal().array() += 1e-6 * (1.0 + H.trace());
    const Eigen::Matrix<double, 6, 1> dx = H.ldlt().solve(-b);
    if (!dx.allFinite()) return false;

    const Eigen::Vector3d w = dx.head<3>();
    const double angle = w.norm();
    if (angle > 1e-12) {
      R = Eigen::AngleAxisd(angle, w / angle).toRotationMatrix() * R;
    }
    t += dx.tail<3>();

    if (dx.norm() < 1e-10) break;
  }
  return R.allFinite() && t.allFinite();
}

// bool PoseEstimator::solve(const std::vector<cv::Point2f>& image_points,
//                           const std::vector<cv::Point3f>& object_points) {
//   if (image_points.size() < 4 || object_points.size() < 4) return false;

//   double fx = camera_matrix_.at<double>(0, 0);
//   double fy = camera_matrix_.at<double>(1, 1);
//   double cx = camera_matrix_.at<double>(0, 2);
//   double cy = camera_matrix_.at<double>(1, 2);

//   // 构造点集
//   std::vector<Eigen::Vector3d> x, X;
//   for (size_t i = 0; i < 4; ++i) {
//     double xn = (image_points[i].x - cx) / fx;
//     double yn = (image_points[i].y - cy) / fy;
//     Eigen::Vector3d bearing(xn, yn, 1.0);
//     bearing.normalize();
//     x.push_back(bearing);
//     X.push_back(Eigen::Vector3d(object_points[i].x, object_points[i].y,
//                                 object_points[i].z));
//   }

//   // P3P 传入前 3 个点
//   std::vector<Eigen::Vector3d> x3(x.begin(), x.begin() + 3);
//   std::vector<Eigen::Vector3d> X3(X.begin(), X.begin() + 3);

//   std::vector<poselib::CameraPose> poses;
//   int num_solutions = poselib::p3p(x3, X3, &poses);

//   if (num_solutions == 0 || poses.empty()) return false;

//   double min_reproj_err = 1e10;
//   int best_idx = -1;

//   for (size_t i = 0; i < poses.size(); ++i) {
//     Eigen::Matrix3d R = poses[i].R();
//     Eigen::Vector3d t = poses[i].t;

//     // 用当前解，把第 4 个 3D 点投影到图像上
//     Eigen::Vector3d P_c = R * X[3] + t;

//     // 必须在相机前方
//     if (P_c.z() <= 0) continue;

//     // 投影计算像素坐标
//     double u_pred = fx * (P_c.x() / P_c.z()) + cx;
//     double v_pred = fy * (P_c.y() / P_c.z()) + cy;

//     // 计算与真实检测到的第 4 个点的像素误差（平方和）
//     double err = std::pow(u_pred - image_points[3].x, 2) +
//                  std::pow(v_pred - image_points[3].y, 2);

//     // maix::log::info("Solution %d: reproj_err = %f", i, err);
//     if (err < min_reproj_err) {
//       min_reproj_err = err;
//       best_idx = i;
//     }
//   }

//   if (best_idx == -1) return false;

//   // 保存最佳解的位姿
//   R_eigen_ = poses[best_idx].R();
//   Eigen::Vector3d t = poses[best_idx].t;
//   tvec_ = cv::Vec3d(t.x(), t.y(), t.z());

//   return true;
// }

bool PoseEstimator::solve(const std::vector<cv::Point2f>& image_points,
                          const std::vector<cv::Point3f>& object_points) {
  // [M5] 先全部算到局部量，成功后再写成员，避免解算失败时留下半个结果
  if (image_points.size() < 4 || image_points.size() != object_points.size())
    return false;

  const double fx = camera_matrix_.at<double>(0, 0);
  const double fy = camera_matrix_.at<double>(1, 1);

  std::vector<cv::Point2d> obs_norm;
  std::vector<Eigen::Vector3d> obj_pts;
  obs_norm.reserve(image_points.size());
  obj_pts.reserve(object_points.size());

  for (size_t i = 0; i < image_points.size(); ++i) {
    cv::Point2d n;
    if (!pixelToNormalized(image_points[i], n)) return false;
    obs_norm.push_back(n);
    obj_pts.push_back(Eigen::Vector3d(object_points[i].x, object_points[i].y,
                                      object_points[i].z));
  }

  // 用去畸变后的归一化坐标做单位方向向量
  std::vector<Eigen::Vector3d> bearings;
  bearings.reserve(obs_norm.size());
  for (const auto& n : obs_norm) {
    Eigen::Vector3d b(n.x, n.y, 1.0);
    b.normalize();
    bearings.push_back(b);
  }

  // P3P 传前 3 个点拿到初值（P3P 一定给出前 3 点的精确解）
  std::vector<Eigen::Vector3d> x3(bearings.begin(), bearings.begin() + 3);
  std::vector<Eigen::Vector3d> X3(obj_pts.begin(), obj_pts.begin() + 3);
  std::vector<poselib::CameraPose> poses;
  if (poselib::p3p(x3, X3, &poses) == 0 || poses.empty()) return false;

  // [M2] 旧的评分只用第 4 个点的重投影误差，而且 3 个点构成直角等腰三角形，
  //      是 P3P 最经典的歧义构型。现在：所有候选解都用全部点打分，选最优后用
  //      全部点做 6 自由度精化（相当于把 4 个角点的噪声平均掉）。
  double best_cost = std::numeric_limits<double>::max();
  int best_idx = -1;
  for (size_t i = 0; i < poses.size(); ++i) {
    const Eigen::Matrix3d R = poses[i].R();
    const Eigen::Vector3d t = poses[i].t;
    double cost = 0.0;
    bool ok = true;
    for (size_t k = 0; k < obj_pts.size(); ++k) {
      const Eigen::Vector3d Pc = R * obj_pts[k] + t;
      if (Pc.z() <= 1e-6) {
        ok = false;
        break;
      }
      const double u = Pc.x() / Pc.z();
      const double v = Pc.y() / Pc.z();
      cost += (u - obs_norm[k].x) * (u - obs_norm[k].x) +
              (v - obs_norm[k].y) * (v - obs_norm[k].y);
    }
    if (!ok) continue;
    if (cost < best_cost) {
      best_cost = cost;
      best_idx = static_cast<int>(i);
    }
  }
  if (best_idx < 0) return false;

  Eigen::Matrix3d R = poses[best_idx].R();
  Eigen::Vector3d t = poses[best_idx].t;
  const double cost_before = best_cost;

  if (!refinePose(obs_norm, obj_pts, R, t)) {
    R = poses[best_idx].R();
    t = poses[best_idx].t;
  } else {
    // 精化后如果反而变差，退回 P3P 的原始解
    double cost_after = 0.0;
    bool ok = true;
    for (size_t k = 0; k < obj_pts.size(); ++k) {
      const Eigen::Vector3d Pc = R * obj_pts[k] + t;
      if (Pc.z() <= 1e-6) {
        ok = false;
        break;
      }
      const double u = Pc.x() / Pc.z();
      const double v = Pc.y() / Pc.z();
      cost_after += (u - obs_norm[k].x) * (u - obs_norm[k].x) +
                    (v - obs_norm[k].y) * (v - obs_norm[k].y);
    }
    if (!ok || cost_after > cost_before) {
      R = poses[best_idx].R();
      t = poses[best_idx].t;
    }
    best_cost = std::min(cost_before, ok ? cost_after : cost_before);
  }

  // 重投影 RMS（像素）做一次整体可信度检查
  const double rms_px =
      std::sqrt(best_cost / static_cast<double>(obj_pts.size())) *
      std::sqrt((fx * fx + fy * fy) * 0.5);
  if (!std::isfinite(rms_px) || rms_px > kMaxReprojRmsPx) return false;

  // [M4] 旋转向量也要写，否则 getRvec() 永远返回 (0,0,0)
  Eigen::AngleAxisd aa(R);
  cv::Vec3d rvec = cv::Vec3d(0, 0, 0);
  if (aa.angle() > 1e-9) {
    const Eigen::Vector3d axis = aa.axis() * aa.angle();
    rvec = cv::Vec3d(axis.x(), axis.y(), axis.z());
  }

  R_eigen_ = R;
  tvec_ = cv::Vec3d(t.x(), t.y(), t.z());
  rvec_ = rvec;
  return true;
}

// [H8] 相机系 -> 车体地面系。
// 相机系：x 右、y 下、z 前（光轴）；地面系：x 右、y 前、z 上。
// 光轴下俯 pitch 时：z_c = cos*p - sin*q, y_c = -sin*p - cos*q（p=前, q=上）
// 反解：p =  cos*Z_c - sin*Y_c
//       q = -sin*Z_c - cos*Y_c
bool PoseEstimator::getGroundOffset(double& right_mm, double& forward_mm,
                                    double& height_residual_mm) const {
  const double th = extrinsics_.pitch_deg * M_PI / 180.0;
  const double s = std::sin(th), c = std::cos(th);
  const double xc = tvec_[0], yc = tvec_[1], zc = tvec_[2];

  const double right = xc;
  const double forward = c * zc - s * yc;
  const double up = -s * zc - c * yc;

  right_mm = right * 1000.0;
  forward_mm = (forward + extrinsics_.cam_forward_m) * 1000.0;
  height_residual_mm =
      (up - (extrinsics_.target_height_m - extrinsics_.cam_height_m)) * 1000.0;
  return std::isfinite(right_mm) && std::isfinite(forward_mm);
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

  // [M] 三个点都要校验，旧实现只查了前两个
  if (image_points[0].x >= 0 && image_points[1].x >= 0 &&
      image_points[2].x >= 0) {
    cv::line(frame, image_points[0], image_points[1], cv::Scalar(0, 0, 255),
             3);  // X (红)
    cv::line(frame, image_points[0], image_points[2], cv::Scalar(255, 0, 0),
             3);  // Z (蓝)
  }
}