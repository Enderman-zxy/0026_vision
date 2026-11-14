#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

/*
 * PoseEstimator
 * 类用于估计相机相对标靶的位姿（位置和方向），并提供绘制坐标轴的功能
 * 通过给定的相机内参和畸变系数，PoseEstimator
 * 可以根据图像中的标记点和对应的三维点计算相机的旋转向量和平移向量，并在图像上绘制坐标轴以可视化
 *
 * [H8] 车上没有云台 => 相机与车体刚性固连，相机系 -> 车体系是一个常量变换。
 *      所以这里除了相机系的 tvec，还提供"目标在车体地面系下的右/前偏移"，
 *      电控要的是后者，不是 |tvec|（斜距会把横向偏差和高度差一起算进去）。
 */

// 相机安装外参（车上没有云台，这几个量装好车量一次就固定了）
struct CameraExtrinsics {
  // 光轴相对水平面的下俯角（度），正值 = 镜头朝下看。
  double pitch_deg = 0.0;
  // 相机光心离地高度（米）。
  double cam_height_m = 0.20;
  // MARK 灯中心离地高度（米）。规则手册图 3-6/3-7 推算：
  // 地面到后框顶面 203mm，后框高 128mm -> 箱底离地 75mm；识别板高 120mm
  // -> 板中心 75 + 60 = 135mm。取 MARK 灯环中心与板中心一致。
  double target_height_m = 0.135;
  // 相机光心相对车体旋转中心的前向偏移（米），装在车头前方时为正。
  double cam_forward_m = 0.0;
};

class PoseEstimator {
 public:
  PoseEstimator(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs,
                const CameraExtrinsics& extrinsics = CameraExtrinsics());

  // 位姿解算（>=4 个点，全部点参与精化，不再只拿第 4 个点做校验）
  bool solve(const std::vector<cv::Point2f>& image_points,
             const std::vector<cv::Point3f>& object_points);

  // 绘制坐标轴
  void drawAxis(cv::Mat& frame, float axis_length);

  // 获取平移向量和旋转向量
  cv::Vec3d getTvec() const { return tvec_; }
  cv::Vec3d getRvec() const { return rvec_; }
  const Eigen::Matrix3d& getR() const { return R_eigen_; }

  /**
   * @brief 目标在车体地面系下的偏移（毫米）。
   *        right_mm            目标在车体右侧多少（右为正）
   *        forward_mm 目标在车体前方多少（前为正，已补上相机安装的前移量）
   *        height_residual_mm  竖直方向残差（诊断用，正常应接近 0；
   *                            明显非 0 说明 pitch_deg / cam_height_m /
   * 灯高标错了）
   * @return 失败（还没有有效位姿）时返回 false。
   */
  bool getGroundOffset(double& right_mm, double& forward_mm,
                       double& height_residual_mm) const;
  // 像素 -> 归一化坐标（含畸变校正）。给调试/自检用。
  bool pixelToNormalized(const cv::Point2f& px, cv::Point2d& norm) const;

 private:
  // 用全部点做 6 自由度 Gauss-Newton 精化
  bool refinePose(const std::vector<cv::Point2d>& obs_norm,
                  const std::vector<Eigen::Vector3d>& obj_pts,
                  Eigen::Matrix3d& R, Eigen::Vector3d& t) const;
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  CameraExtrinsics extrinsics_;

  cv::Vec3d rvec_ = cv::Vec3d(0, 0, 0);
  cv::Vec3d tvec_;
  Eigen::Matrix3d R_eigen_ = Eigen::Matrix3d::Identity();
};