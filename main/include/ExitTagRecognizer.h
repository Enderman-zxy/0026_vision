#pragma once

#include <opencv2/opencv.hpp>

// #include "maix_image.hpp"

struct ExitTagResult {
  bool detected = false;
  int id = -1;
  int16_t err_x = 0;
  int16_t err_y = 0;
  int16_t width_px = 0;
  std::vector<cv::Point2f> corners;
  cv::Point2f center{0.f,0.f};
};
class ExitTagRecognizer {
 public:
  ExitTagRecognizer();

  /**
   * @brief 识别图像中的 AprilTag
   * @param maix_img 摄像头输入的图像
   * @param img_w 图像宽度 (用于计算中心)
   * @param img_h 图像高度
   * @param result 输出的识别结果
   * @return 是否成功识别
   */
  bool detect(cv::Mat& frame, int img_w, int img_h, ExitTagResult& result);
};