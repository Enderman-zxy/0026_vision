#pragma once

#include <opencv2/opencv.hpp>

#include "MarkerDetector.h"

/*
 * Visualizer
 * 类用于在图像上绘制标记的检测结果，包括标记的四个角点、边界框和角点编号
 * 它使用不同的颜色来区分标记的四个角点，并在每个角点旁边绘制编号，以便更直观地显示标记的检测结果
 */

class Visualizer {
 public:
  Visualizer();
  void drawMarker(cv::Mat& frame, const MarkerResult& result);

 private:
  int line_thickness_ = 2;
  double font_scale_ = 0.8;
  int font_face_ = cv::FONT_HERSHEY_SIMPLEX;

  cv::Scalar color_lt_{0, 0, 255};        // 左上：红色
  cv::Scalar color_rt_{0, 255, 0};        // 右上：绿色
  cv::Scalar color_rb_{255, 0, 0};        // 右下：蓝色
  cv::Scalar color_lb_{0, 255, 255};      // 左下：黄色
  cv::Scalar color_box_{0, 255, 0};       // 外接框：绿色
  cv::Scalar color_text_{255, 255, 255};  // 文字：白色
};