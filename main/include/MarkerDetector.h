#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

struct MarkerResult {
  bool detected_ = false;
  std::vector<cv::Point2f> points_;
  cv::Rect bounding_box_;
};

/*
 * MarkerDetector 类用于检测图像中的标记（Marker），并返回检测结果
 * 包括标记是否被检测到、标记的四个角点坐标以及标记的边界框
 */

class MarkerDetector {
 public:
  MarkerDetector();
  std::vector<MarkerResult> detectMultiple(const cv::Mat& frame);

 private:
  cv::Mat preprocess(const cv::Mat& frame);
  std::vector<cv::Point2f> sortPoints(const std::vector<cv::Point2f>& pts);
};