// #include "ExitTagRecognizer.h"

// #include "maix_log.hpp"

// using namespace maix;

// ExitTagRecognizer::ExitTagRecognizer() {}

// bool ExitTagRecognizer::detect(maix::image::Image* maix_img, int img_w,
//                                int img_h, ExitTagResult& result) {
//   result.detected = false;
//   result.id = -1;
//   result.corners.clear();

//   if (!maix_img) return false;

//   // 调用 MaixCDK 底层接口，指定 TAG36H11 家族
//   auto apriltags =
//   maix_img->find_apriltags({image::ApriltagFamilies::TAG36H11});

//   if (apriltags.empty()) {
//     return false;
//   }

//   // 默认取第一个识别到的标签
//   auto& tag = apriltags[0];
//   result.detected = true;
//   result.id = tag.id();

//   // 提取角点
//   auto tag_corners = tag.corners();
//   for (const auto& pt : tag_corners) {
//     result.corners.push_back(cv::Point2f(pt[0], pt[1]));
//   }

//   // 计算中心点
//   result.center = cv::Point2f(0, 0);
//   for (const auto& pt : result.corners) {
//     result.center += pt;
//   }
//   result.center *= 0.25f;

//   // 计算相对画面中心的偏差 (向右为正，向下为正)
//   float center_u = img_w / 2.0f;
//   float center_v = img_h / 2.0f;
//   result.err_x = static_cast<int16_t>(result.center.x - center_u);
//   result.err_y = static_cast<int16_t>(result.center.y - center_v);

//   // 计算标签宽度 (取左上角到右上角的欧氏距离)
//   float w = cv::norm(result.corners[1] - result.corners[0]);
//   result.width_px = static_cast<int16_t>(w);

//   return true;
// }

#include "ExitTagRecognizer.h"

#include <opencv2/opencv.hpp>

#include "maix_log.hpp"

using namespace maix;

ExitTagRecognizer::ExitTagRecognizer() {}

bool ExitTagRecognizer::detect(cv::Mat& frame, int img_w, int img_h,
                               ExitTagResult& result) {
  result.detected = false;
  if (frame.empty()) return false;

  // 不需要 image2cv 了，直接用 frame
  cv::Mat gray, binary;
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  cv::threshold(gray, binary, 100, 255,
                cv::THRESH_BINARY_INV);  // 黑底白字，反向阈值化找黑框

  // 找轮廓
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  double max_area = 0;
  std::vector<cv::Point> best_square;

  for (const auto& contour : contours) {
    double area = cv::contourArea(contour);
    if (area < 1000) continue;  // 忽略噪点

    std::vector<cv::Point> poly;
    cv::approxPolyDP(contour, poly, cv::arcLength(contour, true) * 0.05, true);

    if (poly.size() == 4) {
      if (area > max_area) {
        max_area = area;
        best_square = poly;
      }
    }
  }

  if (!best_square.empty()) {
    result.detected = true;

    for (const auto& pt : best_square) {
      result.corners.push_back(cv::Point2f(pt.x, pt.y));
    }

    result.center = cv::Point2f(0, 0);
    for (const auto& pt : result.corners) {
      result.center += pt;
    }
    result.center *= 0.25f;

    result.err_x = static_cast<int16_t>(result.center.x - (img_w / 2.0f));
    result.err_y = static_cast<int16_t>(result.center.y - (img_h / 2.0f));
    result.width_px =
        static_cast<int16_t>(cv::norm(result.corners[1] - result.corners[0]));
  }

  return result.detected;
}