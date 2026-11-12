#include "MarkerDetector.h"

#include <string>

#include "maix_basic.hpp"

cv::Point2f getContourCenter(const std::vector<cv::Point>& contour) {
  cv::Moments m = cv::moments(contour);
  if (m.m00 != 0) {
    return cv::Point2f(static_cast<float>(m.m10 / m.m00),
                       static_cast<float>(m.m01 / m.m00));
  }
  cv::Rect rect = cv::boundingRect(contour);
  return cv::Point2f(rect.x + rect.width / 2.0f, rect.y + rect.height / 2.0f);
}

MarkerDetector::MarkerDetector() {}

// 预处理图像：灰度化、二值化、形态学操作
cv::Mat MarkerDetector::preprocess(const cv::Mat& frame) {
  if (frame.empty()) {
    return cv::Mat();
  }

  cv::Mat gray;
  // 1. 预处理：转灰度
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

  // 2. 二值化：因为Marker是发光的，提取高亮区域
  // 阈值(200)需要根据实际视频调整
  cv::inRange(gray, cv::Scalar(200), cv::Scalar(255), gray);

  // 3. 形态学操作：闭运算将四个分离的L型连接成一个整体，再开运算去除噪点
  // 补充：这里连接效果不好，所以改为只进行开运算，去除噪点
  cv::Mat morphed;
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  // cv::morphologyEx(gray, gray, cv::MORPH_CLOSE, kernel);
  cv::morphologyEx(gray, morphed, cv::MORPH_OPEN, kernel);

  return morphed;
}

// 检测Marker
// MarkerResult MarkerDetector::detect(const cv::Mat& frame) {
//   MarkerResult result;

//   if (frame.empty()) {
//     return result;
//   }

//   // (1) 预处理：灰度化、二值化、形态学操作
//   cv::Mat preprocessed_img = preprocess(frame);
//   if (preprocessed_img.empty()) {
//     return result;
//   }

//   std::vector<std::vector<cv::Point>> contours;
//   cv::findContours(preprocessed_img, contours, cv::RETR_EXTERNAL,
//                    cv::CHAIN_APPROX_SIMPLE);

//   std::vector<cv::Point2i> all_l_points;

//   double min_area = 100.0;  // 最小面积阈值，根据实际情况调整

//   // 4. 寻找轮廓
//   // 5. 几何筛选：多边形拟合，判断是否为四边形
//   for (const auto& contour : contours) {
//     double area = cv::contourArea(contour);
//     if (area < min_area) {
//       continue;  // 面积太小，忽略
//     }

//     cv::Rect rect = cv::boundingRect(contour);

//     float aspect_ratio = static_cast<float>(rect.width) / rect.height;

//     if (aspect_ratio < 0.5 || aspect_ratio > 2.0) {
//       continue;  // 长宽比不符合要求，忽略
//     }

//     for (const auto& point : contour) {
//       all_l_points.push_back(point);
//     }
//   }
//   if (all_l_points.size() > 20) {
//     cv::Rect bouding_box = cv::boundingRect(all_l_points);

//     cv::RotatedRect rotated_rect = cv::minAreaRect(all_l_points);

//     cv::Point2f rect_points[4];
//     rotated_rect.points(rect_points);

//     result.detected_ = true;
//     result.bounding_box_ = bouding_box;

//     std::vector<cv::Point2f> pts(rect_points, rect_points + 4);
//     result.points_ = sortPoints(pts);
//   }
//   return result;
// }

std::vector<MarkerResult> MarkerDetector::detectMultiple(const cv::Mat& frame) {
  std::vector<MarkerResult> results;
  if (frame.empty()) {
    return results;
  }

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(preprocess(frame), contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  std::vector<std::vector<cv::Point>> l_contours;
  std::vector<cv::Point2f> l_centers;

  std::vector<int> l_sizes;

  double min_l_area = 30.0;  // 最小面积阈值，根据实际情况调整

  for (const auto& contour : contours) {
    double area = cv::contourArea(contour);
    // maix::log::info("Contour area: %f", area);
    if (area < min_l_area) {
      continue;  // 面积太小，忽略
    }

    l_contours.push_back(contour);
    l_centers.push_back(getContourCenter(contour));
    cv::Rect contour_bbox = cv::boundingRect(contour);
    l_sizes.push_back(std::max(contour_bbox.width, contour_bbox.height));
  }

  std::vector<bool> used(l_contours.size(), false);

  double base_threshold = 0.25 * std::min(frame.cols, frame.rows);
  for (size_t i = 0; i < l_contours.size(); ++i) {
    if (used[i]) continue;

    std::vector<size_t> group_indices;
    group_indices.push_back(i);
    std::vector<cv::Point2f> group_points(l_contours[i].begin(),
                                          l_contours[i].end());

    for (size_t j = 0; j < l_contours.size(); ++j) {
      if (i == j || used[j]) continue;
      double cluster_distance_threshold =
          std::max(base_threshold, 3.5 * std::max(l_sizes[i], l_sizes[j]));
      double distance = cv::norm(l_centers[i] - l_centers[j]);
      if (distance < cluster_distance_threshold) {
        group_indices.push_back(j);
        group_points.insert(group_points.end(), l_contours[j].begin(),
                            l_contours[j].end());
      }
    }
    // maix::log::info("Current cluster: l_count = %d", l_count);
    if (group_indices.size() < 4 || group_points.size() < 20) {
      continue;
    }

    cv::Rect bounding_box = cv::boundingRect(group_points);
    cv::RotatedRect rotated_rect = cv::minAreaRect(group_points);
    if (rotated_rect.size.width < 10 || rotated_rect.size.height < 10) {
      continue;  // 忽略过小的标记
    }

    float aspect_ratio = rotated_rect.size.width / rotated_rect.size.height;
    if (aspect_ratio < 0.85f || aspect_ratio > 1.15f) {
      continue;  // 长宽比不符合要求，忽略
    }
    // maix::log::info("Aspect ratio: %f", aspect_ratio);

    // std::vector<cv::Point> hull;
    // cv::convexHull(group_points, hull);
    // float hull_area = cv::contourArea(hull);
    // if (rotated_rect.size.width * rotated_rect.size.height > hull_area
    // * 1.5f) {
    //   continue;  // 凸包面积与最小外接矩形面积差异过大，忽略
    // }
    for (size_t idx : group_indices) {
      used[idx] = true;
    }
    cv::Point2f rect_points[4];
    rotated_rect.points(rect_points);
    MarkerResult result;
    result.detected_ = true;
    result.bounding_box_ = bounding_box;
    std::vector<cv::Point2f> pts(rect_points, rect_points + 4);
    result.points_ = sortPoints(pts);
    results.push_back(result);
  }

  return results;
}

// 将四个点按顺时针排序，左上、右上、右下、左下
std::vector<cv::Point2f> MarkerDetector::sortPoints(
    const std::vector<cv::Point2f>& pts) {
  std::vector<cv::Point2f> sorted(4);
  cv::Point2f center(0, 0);

  for (const auto& pt : pts) {
    center += pt;
  }
  center *= 0.25f;

  // for (const auto& pt : pts) {
  //   if (pt.x < center.x && pt.y < center.y) {
  //     sorted[0] = pt;  // 左上
  //   } else if (pt.x > center.x && pt.y < center.y) {
  //     sorted[1] = pt;  // 右上
  //   } else if (pt.x > center.x && pt.y > center.y) {
  //     sorted[2] = pt;  // 右下
  //   } else if (pt.x < center.x && pt.y > center.y) {
  //     sorted[3] = pt;  // 左下
  //   }
  // }
  std::vector<std::pair<double, cv::Point2f>> angle_pts;
  for (const auto& pt : pts) {
    double angle = std::atan2(pt.y - center.y, pt.x - center.x);
    angle_pts.push_back(std::make_pair(angle, pt));
  }

  std::sort(angle_pts.begin(), angle_pts.end(),
            [](const std::pair<double, cv::Point2f>& a,
               const std::pair<double, cv::Point2f>& b) {
              return a.first < b.first;
            });

  int start_index = 0;
  float min_sum = 1e6;
  for (int i = 0; i < 4; ++i) {
    float sum = angle_pts[i].second.x + angle_pts[i].second.y;
    if (sum < min_sum) {
      min_sum = sum;
      start_index = i;
    }
  }
  for (int i = 0; i < 4; ++i) {
    sorted[i] = angle_pts[(start_index + i) % 4].second;
  }
  return sorted;
}
