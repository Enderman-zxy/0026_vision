#include "Visualizer.h"

Visualizer::Visualizer() {}

// 绘制标记的检测结果，包括四个角点、边界框和角点编号
void Visualizer::drawMarker(cv::Mat& frame, const MarkerResult& result) {
  if (frame.empty()) {
    return;
  }
  // 如果未检测到目标，画一个提示文字，直接返回
  if (!result.detected_) {
    cv::putText(frame, "Target Lost", cv::Point(30, 50), font_face_, 1.0,
                cv::Scalar(0, 0, 255), 2);
    return;
  }

  // 绘制外接框 (Bounding Box)

  if (result.points_.size() == 4) {
    for (int i = 0; i < 4; i++) {
      cv::line(frame, result.points_[i], result.points_[(i + 1) % 4],
               cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    cv::circle(frame, result.points_[0], 5, color_lt_, -1);  // 左上角
    cv::putText(frame, "LT", result.points_[0] + cv::Point2f(10, -10),
                font_face_, font_scale_, color_lt_, line_thickness_);

    cv::circle(frame, result.points_[1], 5, color_rt_, -1);  // 右上角
    cv::putText(frame, "RT", result.points_[1] + cv::Point2f(10, -10),
                font_face_, font_scale_, color_rt_, line_thickness_);

    cv::circle(frame, result.points_[2], 5, color_rb_, -1);  // 右下角
    cv::putText(frame, "RB", result.points_[2] + cv::Point2f(10, -10),
                font_face_, font_scale_, color_rb_, line_thickness_);

    cv::circle(frame, result.points_[3], 5, color_lb_, -1);  // 左下角
    cv::putText(frame, "LB", result.points_[3] + cv::Point2f(10, -10),
                font_face_, font_scale_, color_lb_, line_thickness_);

    cv::Point2f center(0, 0);

    for (const auto& point : result.points_) {
      center += point;
    }
    center *= 0.25;                                             // 平均值
    cv::circle(frame, center, 5, cv::Scalar(255, 0, 255), -1);  // 中心点
  }
}