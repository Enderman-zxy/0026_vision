#pragma once

#include <cstdint>
#include <opencv2/opencv.hpp>

// #include "maix_image.hpp"

struct ExitTagResult {
  bool detected = false;
  int id = -1;
  int16_t err_x = 0;
  int16_t err_y = 0;
  int16_t width_px = 0;
  std::vector<cv::Point2f> corners;
  cv::Point2f center{0.f, 0.f};
  // 命中时的旋转（0~3，90 度步进），调试用
  int rotation = 0;
  // 采样到的 6x6=36 位（行优先，bit35 是左上角），现场标定/排错用
  uint64_t observed_bits = 0;

  ExitTagResult() {}
};

class ExitTagRecognizer {
 public:
  /**
   * @param cx 计算 err_x 用的零点（应传相机主点）；传负数则用图像几何中心
   * @param cy 同上
   */
  explicit ExitTagRecognizer(float cx = -1.f, float cy = -1.f);

  /**
   * @brief 识别图像中的"结束兑换视觉标签"（AprilTag TAG36H11）
   *
   * [H4] 旧实现只是"找一个黑方块 + 中心亮"，并不解码，也没有 ID 校验，
   *      所以无法把结束兑换标签、兑换框识别板和其他 RFID 标签区分开。
   *      现在按图 3-8 的 6x6 数据位做模板匹配（4 个旋转方向取最小汉明距离），
   *      并且校验外圈黑边。识别不到就是识别不到，宁缺毋滥。
   *
   * @param frame 输入图像（BGR 或灰度）
   * @param img_w 图像宽度 (用于计算中心)
   * @param img_h 图像高度
   * @param result 输出的识别结果
   * @return 是否成功识别
   */
  bool detect(cv::Mat& frame, int img_w, int img_h, ExitTagResult& result);

 private:
  bool tryBinary(const cv::Mat& gray, const cv::Mat& binary, int img_w,
                 int img_h, float zero_x, float zero_y, ExitTagResult& result);

  float cx_ = -1.f;  // err_x 的零点（相机主点 x），<0 表示用图像中心
  float cy_ = -1.f;  // err_y 的零点（相机主点 y）
};
