
#include "main.h"

#include <unistd.h>

#include "ExitTagRecognizer.h"
#include "PoseEstimator.h"
#include "VideoProcessor.h"
#include "maix_basic.hpp"
#include "maix_camera.hpp"
#include "maix_display.hpp"
#include "maix_image.hpp"
#include "maix_image_cv.hpp"
#include "opencv2/freetype.hpp"
#include "opencv2/opencv.hpp"

// #include "opencv2/"

using namespace std;
using namespace maix;

uint64_t t = 0;
#define t_start() t = time::ticks_us()
#define t_end(msg)                                          \
  do {                                                      \
    log::info("%s cost: %d us", msg, time::ticks_us() - t); \
  } while (0)
#define t_end_start(msg)                                    \
  do {                                                      \
    log::info("%s cost: %d us", msg, time::ticks_us() - t); \
    t_start();                                              \
  } while (0)

const int IMG_W = 320;
const int IMG_H = 240;
const int HD_W = 640;
const int HD_H = 480;

const float MARKER_SIZE_M = 0.08f;
const float HALF_SIZE = MARKER_SIZE_M / 2.0f;

const std::vector<cv::Point3f> L_MARKER_3D_POINTS = {
    {-HALF_SIZE, -HALF_SIZE, 0},
    {HALF_SIZE, -HALF_SIZE, 0},
    {HALF_SIZE, HALF_SIZE, 0},
    {-HALF_SIZE, HALF_SIZE, 0}};

// 1. 准备临时相机内参（由于未标定，使用按 320x240 估算的假内参）
cv::Mat camera_matrix =
    (cv::Mat_<double>(3, 3) << 457.0, 0., 160.0, 0., 457.0, 120.0, 0., 0., 1.);

// 畸变系数继续保持为 0 即可（未标定时，忽略畸变更安全）
cv::Mat dist_coeffs = (cv::Mat_<double>(1, 5) << 0, 0, 0, 0, 0);

MarkerDetector l_marker_detector;
PoseEstimator l_marker_pose(camera_matrix, dist_coeffs);
ExitTagRecognizer exit_tag_recognizer;

enum VisionMode { MODE_L_MARKER, MODE_EXIT_TAG, MODE_IDLE };
VisionMode current_mode = MODE_EXIT_TAG;

bool process_l_marker(cv::Mat& frame) {
  auto results = l_marker_detector.detectMultiple(frame);

  if (results.empty() || !results[0].detected_ ||
      results[0].points_.size() != 4) {
    return false;
  }

  auto& pts = results[0].points_;

  float sum_u = 0, sum_v = 0;
  for (const auto& pt : pts) {
    sum_u += pt.x;
    sum_v += pt.y;
  }
  float center_u = sum_u / 4.0f;
  float center_v = sum_v / 4.0f;
  int16_t err_x = (int16_t)(center_u - (IMG_W / 2.0f));
  int16_t err_y = (int16_t)(center_v - (IMG_H / 2.0f));

  if (l_marker_pose.solve(pts, L_MARKER_3D_POINTS)) {
    cv::Vec3d tvec = l_marker_pose.getTvec();
    int16_t distance_mm = (int16_t)(cv::norm(tvec) * 1000.0f);

    // 画个坐标轴和中心点方便调试
    l_marker_pose.drawAxis(frame, 0.04f);
    cv::circle(frame, cv::Point(center_u, center_v), 4, cv::Scalar(0, 0, 255),
               -1);

    // 打印调试
    log::info("[L-Marker] X:%d Y:%d D:%d mm", err_x, err_y, distance_mm);
    return true;
    // TODO: 电控串口发送
    // send_to_mcu(err_x, err_y, distance_mm);
  }
  return false;
}

bool process_exit_tag(cv::Mat& frame) {
  ExitTagResult result;
  if (!exit_tag_recognizer.detect(frame, frame.cols, frame.rows, result)) {
    return false;
  }

  for (int i = 0; i < 4; i++) {
    cv::line(frame, result.corners[i], result.corners[(i + 1) % 4],
             cv::Scalar(0, 255, 0), 2);
  }
  cv::circle(frame, result.center, 4, cv::Scalar(0, 0, 255), -1);
  // cv::line(frame, cv::Point(IMG_WIDTH / 2, 0),
  //          cv::Point(IMG_WIDTH / 2, IMG_HEIGHT), cv::Scalar(200, 200, 200),
  //          1);

  // 打印调试 (W 是像素宽度，越宽说明靠得越近)
  log::info("[Exit-Tag] X:%d Y:%d W:%d", result.err_x, result.err_y,
            result.width_px);
  return true;
  // TODO: 电控串口发送
  // send_to_mcu(result.err_x, result.err_y, result.width_px);
}

int _main(int argc, char* argv[]) {
  log::info("Vision System Started.");

  // 分辨率必须开到 640x480
  camera::Camera cam(HD_W, HD_H);

  int saved_count = 0;
  int frame_count = 0;

  while (frame_count < 300) {
    image::Image* img = cam.read();
    if (!img) break;

    cv::Mat frame_hd;
    image::image2cv(*img, frame_hd);  // 得到 640x480 大图

    bool detected = false;
    cv::Mat frame_to_save;

    // 调度器
    if (current_mode == MODE_L_MARKER) {
      cv::Mat frame_small;
      cv::resize(frame_hd, frame_small, cv::Size(IMG_W, IMG_H));  // 缩小给算法
      detected = process_l_marker(frame_small);
      frame_to_save = frame_small;  // 如果要存 L 型的图，存小图即可
    } else if (current_mode == MODE_EXIT_TAG) {
      detected = process_exit_tag(frame_hd);
      frame_to_save = frame_hd;  // AprilTag 存高清大图
    }

    // ==========================================
    // 统一存图逻辑
    // ==========================================
    if (detected && saved_count < 5) {
      std::string prefix =
          (current_mode == MODE_L_MARKER) ? "l_marker_" : "exit_tag_";
      std::string out_path =
          "/root/" + prefix + std::to_string(saved_count) + ".jpg";

      cv::imwrite(out_path, frame_to_save);
      saved_count++;
      log::info("Saved test image: %s", out_path.c_str());
      usleep(300000);  // 存图后歇 300ms，方便你移动纸张
    }

    delete img;
    frame_count++;
    usleep(30000);  // 限制 ~30fps
  }

  log::info("Test done. Saved %d images.", saved_count);
  return 0;
}

int main(int argc, char* argv[]) {
  // Catch signal and process
  sys::register_default_signal_handle();

  // Use CATCH_EXCEPTION_RUN_RETURN to catch exception,
  // if we don't catch exception, when program throw exception, the objects will
  // not be destructed. So we catch exception here to let resources be
  // released(call objects' destructor) before exit.
  CATCH_EXCEPTION_RUN_RETURN(_main, -1, argc, argv);
}

/*
    0xFF

x 2 Byte
y 2 Byte
d 2 Byte
...
    0xFE

6n + 2 Byte
*/

/*
0x01 识别 兑换标志
0x02 识别 退出兑换标志
0x03 退出
*/

/*

*/