
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
#include "maix_uart.hpp"
#include "opencv2/freetype.hpp"
#include "opencv2/opencv.hpp"

// #include "opencv2/"

// using namespace std;
using namespace maix;

const int CAM_W = 640;
const int CAM_H = 320;
const int ALGO_W = 320;
const int ALGO_H = 240;

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

enum VisionMode {
  STATE_TRACK_BOX,
  STATE_TRACK_EXIT,
  STATE_IDLE,
  STATE_FINISHED
};
VisionMode current_state = STATE_IDLE;

maix::peripheral::uart::UART uart("/dev/ttyS0", 11520);

void send_single_byte(uint8_t cmd) {
  uart.write(&cmd, 1);
  log::info("[UART TX] Sent: 0x%02X", cmd);
}

void send_data_packet(int16_t x, int16_t y, int16_t d) {
  uint8_t buf[6];
  memcpy(&buf[0], &x, 2);
  memcpy(&buf[2], &y, 2);
  memcpy(&buf[4], &d, 2);
  uart.write(buf, 6);
}

bool process_l_marker(cv::Mat& frame_small, int16_t& out_x, int16_t& out_y,
                      int16_t& out_d) {
  auto results = l_marker_detector.detectMultiple(frame_small);

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
  out_x = (int16_t)(center_u - (ALGO_W / 2.0f));
  out_y = (int16_t)(center_v - (ALGO_H / 2.0f));

  if (l_marker_pose.solve(pts, L_MARKER_3D_POINTS)) {
    cv::Vec3d tvec = l_marker_pose.getTvec();
    int16_t distance_mm = (int16_t)(cv::norm(tvec) * 1000.0f);

    // 画个坐标轴和中心点方便调试
    l_marker_pose.drawAxis(frame_small, 0.04f);
    cv::circle(frame_small, cv::Point(center_u, center_v), 4,
               cv::Scalar(0, 0, 255), -1);

    // 打印调试
    // log::info("[L-Marker] X:%d Y:%d D:%d mm", err_x, err_y, distance_mm);
    return true;
    // TODO: 电控串口发送
    // send_to_mcu(err_x, err_y, distance_mm);
  }
  return false;
}

bool process_exit_tag(cv::Mat& frame_hd, int16_t& out_x, int16_t& out_y,
                      int16_t& out_d) {
  ExitTagResult result;
  if (!exit_tag_recognizer.detect(frame_hd, frame_hd.cols, frame_hd.rows,
                                  result)) {
    return false;
  }

  for (int i = 0; i < 4; i++) {
    cv::line(frame_hd, result.corners[i], result.corners[(i + 1) % 4],
             cv::Scalar(0, 255, 0), 2);
  }
  cv::circle(frame_hd, result.center, 4, cv::Scalar(0, 0, 255), -1);
  // cv::line(frame, cv::Point(IMG_WIDTH / 2, 0),
  //          cv::Point(IMG_WIDTH / 2, IMG_HEIGHT), cv::Scalar(200, 200, 200),
  //          1);

  // 打印调试 (W 是像素宽度，越宽说明靠得越近)
  // log::info("[Exit-Tag] X:%d Y:%d W:%d", result.err_x, result.err_y,
  //           result.width_px);

  out_x = result.err_x;
  out_y = result.err_y;
  out_d = result.width_px;
  return true;
  // TODO: 电控串口发送
  // send_to_mcu(result.err_x, result.err_y, result.width_px);
}

int _main(int argc, char* argv[]) {
  log::info("Vision System Started. Waiting for commands ...");

  // 分辨率必须开到 640x480
  camera::Camera cam(CAM_W, CAM_H);

  while (true) {
    uint8_t cmd = 0;

    if (uart.read(&cmd, 1) > 0) {
      if (cmd == 0xFF) {
        send_single_byte(0xFE);
        current_state = STATE_IDLE;
        log::info("Received 0xFF: start tracking box.");
      } else if (cmd == 0xFD) {
        send_single_byte(0xFE);
        current_state = STATE_TRACK_EXIT;
        log::info("Received 0xFD: switch to exit tag.");
      }
    }

    if (current_state == STATE_IDLE) {
      usleep(10000);
      continue;
    }

    image::Image* img = cam.read();

    if (!img) continue;

    cv::Mat frame_hd;
    image::image2cv(*img, frame_hd);

    int16_t send_x = 0, send_y = 0, send_d = 0;

    bool detected = false;

    if (current_state == STATE_TRACK_BOX) {
      cv::Mat frame_small;
      cv::resize(frame_hd, frame_small, cv::Size(ALGO_W, ALGO_H));
      detected = process_l_marker(frame_small, send_x, send_y, send_d);
    } else if (current_state == STATE_TRACK_EXIT) {
      detected = process_exit_tag(frame_hd, send_x, send_y, send_d);
    }

    if (detected) {
      log::info("[State %d] X:%d Y:%d D:%d", current_state, send_x, send_y,
                send_d);

      send_data_packet(send_x, send_y, send_d);
      
      if (current_state == STATE_TRACK_EXIT) {
        send_single_byte(0xFC);
        current_state = STATE_IDLE;
        log::info("Exit Tag found. Send 0xFC, entering IDLE");
      }
    }
    delete img;
    usleep(30000);
  }
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