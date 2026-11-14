// 串口通信开关， 1=关闭串口，使用终端输出，0=开启串口
#define MOCK_UART 1
// 防抖开关，1=打开防抖，0=关闭防抖
#define ENABLE_DEBOUNCE 1
// 单位开关，1=x坐标单位为毫米，0=x坐标单位为像素
#define mm_or_px 0

// #include "main.h"

#include <unistd.h>

#include "ExitTagRecognizer.h"
#include "MarkerDetector.h"
#include "PoseEstimator.h"
// #include "VideoProcessor.h"
#include "maix_basic.hpp"
#include "maix_camera.hpp"
#include "maix_display.hpp"
#include "maix_image.hpp"
#include "maix_image_cv.hpp"
#include "maix_key.hpp"
#include "maix_uart.hpp"
#include "opencv2/freetype.hpp"
#include "opencv2/opencv.hpp"

// #include "opencv2/"

// using namespace std;
using namespace maix;

const int CAM_W = 640;
const int CAM_H = 480;
const int ALGO_W = 320;
const int ALGO_H = 240;

// MARK 灯外廓 80x80mm（规则手册图 5-8）。
// 注意：实测这个常数直接决定距离读数的比例，必须台架标定 ——
// 把灯放在已知距离 D 处，读 forward_mm，则
//     MARKER_SIZE_M_有效 = 0.08 * D / forward_mm
// （阈值/曝光会让亮区胖或瘦 0.5~1px，80mm 的目标在 50px 上就是 2~4%）。
const float MARKER_SIZE_M = 0.08f;
const float HALF_SIZE = MARKER_SIZE_M / 2.0f;

const std::vector<cv::Point3f> L_MARKER_3D_POINTS = {
    {-HALF_SIZE, -HALF_SIZE, 0},
    {HALF_SIZE, -HALF_SIZE, 0},
    {HALF_SIZE, HALF_SIZE, 0},
    {-HALF_SIZE, HALF_SIZE, 0}};

// 边沿补偿（像素，每个边往外扩多少）。
// 二值化 + 3x3 开运算会把发光区域的边啃掉一点，"目标看起来比实际小 ->
// 距离读大"。 默认 0（不动几何）。台架标定方法：正对 0.3m / 0.6m 各读一次
// forward_mm，
//   - 两处都偏大且远处偏得更多  -> 把这里调成正的（0.3~1.0 之间试）
//   - 两处都偏小（灯过曝外扩）  -> 调成负的
//   - 只是整体等比偏大          -> 改 MARKER_SIZE_M 更合适
const float kQuadEdgeGrowPx = 0.0f;

// ===== 相机安装外参 ========================================================
// [H8] 车上没有云台：相机和车体刚性固连，所以这组量是常量，装好车量一次填进来。
// 不改这组量也能跑，但 d 会把高度差和横向偏差一起算进去（斜距），
// 近距离误差可以到 +20%~+60%，而且"到位"判据永远达不到 0。
//
// 怎么量：
//   pitch_deg     拿水平仪/瞄准线：镜头中心对着 2m
//   外一个和相机等高的点，读数应为 0；
//                 若对着地面某点，量出"该点离地高度差 / 水平距离"取 atan。
//   cam_height_m  卷尺量镜头中心到地面。
//   cam_forward_m
//   卷尺量镜头中心到车体旋转中心（麦克纳姆轮几何中心）的前向距离，
//                 镜头在旋转中心前方为正。
CameraExtrinsics kCameraExtrinsics = {
    0.0,    // pitch_deg      光轴下俯角（度），正 = 向下看
    0.20,   // cam_height_m   相机光心离地高度（米）—— 必须实测
    0.135,  // target_height_m MARK 灯中心离地高度（米，规则手册推算，不用改）
    0.0,    // cam_forward_m  相机相对车体旋转中心的前移量（米）
};

// 相机参数
cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 249.7885517, 0., 162.208795,
                         0., 249.9703717, 115.9213142, 0., 0., 1.);
cv::Mat dist_coeffs = (cv::Mat_<double>(1, 5) << -0.00517909478, -0.00654534888,
                       0.00124095615, -0.0000318773159, 0.0925784765);

MarkerDetector l_marker_detector;
PoseEstimator l_marker_pose(camera_matrix, dist_coeffs, kCameraExtrinsics);
// 退出标志跑在 640x480 的 frame_hd 上，所以零点也要从 320x240 的内参换算过去
// （cx,cy 乘以 CAM_W/ALGO_W = 2）。[M3] 统一用主点当零点。
ExitTagRecognizer exit_tag_recognizer(
    static_cast<float>(camera_matrix.at<double>(0, 2) * CAM_W / ALGO_W),
    static_cast<float>(camera_matrix.at<double>(1, 2) * CAM_H / ALGO_H));

enum VisionMode {
  STATE_TRACK_BOX,
  STATE_TRACK_EXIT,
  STATE_IDLE,
  STATE_FINISHED
};
VisionMode current_state = STATE_IDLE;

#if ENABLE_DEBOUNCE
const int MIN_DETECTED_FRAMES = 3;
const int MAX_LOST_FRAMES = 5;

int box_detected_frames = 0;
int box_lost_frames = 0;
int16_t last_valid_x = 0, last_valid_y = 0, last_valid_d = 0;
bool box_is_stable = false;
#endif

// int16 上限保护：位姿偶尔解出离谱值时不要把负数当距离发出去
int16_t clamp_i16(double v) {
  if (v > 32767.0) return 32767;
  if (v < -32768.0) return -32768;
  return static_cast<int16_t>(std::lround(v));
}

#if MOCK_UART
int uart_read_mock() { return 0; }
void uart_write_mock(uint8_t* buf, int len) {
  char hex_str[128] = {0};
  for (int i = 0; i < len; i++) sprintf(hex_str + i * 3, "%02X ", buf[i]);
  log::info("[MOCK TX] %s", hex_str);
}
#else
maix::peripheral::uart::UART uart("/dev/ttyS0", 115200);
#endif

void send_single_byte(uint8_t cmd) {
#if MOCK_UART
  uart_write_mock(&cmd, 1);
#else
  uart.write(&cmd, 1);
  log::info("[UART TX] Sent: 0x%02X", cmd);
#endif
}

void send_data_packet(int16_t x, int16_t y, int16_t d) {
  uint8_t buf[6];
  memcpy(&buf[0], &x, 2);
  memcpy(&buf[2], &y, 2);
  memcpy(&buf[4], &d, 2);
#if MOCK_UART
  uart_write_mock(buf, 6);
#else
  uart.write(buf, 6);
#endif
}

int read_uart(uint8_t* cmd) {
#if MOCK_UART
  return 0;  // 假串口无输入
#else
  return uart.read(cmd, 1);
#endif
}

// 检测 MARK 灯并换算成"车体地面系"下的偏移量（x为像素，d为毫米）。
//   out_x : 目标在车体右侧多少（右为正）  ← 旧版是像素偏差
//   out_y : 固定 0（没有云台，垂直方向不可控，不作为控制量）
//   out_d : 目标到车体的水平距离（毫米）  ← 旧版是相机系斜距 |tvec|
// 成功时顺带把竖直残差写进 last_height_residual_mm（诊断用）。
double last_height_residual_mm = 0.0;

bool process_l_marker(cv::Mat& frame_small, int16_t& out_x, int16_t& out_y,
                      int16_t& out_d) {
  auto results = l_marker_detector.detectMultiple(frame_small);

  // if (results.empty() || !results[0].detected_ ||
  //     results[0].points_.size() != 4) {
  //   return false;
  // }

  if (results.empty()) return false;

  std::vector<size_t> order;
  order.reserve(results.size());

  for (size_t i = 0; i < results.size(); i++) {
    if (results[i].detected_ && results[i].points_.size() == 4) {
      order.push_back(i);
    }
  }
  if (order.empty()) return false;
  std::sort(order.begin(), order.end(), [&results](size_t a, size_t b) {
    return results[a].bounding_box_.area() > results[b].bounding_box_.area();
  });

  for (size_t idx : order) {
    auto& pts = results[idx].points_;

    std::vector<cv::Point2f> quad = pts;

    if (kQuadEdgeGrowPx != 0.0f) {
      cv::Point2f c(0, 0);
      for (const auto& p : quad) c += p;
      c *= 0.25f;
      double side = 0.0;
      for (int i = 0; i < 4; ++i) side += cv::norm(quad[i] - quad[(i + 1) % 4]);
      side *= 0.25;
      if (side > 1.0) {
        const float s = static_cast<float>(1.0 + 2.0 * kQuadEdgeGrowPx / side);
        for (auto& p : quad) p = c + (p - c) * s;
      }
    }

    if (!l_marker_pose.solve(quad, L_MARKER_3D_POINTS)) continue;

    double right_mm = 0.0, forward_mm = 0.0, height_residual_mm = 0.0;

    if (!l_marker_pose.getGroundOffset(right_mm, forward_mm,
                                       height_residual_mm)) {
      continue;
    }
    last_height_residual_mm = height_residual_mm;
#if mm_or_px
    const int16_t x_mm = clamp_i16(right_mm);
    out_x = x_mm;
#else
    float sum_u = 0;
    for (const auto& pt : quad) {
      sum_u += pt.x;
    }
    float center_u = sum_u * 0.25f;
    const double cx = camera_matrix.at<double>(0, 2);

    const int16_t x_px = clamp_i16((center_u - cx) * 2.0f);
    out_x = x_px;
#endif
    const int16_t y = 0;
    const int16_t d = clamp_i16(forward_mm);
    out_y = y;
    out_d = d;
    return true;
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

  // for (int i = 0; i < 4; i++) {
  //   cv::line(frame_hd, result.corners[i], result.corners[(i + 1) % 4],
  //            cv::Scalar(0, 255, 0), 2);
  // }
  // cv::circle(frame_hd, result.center, 4, cv::Scalar(0, 0, 255), -1);
  // cv::line(frame, cv::Point(IMG_WIDTH / 2, 0),
  //          cv::Point(IMG_WIDTH / 2, IMG_HEIGHT), cv::Scalar(200, 200, 200),
  //          1);

  // 打印调试 (W 是像素宽度，越宽说明靠得越近)
  // log::info("[Exit-Tag] X:%d Y:%d W:%d", result.err_x, result.err_y,
  //           result.width_px);

  // 注意：这个字段在 TRACK_EXIT 状态下是"标签像素宽度"（越大越近），
  // 和 TRACK_BOX 状态下的毫米不是同一个量纲，电控必须按状态解释。
  out_x = result.err_x;
  out_y = result.err_y;
  out_d = result.width_px;
  return true;
}

int _main(int argc, char* argv[]) {
  peripheral::key::Key key(nullptr, true);

#if MOCK_UART
  log::warn(
      "!!! MOCK_UART=1: 串口是假的，所有 0xFE/0xFC/6 字节包只会打到日志 !!!");
#endif

  log::info("Vision System Started. Waiting for commands ...");
  log::info("[Config] algo=%dx%d cam=%dx%d fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
            ALGO_W, ALGO_H, CAM_W, CAM_H, camera_matrix.at<double>(0, 0),
            camera_matrix.at<double>(1, 1), camera_matrix.at<double>(0, 2),
            camera_matrix.at<double>(1, 2));
  log::info(
      "[Config] extrinsics pitch=%.1fdeg cam_h=%.3fm target_h=%.3fm fwd=%.3fm",
      kCameraExtrinsics.pitch_deg, kCameraExtrinsics.cam_height_m,
      kCameraExtrinsics.target_height_m, kCameraExtrinsics.cam_forward_m);
  {
    // 分辨率必须开到 640x480
    camera::Camera cam(CAM_W, CAM_H);

    int frame_count = 0;

    while (!app::need_exit()) {
      int key_code = 0, key_state = 0;
      if (key.read(key_code, key_state) == err::Err::ERR_NONE) {
        if (key_code == maix::peripheral::key::KEY_OK &&
            key_state == maix::peripheral::key::KEY_PRESSED) {
          log::info("OK button pressed, exiting");
          break;
        }
      }

      uint8_t cmd = 0;
      int read_ret = read_uart(&cmd);

#if MOCK_UART
      // 假串口模式下，我们用帧数模拟电控发指令
      if (frame_count == 30) {
        cmd = 0xFF;
        read_ret = 1;
      }  // 第30帧模拟发 0xFF
      else if (frame_count == 200) {
        cmd = 0xFD;
        read_ret = 1;
      }  // 第200帧模拟发 0xFD

#endif
      if (read_ret > 0) {
        if (cmd == 0xFF) {
          send_single_byte(0xFE);
          current_state = STATE_TRACK_BOX;
#if ENABLE_DEBOUNCE
          box_detected_frames = 0;
          box_lost_frames = 0;
          box_is_stable = false;
#endif
          log::info("Received 0xFF: start tracking box.");
        } else if (cmd == 0xFD) {
          send_single_byte(0xFE);
          current_state = STATE_TRACK_EXIT;
          log::info("Received 0xFD: switch to exit tag.");
        }
      }

      if (current_state == STATE_IDLE) {
        // [M10] 空闲时也把帧取走丢掉：否则切到 TRACK 状态后 cam.read() 先返回
        // 队列里积压的旧帧（buff_num 默认 3），白白多 100ms 延迟。
        image::Image* idle_img = cam.read();
        if (idle_img) {
          delete idle_img;
        } else {
          usleep(5000);
        }
        // usleep(10000);
        frame_count++;
        continue;
      }

      image::Image* img = cam.read();
      if (!img) {
        usleep(10000);
        frame_count++;
        continue;
      }

      cv::Mat frame_hd;
      // [L1] 相机默认输出 RGB888，要让 OpenCV 拿到真正的 BGR
      image::image2cv(*img, frame_hd, true);

      int16_t send_x = 0, send_y = 0, send_d = 0;

      bool detected = false;

      if (current_state == STATE_TRACK_BOX) {
        cv::Mat frame_small;
        cv::resize(frame_hd, frame_small, cv::Size(ALGO_W, ALGO_H));
        detected = process_l_marker(frame_small, send_x, send_y, send_d);
#if ENABLE_DEBOUNCE
        if (detected) {
          box_detected_frames++;
          box_lost_frames = 0;

          last_valid_x = send_x;
          last_valid_y = send_y;
          last_valid_d = send_d;

          if (box_detected_frames >= MIN_DETECTED_FRAMES) {
            box_is_stable = true;
          }
        } else {
          box_detected_frames = 0;
          box_lost_frames++;

          if (box_lost_frames > MAX_LOST_FRAMES) {
            box_is_stable = false;
          }
        }
        detected = box_is_stable;
        send_x = last_valid_x;
        send_y = last_valid_y;
        send_d = last_valid_d;
#endif
      } else if (current_state == STATE_TRACK_EXIT) {
        detected = process_exit_tag(frame_hd, send_x, send_y, send_d);
      }

      if (detected) {
        log::info("[State %d] X:%d Y:%d D:%d", current_state, send_x, send_y,
                  send_d);

        // 外参自检：这个残差正常应接近 0；明显非 0 说明
        // kCameraExtrinsics.pitch_deg / cam_height_m 标错了（每 30 帧打一次）
        if (current_state == STATE_TRACK_BOX && frame_count % 30 == 0) {
          log::info("[Calib] height_residual=%.0f mm (应接近 0)",
                    last_height_residual_mm);
        }
        send_data_packet(send_x, send_y, send_d);

        if (current_state == STATE_TRACK_EXIT) {
          send_single_byte(0xFC);
          current_state = STATE_IDLE;
          log::info("Exit Tag found. Send 0xFC, entering IDLE");
        }
      }
      delete img;
      frame_count++;
      // [L3] 不再固定 sleep 30ms：cam.read() 本身就被相机帧率限制，
      // 固定 sleep 会把循环从 30fps 压到 ~20fps，兑换框 0.5m/s 时多走 10mm。
      // usleep(30000);
    }
  }
  usleep(2000000);
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