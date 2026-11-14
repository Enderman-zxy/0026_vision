#include "ExitTagRecognizer.h"

#include <opencv2/opencv.hpp>

#include "maix_log.hpp"

using namespace maix;

namespace {

// ===== TAG36H11 模板 =======================================================
//
// 规则手册图 3-8「退出兑换区视觉标签（AprilTag TAG36H11）」的 6x6 数据位，
// 直接从手册那张图上采出来的（外圈 1 格是黑边，不是数据）。
//   110110
//   010111
//   111100
//   011000
//   101101
//   001001
//
// 如果现场贴的标签和图 3-8 不是同一张：
//   1) 让程序跑到这个标签前面，看日志里打印的
//   observed_bits（会打印前几次不匹配的）； 2) 把它按同样的 6x6
//   行优先布局填到下面这张表里（多张标签就多填几行）。
constexpr int kTagWhitelistCount = 1;
constexpr uint8_t kTagBits[kTagWhitelistCount][6] = {
    {0x36, 0x17, 0x3C, 0x18, 0x2D,
     0x09},  // 110110 / 010111 / 111100 / 011000 / 101101 / 001001
};

// 汉明距离上限。TAG36H11 家族内任意两个码字的最小汉明距离是 11，
// 所以取 6 既能容忍采样噪声/边缘模糊，又不会认错码字。
constexpr int kMaxHamming = 6;

// 候选四边形
constexpr double kMinAreaPx =
    1000.0;  // 太小解不出 6x6 格（640x480 下约 32px 见方）
constexpr float kAspectMin = 0.70f;  // 斜视时是梯形，别卡太死
constexpr float kAspectMax = 1.0f / 0.70f;
constexpr double kApproxEpsRatio = 0.02;

// 双线性采样
float sampleBilinear(const cv::Mat& gray, const cv::Point2f& p) {
  if (gray.empty()) return 0.0f;
  float x = std::min(std::max(p.x, 0.0f), gray.cols - 1.001f);
  float y = std::min(std::max(p.y, 0.0f), gray.rows - 1.001f);
  int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
  int x1 = std::min(x0 + 1, gray.cols - 1);
  int y1 = std::min(y0 + 1, gray.rows - 1);
  float fx = x - x0, fy = y - y0;
  float v00 = gray.at<uchar>(y0, x0), v10 = gray.at<uchar>(y0, x1);
  float v01 = gray.at<uchar>(y1, x0), v11 = gray.at<uchar>(y1, x1);
  return (v00 * (1 - fx) + v10 * fx) * (1 - fy) +
         (v01 * (1 - fx) + v11 * fx) * fy;
}

// 把四边形排成"左上 -> 右上 -> 右下 -> 左下"（顺时针，图像坐标 y 向下）。
// 必须保证方向一致，否则采样出来的位图案是镜像的，永远匹配不上。
bool orderQuadClockwise(const std::vector<cv::Point2f>& quad,
                        std::vector<cv::Point2f>& out) {
  if (quad.size() != 4) return false;
  out.assign(4, cv::Point2f(0, 0));
  cv::Point2f center(0, 0);
  for (const auto& p : quad) center += p;
  center *= 0.25f;

  std::vector<std::pair<double, cv::Point2f>> angle_pts;
  angle_pts.reserve(4);
  for (const auto& p : quad)
    angle_pts.push_back({std::atan2(p.y - center.y, p.x - center.x), p});
  std::sort(angle_pts.begin(), angle_pts.end(),
            [](const std::pair<double, cv::Point2f>& a,
               const std::pair<double, cv::Point2f>& b) {
              return a.first < b.first;
            });
  int start = 0;
  float best = 1e30f;
  for (int i = 0; i < 4; ++i) {
    float sum = angle_pts[i].second.x + angle_pts[i].second.y;
    if (sum < best) {
      best = sum;
      start = i;
    }
  }
  for (int i = 0; i < 4; ++i) out[i] = angle_pts[(start + i) % 4].second;
  return true;
}

// 6x6 位图顺时针旋转 90 度
void rotateBits90cw(uint8_t bits[6]) {
  uint8_t tmp[6];
  for (int r = 0; r < 6; ++r) {
    tmp[r] = 0;
    for (int c = 0; c < 6; ++c) {
      // new[r][c] = old[5-c][r]
      int bit = (bits[5 - c] >> (5 - r)) & 1;
      tmp[r] |= static_cast<uint8_t>(bit << (5 - c));
    }
  }
  for (int r = 0; r < 6; ++r) bits[r] = tmp[r];
}

int hamming6x6(const uint8_t a[6], const uint8_t b[6]) {
  int d = 0;
  for (int r = 0; r < 6; ++r) d += __builtin_popcount(a[r] ^ b[r]);
  return d;
}

uint64_t packBits(const uint8_t bits[6]) {
  uint64_t v = 0;
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      int bit = (bits[r] >> (5 - c)) & 1;
      v = (v << 1) | static_cast<uint64_t>(bit);
    }
  }
  return v;
}

// 每帧最多打印几次"几何像标签但位图案不匹配"的观测值，方便现场换模板
int g_bit_report_budget = 5;

}  // namespace

ExitTagRecognizer::ExitTagRecognizer(float cx, float cy) : cx_(cx), cy_(cy) {}

bool ExitTagRecognizer::tryBinary(const cv::Mat& gray, const cv::Mat& binary,
                                  int img_w, int img_h, float zero_x,
                                  float zero_y, ExitTagResult& result) {
  std::vector<std::vector<cv::Point>> contours;

  cv::findContours(binary.clone(), contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  struct Candidate {
    double area;
    std::vector<cv::Point> poly;
  };
  std::vector<Candidate> candidates;

  for (const auto& contour : contours) {
    double area = cv::contourArea(contour);
    if (area < kMinAreaPx) continue;

    std::vector<cv::Point> poly;

    cv::approxPolyDP(contour, poly,
                     cv::arcLength(contour, true) * kApproxEpsRatio, true);
    if (poly.size() != 4) continue;

    // 用相邻边长均值判方，不依赖包围盒（斜视时包围盒会失真）
    double e0 = cv::norm(poly[0] - poly[1]);
    double e1 = cv::norm(poly[1] - poly[2]);
    double e2 = cv::norm(poly[2] - poly[3]);
    double e3 = cv::norm(poly[3] - poly[0]);

    double w_mean = (e0 + e2) * 0.5;
    double h_mean = (e1 + e3) * 0.5;
    if (h_mean < 1e-6) continue;
    double aspect = w_mean / h_mean;
    if (aspect < kAspectMin || aspect > kAspectMax) continue;

    candidates.push_back({area, poly});
  }

  std::sort(
      candidates.begin(), candidates.end(),
      [](const Candidate& a, const Candidate& b) { return a.area > b.area; });

  for (const auto& cand : candidates) {
    std::vector<cv::Point2f> raw;
    raw.reserve(4);
    for (const auto& p : cand.poly) raw.push_back(cv::Point2f(p));

    std::vector<cv::Point2f> quad;
    if (!orderQuadClockwise(raw, quad)) continue;

    // 单位方框 -> 图像。TAG36H11 的标准图是 8x8 格（外圈 1 格黑边 + 6x6
    // 数据）， 四边形的外沿就是这 8x8 的边界，所以数据格中心在 (i+1.5)/8。
    const cv::Point2f src[4] = {cv::Point2f(0.f, 0.f), cv::Point2f(1.f, 0.f),
                                cv::Point2f(1.f, 1.f), cv::Point2f(0.f, 1.f)};
    cv::Mat Hm = cv::getPerspectiveTransform(src, quad.data());
    if (Hm.empty()) continue;

    std::vector<cv::Point2f> cell_uv;
    cell_uv.reserve(64);
    for (int r = 0; r < 8; ++r)
      for (int c = 0; c < 8; ++c)
        cell_uv.push_back(cv::Point2f((c + 0.5f) / 8.0f, (r + 0.5f) / 8.0f));
    std::vector<cv::Point2f> cell_xy;
    cv::perspectiveTransform(cell_uv, cell_xy, Hm);

    // 每个格子取中心 + 4 个偏移点的均值，抗压缩噪声
    std::vector<float> vals(64, 0.0f);
    float vmin = 1e9f, vmax = -1e9f;
    for (int i = 0; i < 64; ++i) {
      const cv::Point2f& center = cell_xy[static_cast<size_t>(i)];
      cv::Point2f du =
          (cell_xy[static_cast<size_t>(i % 8 == 7 ? i - 1 : i + 1)] - center);
      cv::Point2f dv =
          (cell_xy[static_cast<size_t>(i + 8 < 64 ? i + 8 : i - 8)] - center);
      float dux = du.x * 0.18f, duy = du.y * 0.18f;
      float dvx = dv.x * 0.18f, dvy = dv.y * 0.18f;
      float v = sampleBilinear(gray, center);
      v += sampleBilinear(gray, cv::Point2f(center.x + dux, center.y + duy));
      v += sampleBilinear(gray, cv::Point2f(center.x - dux, center.y - duy));
      v += sampleBilinear(gray, cv::Point2f(center.x + dvx, center.y + dvy));
      v += sampleBilinear(gray, cv::Point2f(center.x - dvx, center.y - dvy));
      vals[static_cast<size_t>(i)] = v * 0.2f;
      vmin = std::min(vmin, vals[static_cast<size_t>(i)]);
      vmax = std::max(vmax, vals[static_cast<size_t>(i)]);
    }
    // 对比度太低就不是标签（白墙、暗块）
    if (vmax - vmin < 30.0f) continue;

    const float thr = 0.5f * (vmin + vmax);
    uint8_t grid[8][8];
    for (int r = 0; r < 8; ++r)
      for (int c = 0; c < 8; ++c)
        grid[r][c] = (vals[static_cast<size_t>(r * 8 + c)] > thr) ? 1 : 0;

    // 外圈必须是黑的（TAG36H11 的黑边），最多容忍 2 格
    int border_bad = 0;
    for (int i = 0; i < 8; ++i) {
      border_bad += grid[0][i];
      border_bad += grid[7][i];
      border_bad += grid[i][0];
      border_bad += grid[i][7];
    }
    if (border_bad > 2) continue;

    // 取 6x6 数据区，4 个旋转方向都和模板比一遍
    uint8_t data[6];
    for (int r = 0; r < 6; ++r) {
      data[r] = 0;
      for (int c = 0; c < 6; ++c)
        data[r] |= static_cast<uint8_t>(grid[r + 1][c + 1] << (5 - c));
    }

    int best_hamming = 99;
    int best_tag = -1;
    int best_rot = 0;
    uint8_t rotated[6];
    for (int tag = 0; tag < kTagWhitelistCount; ++tag) {
      for (int r = 0; r < 6; ++r) rotated[r] = data[r];
      for (int rot = 0; rot < 4; ++rot) {
        int d = hamming6x6(rotated, kTagBits[tag]);
        if (d < best_hamming) {
          best_hamming = d;
          best_tag = tag;
          best_rot = rot;
        }
        rotateBits90cw(rotated);
      }
    }

    if (best_hamming > kMaxHamming || best_tag < 0) {
      if (g_bit_report_budget > 0) {
        --g_bit_report_budget;
        log::info(
            "[ExitTag] geometry ok but pattern mismatch (hamming=%d) "
            "bits=0x%010llx",
            best_hamming, static_cast<unsigned long long>(packBits(data)));
      }
      continue;
    }

    // 通过：填结果
    result.detected = true;
    result.id = best_tag;
    result.rotation = best_rot;
    result.observed_bits = packBits(data);
    result.corners = quad;

    // 透视下的真实中心：把 (0.5,0.5) 映射过去
    std::vector<cv::Point2f> uv{cv::Point2f(0.5f, 0.5f)};
    std::vector<cv::Point2f> xy;
    cv::perspectiveTransform(uv, xy, Hm);
    result.center = xy[0];

    result.err_x = static_cast<int16_t>(result.center.x - zero_x);
    result.err_y = static_cast<int16_t>(result.center.y - zero_y);

    // "距离"仍用平均边长（电控按状态解释这个字段，含义不变）
    result.width_px = static_cast<int16_t>(
        0.25 * (cv::norm(quad[1] - quad[0]) + cv::norm(quad[2] - quad[1]) +
                cv::norm(quad[3] - quad[2]) + cv::norm(quad[0] - quad[3])));
    (void)img_w;
    (void)img_h;
    return true;
  }

  return false;
}

bool ExitTagRecognizer::detect(cv::Mat& frame, int img_w, int img_h,
                               ExitTagResult& result) {
  result = ExitTagResult();
  result.detected = false;
  result.id = -1;
  if (frame.empty()) return false;

  cv::Mat gray;
  if (frame.channels() == 3) {
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = frame;
  }

  // [M3] 零点用相机主点（光学轴），和 MARK 灯那条链路保持一致；
  //      没传主点时退回图像几何中心。
  const float zero_x = (cx_ >= 0.f) ? cx_ : (img_w / 2.0f);
  const float zero_y = (cy_ >= 0.f) ? cy_ : (img_h / 2.0f);

  // [H6] 旧实现用固定的 THRESH_BINARY_INV(100)：场景一暗，背景和标签黑边一起
  //      变成前景，RETR_EXTERNAL 只返回"整幅画面"一个轮廓，永远检不到。
  //      这里有三级：先 Otsu（正常光照），再 Otsu+反相（浅底深标），
  //      最后局部自适应阈值（光照不均/局部过暗）。
  cv::Mat otsu_inv;
  cv::threshold(gray, otsu_inv, 0, 255,
                cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
  if (tryBinary(gray, otsu_inv, img_w, img_h, zero_x, zero_y, result))
    return true;

  cv::Mat adaptive;
  cv::adaptiveThreshold(gray, adaptive, 255, cv::ADAPTIVE_THRESH_MEAN_C,
                        cv::THRESH_BINARY_INV, 31, 5);
  if (tryBinary(gray, adaptive, img_w, img_h, zero_x, zero_y, result))
    return true;

  return false;
}