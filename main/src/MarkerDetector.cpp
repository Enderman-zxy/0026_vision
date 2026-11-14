#include "MarkerDetector.h"

#include <string>

#include "maix_basic.hpp"

namespace {

// ===== 判据参数（依据规则手册图 5-8 的 MARK 灯几何）=========================
//
// MARK 灯：外廓 80x80mm，4 个 L 各 30x30mm、条宽 8mm，相邻 L 的中心距 50mm，
// 中间是 20x20mm 的暗区（镜片）。下面这些常数都是按这张图定的，改灯就要重标。
//
// [H2] 两个亮斑中心距 < kClusterRatio * max(两斑尺寸) 才认为属于同一个灯。
//      几何依据：相邻 L 的中心距 50mm，L 外廓 30mm → 至少 1.67；阈值 200 +
//      开运算 会把这个 30mm 侵蚀到 16~17px（0.4m 处实测），所以取 2.4
//      才能稳定合上； 而一颗 42mm 矿石球离最近的 L 中心约 48px，2.4*17=41px
//      合不到它 —— 旧实现用的是固定 60px 地板，在 0.3~0.7m
//      这段比目标本身还大，直接吞掉。
constexpr float kClusterRatio = 2.4f;

// [H3] 外接矩形长宽比窗口。斜视时投影是梯形，长宽比必然偏离 1：
//      yaw=30deg 实测 1.24，35deg 约 1.40 ——
//      旧实现卡在 1.15，等于"只准正对着看"， 而没有云台时标靶的 yaw
//      就是车体航向误差，越需要修正时越看不见。 这里放宽到 +-40deg
//      左右（正对时为 1.0）。
constexpr float kAspectMax = 1.60f;
constexpr float kAspectMin = 1.0f / 1.60f;

// [H2] 凸包面积 / 最小外接矩形面积。MARK 灯 4 个 L 的外角正好落在外廓 4
// 个角上，
//      所以凸包 ≈ 外廓正方形，fill≈1；"一条线 / 单个 L / 两个斑"的簇 fill
//      会明显偏小。
constexpr float kMinFillRatio = 0.62f;

constexpr float kMinRectSidePx = 10.0f;  // 小于这个尺寸解不出稳定位姿
constexpr float kMinQuadSidePx = 8.0f;

// [H1] 不再用"必须有 4 个独立轮廓"当判据（旧实现一个 L 少 1 个像素²
// 就整帧丢），
//      改成看 MARK 灯自身的光学签名（图 5-8）：
//        1) 4 个象限里都有亮斑（4 个 L 分布在 4 个角上）；
//        2) 中心是暗的（中间 20x20mm 是镜片，不是亮面）；
//        3) 4 条边的中点都是暗的（相邻两个 L 之间留了 20mm 的缺口）。
//      这三条合起来把"一整块亮矩形 / 墙上反光 / 旁边多一颗矿石"挡在外面，
//      而且不再依赖轮廓个数，单帧鲁棒性由"结构"而不是"计数"保证。
constexpr int kMinBlobsPerQuadrant = 1;
// 4 个 L 的形心离方框中心 ≈0.74 个半边长（L 的形心在距外角 10.35mm 处）。
// 实测视频里是 0.79~0.82（阈值+开运算会把 8mm 细臂啃掉一点，形心外移）。
// 这个区间把"旁边多一颗矿石把方框撑大"的情况挡掉：方框被撑大后，真正的 4 个 L
// 只落在 0.4~0.5 处，会掉出下界。
constexpr float kCornerSpreadMin = 0.55f;
constexpr float kCornerSpreadMax = 0.98f;
constexpr float kCenterBoxRatio = 0.22f;    // 中心取样框占四边形的比例
constexpr float kMaxCenterBright = 0.50f;   // 中心框内亮像素占比上限
constexpr float kMaxEdgeMidBright = 0.35f;  // 边中点缺口内亮像素占比上限

// 灯是自发光、会过曝到 255，所以阈值下限取 200；Otsu 更高时用 Otsu。
// 这样环境变亮（白天/补光）时阈值跟着抬高，不会把墙面当目标。
constexpr int kBrightFloor = 200;

// ===== 小工具 ==============================================================

int findRoot(std::vector<int>& parent, int x) {
  while (parent[x] != x) {
    parent[x] = parent[parent[x]];
    x = parent[x];
  }
  return x;
}

void unionSet(std::vector<int>& parent, int a, int b) {
  int ra = findRoot(parent, a);
  int rb = findRoot(parent, b);
  if (ra != rb) parent[rb] = ra;
}

// 把四边形按"左上角为第 0 点、顺时针"排好。旧实现的排序逻辑保留，只是把
// 起始点的并列情况处理成确定性的（45deg 时旧实现会在两个角之间跳）。
void orderQuadClockwise(const std::vector<cv::Point2f>& quad,
                        std::vector<cv::Point2f>& out) {
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
              return a.first < b.first;  // 图像坐标 y 向下 -> 递增即顺时针
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
}

// 沿四边形某条边、在中点附近 20% 的长度上采样，返回"亮"的比例。
// MARK 灯相邻两个 L 之间在这里留了 20mm 缺口，所以正常应该是暗的。
float edgeMidBrightRatio(const cv::Mat& mask, const cv::Point2f& a,
                         const cv::Point2f& b) {
  int bright = 0, total = 0;
  const int kSamples = 9;
  for (int i = 0; i < kSamples; ++i) {
    float t = 0.40f + 0.20f * (i + 0.5f) / kSamples;
    cv::Point2f p = a + (b - a) * t;
    int x = static_cast<int>(std::lround(p.x));
    int y = static_cast<int>(std::lround(p.y));
    if (x < 0 || y < 0 || x >= mask.cols || y >= mask.rows) continue;
    ++total;
    if (mask.at<uchar>(y, x) > 0) ++bright;
  }
  if (total == 0) return 1.0f;
  return static_cast<float>(bright) / static_cast<float>(total);
}

struct Blob {
  std::vector<cv::Point> contour;
  cv::Point2f center;
  float size = 0.0f;  // 外接框长边，用于"按尺度聚类"
};

// 用给定的若干块亮斑去拟合一个 MARK 灯，并按"灯自身的光学签名"验收。
// 验收不通过时返回 false，调用方会剔除离群亮斑后重试。
bool fitMarker(const std::vector<Blob>& blobs, const std::vector<int>& members,
               const cv::Mat& mask, MarkerResult& out) {
  if (members.size() < 4) return false;

  std::vector<cv::Point> group_points;
  std::vector<cv::Point2f> centers;
  group_points.reserve(members.size() * 8);
  centers.reserve(members.size());
  for (int idx : members) {
    const Blob& b = blobs[idx];
    group_points.insert(group_points.end(), b.contour.begin(), b.contour.end());
    centers.push_back(b.center);
  }
  if (group_points.size() < 20) return false;

  cv::Rect bounding_box = cv::boundingRect(group_points);
  cv::RotatedRect rotated_rect = cv::minAreaRect(group_points);
  float rw = rotated_rect.size.width;
  float rh = rotated_rect.size.height;
  if (rw < kMinRectSidePx || rh < kMinRectSidePx) return false;
  if (rh > rw) std::swap(rw, rh);

  // [H3] 长宽比窗口放宽（旧的是 0.85~1.15）
  float aspect_ratio = rw / rh;
  if (aspect_ratio > kAspectMax || aspect_ratio < kAspectMin) return false;

  // 凸包填充率：排除"细长 / 只有两条边"的簇
  std::vector<cv::Point> hull;
  cv::convexHull(group_points, hull);
  double hull_area = cv::contourArea(hull);
  double rect_area =
      static_cast<double>(rotated_rect.size.width) * rotated_rect.size.height;
  if (rect_area <= 1.0) return false;
  if (hull_area / rect_area < kMinFillRatio) return false;

  // 四边形的 4 个角点：优先用凸包的多边形拟合（斜视时是真正的投影角点，
  // 比 minAreaRect 的外接矩形偏置小），拿不到 4 点再退回 minAreaRect。
  std::vector<cv::Point> quad_poly;
  cv::approxPolyDP(hull, quad_poly, cv::arcLength(hull, true) * 0.03, true);
  std::vector<cv::Point2f> quad;
  if (quad_poly.size() == 4) {
    quad.reserve(4);
    for (const auto& p : quad_poly) quad.push_back(cv::Point2f(p));
  } else {
    cv::Point2f rect_points[4];
    rotated_rect.points(rect_points);
    quad.assign(rect_points, rect_points + 4);
  }

  bool quad_ok = true;
  for (int i = 0; i < 4; ++i) {
    if (cv::norm(quad[i] - quad[(i + 1) % 4]) < kMinQuadSidePx) {
      quad_ok = false;
      break;
    }
  }
  if (!quad_ok) return false;

  cv::Point2f qc(0, 0);
  for (const auto& p : quad) qc += p;
  qc *= 0.25f;

  // 四边形的两条边方向（用对边平均，透视下更稳）
  cv::Point2f axis_u = ((quad[1] - quad[0]) + (quad[2] - quad[3])) * 0.5f;
  cv::Point2f axis_v = ((quad[3] - quad[0]) + (quad[2] - quad[1])) * 0.5f;
  float nu = cv::norm(axis_u), nv = cv::norm(axis_v);
  if (nu < 1e-3f || nv < 1e-3f) return false;
  axis_u /= nu;
  axis_v /= nv;

  // [H1] 4 个象限都必须有亮斑中心。旧实现只数"轮廓个数 >= 4"，
  //      3 个 L + 1 个小方块或 1 颗矿石都能凑够 4，从而拟出错误的方框。
  int quadrant_count[2][2] = {{0, 0}, {0, 0}};
  for (const auto& c : centers) {
    cv::Point2f d = c - qc;
    int iu = (d.dot(axis_u) > 0) ? 1 : 0;
    int iv = (d.dot(axis_v) > 0) ? 1 : 0;
    quadrant_count[iu][iv]++;
  }
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      if (quadrant_count[i][j] < kMinBlobsPerQuadrant) return false;

  // [H2] 每个象限里要有一个"位置正确"的亮斑：形心落在半边长的
  //      [kCornerSpreadMin, kCornerSpreadMax] 区间里（正常 MARK 灯
  //      ≈0.74~0.82）。 干扰亮斑把方框撑大后，4 个 L 的形心会缩到
  //      0.4~0.5，这里会判否， 避免"框被撑大 → d 变小 →
  //      车多冲一段"这种静默错误。
  float half_u =
      0.5f * 0.5f * (cv::norm(quad[1] - quad[0]) + cv::norm(quad[2] - quad[3]));
  float half_v =
      0.5f * 0.5f * (cv::norm(quad[3] - quad[0]) + cv::norm(quad[2] - quad[1]));
  if (half_u < 1e-3f || half_v < 1e-3f) return false;
  bool corner_in_band[2][2] = {{false, false}, {false, false}};
  for (const auto& c : centers) {
    cv::Point2f d = c - qc;
    float du = d.dot(axis_u) / half_u;
    float dv = d.dot(axis_v) / half_v;
    float au = std::fabs(du), av = std::fabs(dv);
    if (au < kCornerSpreadMin || au > kCornerSpreadMax) continue;
    if (av < kCornerSpreadMin || av > kCornerSpreadMax) continue;
    corner_in_band[du > 0 ? 1 : 0][dv > 0 ? 1 : 0] = true;
  }
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      if (!corner_in_band[i][j]) return false;

  // [H1] 中心必须是暗的：MARK 灯中间是 20x20mm 的镜片，不是亮面。
  int bw = std::max(1, static_cast<int>(std::lround(rw * kCenterBoxRatio)));
  int bh = std::max(1, static_cast<int>(std::lround(rh * kCenterBoxRatio)));
  cv::Rect center_box(static_cast<int>(std::lround(qc.x)) - bw / 2,
                      static_cast<int>(std::lround(qc.y)) - bh / 2, bw, bh);
  center_box &= cv::Rect(0, 0, mask.cols, mask.rows);
  if (center_box.width > 0 && center_box.height > 0) {
    double bright_ratio = cv::mean(mask(center_box))[0] / 255.0;
    if (bright_ratio > kMaxCenterBright) return false;
  }

  // [H1] 4 条边的中点必须是暗的（相邻 L 之间的缺口）。
  //      干扰亮斑把外接方框撑大后，合成边会落到缺口之外，这里会判否，
  //      避免把撑大的方框拿去解位姿（那会让 d 变小）。
  for (int i = 0; i < 4; ++i) {
    if (edgeMidBrightRatio(mask, quad[i], quad[(i + 1) % 4]) >
        kMaxEdgeMidBright) {
      return false;
    }
  }

  std::vector<cv::Point2f> sorted;
  orderQuadClockwise(quad, sorted);

  out.detected_ = true;
  out.bounding_box_ = bounding_box;
  out.points_ = sorted;
  return true;
}

}  // namespace

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

// 预处理图像：灰度化、自适应二值化、形态学操作
// [H6 同类] 旧实现用固定的 inRange(200,255)。灯过曝到 255 时没问题，但环境一
// 变亮（白天、补光、自动曝光抬升）墙面/反光也会
// >200，整幅图变成一个或几个大亮块。 现在用 max(200,
// Otsu)：暗场景保持原来的行为，亮场景阈值自动抬高。
cv::Mat MarkerDetector::preprocess(const cv::Mat& frame) {
  if (frame.empty()) {
    return cv::Mat();
  }

  cv::Mat gray;
  if (frame.channels() == 3) {
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = frame;
  }

  cv::Mat otsu_dst;
  double otsu = cv::threshold(gray, otsu_dst, 0, 255,
                              cv::THRESH_BINARY | cv::THRESH_OTSU);
  int thresh = static_cast<int>(std::lround(otsu));
  if (thresh < kBrightFloor) thresh = kBrightFloor;

  cv::Mat binary;
  cv::threshold(gray, binary, thresh, 255, cv::THRESH_BINARY);

  cv::Mat morphed;
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::morphologyEx(binary, morphed, cv::MORPH_OPEN, kernel);

  return morphed;
}

std::vector<MarkerResult> MarkerDetector::detectMultiple(const cv::Mat& frame) {
  std::vector<MarkerResult> results;
  if (frame.empty()) {
    return results;
  }

  cv::Mat binary = preprocess(frame);
  if (binary.empty()) {
    return results;
  }

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(preprocess(frame), contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  // [H1] 面积门槛改成"随画面大小浮动"的同时保留一个很低的绝对地板：
  // 这里只用来滤掉单像素噪点，真正的判据交给后面的聚簇 + 象限 + 中心暗。
  const double min_area =
      std::max(8.0, 0.0002 * static_cast<double>(frame.cols) * frame.rows);

  std::vector<Blob> blobs;

  for (const auto& contour : contours) {
    if (cv::contourArea(contour) < min_area) continue;

    Blob b;

    b.contour = contour;
    b.center = getContourCenter(contour);

    cv::Rect r = cv::boundingRect(contour);

    b.size = static_cast<float>(std::max(r.width, r.height));

    if (b.size < 2.0f) continue;

    blobs.push_back(std::move(b));
  }

  const int n = static_cast<int>(blobs.size());
  if (n < 4) return results;  // 一个灯最少 4 个 L（外加小方块）

  // [H2] 按尺度聚簇：并查集 + "中心距 < kClusterRatio * max(尺寸)"。
  //      不用固定像素半径，所以距离远近都不变味。
  std::vector<int> parent(n);
  std::iota(parent.begin(), parent.end(), 0);
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      float merge_dist = kClusterRatio * std::max(blobs[i].size, blobs[j].size);
      if (cv::norm(blobs[i].center - blobs[j].center) < merge_dist) {
        unionSet(parent, i, j);
      }
    }
  }

  std::vector<std::vector<int>> groups;
  std::vector<int> root_to_group(n, -1);

  for (int i = 0; i < n; ++i) {
    int r = findRoot(parent, i);
    if (root_to_group[r] < 0) {
      root_to_group[r] = static_cast<int>(groups.size());
      groups.push_back({});
    }
    groups[root_to_group[r]].push_back(i);
  }

  for (const auto& group : groups) {
    if (static_cast<int>(group.size()) < 4) continue;

    // [H2] 干扰亮斑（矿石、反光、别的灯）可能因为"自身尺寸更大"把合簇半径撑大，
    //      从而被并进来。所以这里先用宽松的合簇结果去拟合方框，用 MARK 灯自身的
    //      结构（4 象限 + 中心暗 + 4 条边中点暗）判决；不通过就把离簇中心最远的
    //      那一块亮斑剔掉重新拟合，最多剔 2 次。这样"多一颗矿石"能被救回来，
    //      而"一整块亮矩形"永远救不回来（结构不对）。

    std::vector<int> members = group;

    for (int attempt = 0; attempt < 3; ++attempt) {
      MarkerResult candidate;

      if (fitMarker(blobs, members, binary, candidate)) {
        results.push_back(candidate);
        break;
      }
      if (static_cast<int>(members.size()) <= 4) break;
      // 去掉离簇中心最远的那块亮斑

      cv::Point2f mean_center(0, 0);
      for (int idx : members) mean_center += blobs[idx].center;
      mean_center *= 1.0f / static_cast<float>(members.size());

      size_t far_idx = 0;
      float far_dist = -1.0f;
      for (size_t k = 0; k < members.size(); ++k) {
        float d = static_cast<float>(
            cv::norm(blobs[members[k]].center - mean_center));
        if (d > far_dist) {
          far_dist = d;
          far_idx = k;
        }
      }
      members.erase(members.begin() + static_cast<long>(far_idx));
    }
  }
  return results;
}

// 将四个点按顺时针排序，左上、右上、右下、左下
std::vector<cv::Point2f> MarkerDetector::sortPoints(
    const std::vector<cv::Point2f>& pts) {
  std::vector<cv::Point2f> sorted;
  orderQuadClockwise(pts, sorted);
  return sorted;
}
