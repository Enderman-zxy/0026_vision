# PnX 视觉系统 (0026_vision)

## 项目简介
基于 MaixCAM Pro 的视觉识别系统。包含 L 型标靶检测与 PnP 位姿解算，以及退出标志 (AprilTag) 的 2D 偏差识别。底层使用 PoseLib 替代 OpenCV calib3d 模块。

## 编译指南
确保已安装并配置好 MaixCDK。
1. `maixcdk build -p maixcam`
2. `scp -f build/opencv_demo root@<板子IP>:/root/`

## 核心架构
- `MarkerDetector`: 检测 L 型亮斑
- `PoseEstimator`: 基于 PoseLib 的 P3P 解算
- `ExitTagRecognizer`: 纯 OpenCV 四边形识别（替代底层报错的 find_apriltags）
