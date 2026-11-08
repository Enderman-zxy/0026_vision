#include "VideoProcessor.h"

#include <iostream>

#include "Visualizer.h"

VideoProcessor::VideoProcessor() {}

// 处理视频流，检测标记，并根据检测结果更新状态
MarkerResult VideoProcessor::processFrame(cv::Mat& frame) {
  if (frame.empty()) return MarkerResult();

  std::vector<MarkerResult> results = detector_.detectMultiple(frame);
  MarkerResult bestResult;
  float max_area = 0.0f;
  
  for (const auto& res : results) {
    if (res.detected_) {
      float area = res.bounding_box_.width * res.bounding_box_.height;
      if (area > max_area) {
        max_area = area;
        bestResult = res;
      }
    }
  }

  // 2. 更新状态机
  updateState(bestResult);

  // 3. 可视化（直接在原图上画）
  Visualizer visualizer;
  if (detectedFrames_ >= MIN_DETECTED_FRAMES && lostFrames_ == 0) {
    visualizer.drawMarker(frame, lastResult_);
  } else {
    MarkerResult emptyResult;
    emptyResult.detected_ = false;
    visualizer.drawMarker(frame, emptyResult);
  }
  
  return lastResult_;
}

void VideoProcessor::updateState(const MarkerResult& currentResult) {
  if (currentResult.detected_) {
    detectedFrames_++;
    lostFrames_ = 0;
    if (detectedFrames_ >= MIN_DETECTED_FRAMES) {
      lastResult_ = currentResult;
    }
  } else {
    lostFrames_++;
    detectedFrames_ = 0;
    if (lostFrames_ > MAX_LOST_FRAMES) {
      lastResult_.detected_ = false;  
    }
  }
}