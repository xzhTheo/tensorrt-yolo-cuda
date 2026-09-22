#pragma once

#include <map>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "detector.hpp"

namespace videoai {

std::vector<std::string> loadLabels(const std::string& labelFile);

int drawDetections(cv::Mat& image, const std::vector<Detection>& detections,
                   const std::vector<std::string>& labels, float confThreshold);

void drawHud(cv::Mat& image, double fps, const std::string& frameInfo,
             const std::map<std::string, int>& counts, float confThreshold, bool paused,
             const std::string& modelName = "");

cv::Scalar colorForClass(int classId);

}  // namespace videoai
