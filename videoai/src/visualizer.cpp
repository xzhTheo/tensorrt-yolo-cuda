#include "visualizer.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace videoai {

std::vector<std::string> loadLabels(const std::string& labelFile) {
    std::ifstream file(labelFile);
    if (!file.is_open()) throw std::runtime_error("cannot open labels: " + labelFile);
    std::vector<std::string> labels;
    std::string              line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        labels.emplace_back(line);
    }
    return labels;
}

cv::Scalar colorForClass(int classId) {
    const double goldenAngle = 137.508;
    double       hue         = std::fmod(classId * goldenAngle, 180.0);
    cv::Mat      hsv(1, 1, CV_8UC3, cv::Scalar(static_cast<uchar>(hue), 200, 255));
    cv::Mat      bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    cv::Vec3b c = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(c[0], c[1], c[2]);
}

int drawDetections(cv::Mat& image, const std::vector<Detection>& detections,
                   const std::vector<std::string>& labels, float confThreshold) {
    int drawn = 0;
    for (const auto& det : detections) {
        if (det.score < confThreshold) continue;
        ++drawn;
        cv::Scalar  color = colorForClass(det.classId);
        std::string name  = (det.classId >= 0 && det.classId < (int)labels.size())
                                ? labels[det.classId]
                                : ("id_" + std::to_string(det.classId));
        std::string text  = name + " " + cv::format("%.2f", det.score);
        cv::rectangle(image, cv::Point(det.left, det.top), cv::Point(det.right, det.bottom),
                      color, 2, cv::LINE_AA);
        int      base = 0;
        cv::Size ts   = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
        int      top  = std::max(det.top, ts.height + 4);
        cv::rectangle(image, cv::Point(det.left, top - ts.height - 4),
                      cv::Point(det.left + ts.width + 2, top), color, cv::FILLED);
        cv::putText(image, text, cv::Point(det.left + 1, top - 3),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }
    return drawn;
}

void drawHud(cv::Mat& image, double fps, const std::string& frameInfo,
             const std::map<std::string, int>& counts, float confThreshold, bool paused,
             const std::string& modelName) {
    int total = 0;
    for (const auto& kv : counts) total += kv.second;
    std::vector<std::string> lines;
    if (!modelName.empty()) lines.push_back(modelName);
    lines.push_back(cv::format("FPS: %.1f   Objects: %d   Conf>=%.2f", fps, total, confThreshold));
    if (!frameInfo.empty()) {
        lines.push_back("Frame: " + frameInfo + (paused ? "   [PAUSED]" : ""));
    } else if (paused) {
        lines.push_back("[PAUSED]");
    }
    std::vector<std::pair<std::string, int>> sorted(counts.begin(), counts.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    std::string summary;
    for (size_t i = 0; i < sorted.size() && i < 4; ++i) {
        if (!summary.empty()) summary += "  ";
        summary += sorted[i].first + ":" + std::to_string(sorted[i].second);
    }
    if (!summary.empty()) lines.push_back(summary);

    int     pad = 8, lineH = 22, boxW = 420;
    int     boxH = pad * 2 + lineH * (int)lines.size();
    cv::Mat overlay = image.clone();
    cv::rectangle(overlay, cv::Point(0, 0), cv::Point(boxW, boxH), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::addWeighted(overlay, 0.45, image, 0.55, 0, image);
    int y = pad + 16;
    for (const auto& l : lines) {
        cv::putText(image, l, cv::Point(pad, y), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        y += lineH;
    }
}

}  // namespace videoai
