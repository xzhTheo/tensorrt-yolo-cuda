/**
 * @file trtyolo.hpp
 * @brief Detect-only public API (classify / pose / segment / obb removed)
 *
 * Call chain: DetectModel::predict -> TrtBackend::infer (LetterBox + TensorRT) -> postProcessDetect
 * Engine I/O:
 *   [0] input image [N,3,H,W]
 *   [1] num_detections [N,1]
 *   [2] detection_boxes [N,max_det,4]  letterbox xyxy
 *   [3] detection_scores [N,max_det]
 *   [4] detection_classes [N,max_det]
 */
#pragma once

#ifdef _MSC_VER
#define TRTYOLOAPI __declspec(dllexport)
#else
#define TRTYOLOAPI __attribute__((visibility("default")))
#endif

#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace trtyolo {

class DetectModel;

struct TRTYOLOAPI Image {
    void*  ptr;
    int    width    = 0;
    int    height   = 0;
    int    channels = 0;
    size_t pitch    = 0;

    Image(void* data, int width, int height);
    Image(void* data, int width, int height, size_t pitch);
    Image(void* data, int width, int height, int channels, size_t pitch);

    friend std::ostream& operator<<(std::ostream& os, const Image& img) {
        os << "Image(width=" << img.width
           << ", height=" << img.height
           << ", channels=" << img.channels
           << ", pitch=" << img.pitch
           << ", ptr=" << img.ptr << ")";
        return os;
    }
};

struct TRTYOLOAPI Box {
    float left;
    float top;
    float right;
    float bottom;

    Box(float left, float top, float right, float bottom)
        : left(left), top(top), right(right), bottom(bottom) {}

    std::array<int, 4> xyxy() const;

    friend std::ostream& operator<<(std::ostream& os, const Box& box) {
        os << "Box(left=" << box.left << ", top=" << box.top
           << ", right=" << box.right << ", bottom=" << box.bottom << ")";
        return os;
    }
};

struct TRTYOLOAPI DetectRes {
    int                num = 0;
    std::vector<int>   classes;
    std::vector<float> scores;
    std::vector<Box>   boxes;

    DetectRes() = default;
    DetectRes(int num, const std::vector<int>& classes, const std::vector<float>& scores, const std::vector<Box>& boxes)
        : num(num), classes(classes), scores(scores), boxes(boxes) {}

    friend std::ostream& operator<<(std::ostream& os, const DetectRes& res) {
        os << "DetectRes(\n    num=" << res.num << ",\n    classes=[";
        for (const auto& c : res.classes) os << c << ", ";
        os << "],\n    scores=[";
        for (const auto& s : res.scores) os << s << ", ";
        os << "],\n    boxes=[\n";
        for (const auto& box : res.boxes) os << "        " << box << ",\n";
        os << "    ]\n)";
        return os;
    }
};

class TRTYOLOAPI InferOption {
public:
    InferOption();
    ~InferOption();

    void setDeviceId(int id);
    void enableCudaMem();
    void enableManagedMemory();
    void enablePerformanceReport();
    void enableSwapRB();
    void setBorderValue(float border_value);
    void setNormalizeParams(const std::vector<float>& mean, const std::vector<float>& std);
    void setInputDimensions(int width, int height);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    friend class DetectModel;
};

class TRTYOLOAPI DetectModel {
public:
    DetectModel();
    ~DetectModel();

    explicit DetectModel(const std::string& trt_engine_file, const InferOption& infer_option);

    std::unique_ptr<DetectModel> clone() const;

    int batch() const;

    std::tuple<std::string, std::string, std::string> performanceReport();

    DetectRes predict(const Image& image);
    std::vector<DetectRes> predict(const std::vector<Image>& images);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace trtyolo
