#include "detector.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include "trtyolo.hpp"

namespace videoai {
namespace {

constexpr int   kOnnxSize   = 640;
constexpr float kNmsIou     = 0.45f;
constexpr float kNmsConf    = 0.25f;
constexpr int   kNumClasses = 80;

cv::Mat letterbox(const cv::Mat& src, float& scale, int& padX, int& padY) {
    scale = std::min(kOnnxSize / static_cast<float>(src.cols),
                     kOnnxSize / static_cast<float>(src.rows));
    const int nw = static_cast<int>(std::round(src.cols * scale));
    const int nh = static_cast<int>(std::round(src.rows * scale));
    padX         = (kOnnxSize - nw) / 2;
    padY         = (kOnnxSize - nh) / 2;
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(nw, nh));
    cv::Mat out(kOnnxSize, kOnnxSize, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(padX, padY, nw, nh)));
    return out;
}

std::vector<Detection> parseOfficialOnnx(const cv::Mat& pred84x8400, float scale,
                                         int padX, int padY, int imgW, int imgH) {
    std::vector<cv::Rect> boxes;
    std::vector<float>    scores;
    std::vector<int>      classes;

    for (int i = 0; i < pred84x8400.cols; ++i) {
        float best = 0.f;
        int   cls  = 0;
        for (int c = 0; c < kNumClasses; ++c) {
            const float s = pred84x8400.at<float>(4 + c, i);
            if (s > best) {
                best = s;
                cls  = c;
            }
        }
        if (best < kNmsConf) continue;

        const float cx = pred84x8400.at<float>(0, i);
        const float cy = pred84x8400.at<float>(1, i);
        const float bw = pred84x8400.at<float>(2, i);
        const float bh = pred84x8400.at<float>(3, i);
        float x1 = (cx - bw * 0.5f - static_cast<float>(padX)) / scale;
        float y1 = (cy - bh * 0.5f - static_cast<float>(padY)) / scale;
        float x2 = (cx + bw * 0.5f - static_cast<float>(padX)) / scale;
        float y2 = (cy + bh * 0.5f - static_cast<float>(padY)) / scale;
        x1 = std::clamp(x1, 0.f, static_cast<float>(imgW - 1));
        y1 = std::clamp(y1, 0.f, static_cast<float>(imgH - 1));
        x2 = std::clamp(x2, 0.f, static_cast<float>(imgW - 1));
        y2 = std::clamp(y2, 0.f, static_cast<float>(imgH - 1));
        if (x2 <= x1 || y2 <= y1) continue;
        boxes.emplace_back(cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                           cv::Point(static_cast<int>(x2), static_cast<int>(y2)));
        scores.push_back(best);
        classes.push_back(cls);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, kNmsConf, kNmsIou, keep);
    std::vector<Detection> out;
    out.reserve(keep.size());
    for (int idx : keep) {
        const auto& b = boxes[idx];
        out.push_back(Detection{classes[idx], scores[idx], b.x, b.y,
                                b.x + b.width, b.y + b.height});
    }
    return out;
}

}  // namespace

struct Detector::Impl {
    ModelKind                             kind = ModelKind::TensorRT;
    std::string                           name;
    std::unique_ptr<trtyolo::DetectModel> trt;
    cv::dnn::Net                          onnx;
};

Detector::Detector(const std::string& modelPath, ModelKind kind, int deviceId)
    : impl_(std::make_unique<Impl>()) {
    impl_->kind = kind;
    if (kind == ModelKind::OfficialOnnx) {
        impl_->name = "Official ONNX (OpenCV DNN)";
        impl_->onnx = cv::dnn::readNetFromONNX(modelPath);
        if (impl_->onnx.empty()) {
            throw std::runtime_error("failed to load official ONNX: " + modelPath);
        }
        impl_->onnx.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        impl_->onnx.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    } else {
        impl_->name = "TensorRT + CUDA (optimized)";
        trtyolo::InferOption option;
        option.setDeviceId(deviceId);
        option.enableSwapRB();
        impl_->trt = std::make_unique<trtyolo::DetectModel>(modelPath, option);
    }
}

Detector::~Detector() = default;

const std::string& Detector::backendName() const {
    return impl_->name;
}

std::vector<Detection> Detector::detect(void* bgrData, int width, int height) {
    if (impl_->kind == ModelKind::OfficialOnnx) {
        cv::Mat frame(height, width, CV_8UC3, bgrData);
        float   scale = 1.f;
        int     padX = 0, padY = 0;
        cv::Mat boxed = letterbox(frame, scale, padX, padY);
        cv::Mat blob  = cv::dnn::blobFromImage(boxed, 1.0 / 255.0, cv::Size(),
                                               cv::Scalar(), true, false);
        impl_->onnx.setInput(blob);
        cv::Mat out  = impl_->onnx.forward();
        cv::Mat pred = out.reshape(1, out.size[1]);
        return parseOfficialOnnx(pred, scale, padX, padY, width, height);
    }

    trtyolo::Image     img(bgrData, width, height);
    trtyolo::DetectRes res = impl_->trt->predict(img);
    std::vector<Detection> detections;
    detections.reserve(res.num);
    for (int i = 0; i < res.num; ++i) {
        const auto& box = res.boxes[i];
        detections.push_back(Detection{
            res.classes[i], res.scores[i],
            static_cast<int>(box.left), static_cast<int>(box.top),
            static_cast<int>(box.right), static_cast<int>(box.bottom),
        });
    }
    return detections;
}

ModelKind modelKindFromPath(const std::string& path) {
    auto dot = path.find_last_of('.');
    if (dot == std::string::npos) return ModelKind::TensorRT;
    std::string ext = path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return (ext == ".onnx") ? ModelKind::OfficialOnnx : ModelKind::TensorRT;
}

const char* modelKindLabel(ModelKind kind) {
    return kind == ModelKind::OfficialOnnx ? "Official ONNX" : "TensorRT optimized";
}

}  // namespace videoai
