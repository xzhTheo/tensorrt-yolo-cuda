#pragma once

#include <memory>
#include <string>
#include <vector>

namespace videoai {

enum class ModelKind {
    TensorRT,
    OfficialOnnx,
};

struct Detection {
    int   classId = 0;
    float score   = 0.f;
    int   left    = 0;
    int   top     = 0;
    int   right   = 0;
    int   bottom  = 0;
};

class Detector {
public:
    Detector(const std::string& modelPath, ModelKind kind, int deviceId = 0);
    ~Detector();

    Detector(const Detector&)            = delete;
    Detector& operator=(const Detector&) = delete;

    std::vector<Detection> detect(void* bgrData, int width, int height);
    const std::string&     backendName() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

ModelKind   modelKindFromPath(const std::string& path);
const char* modelKindLabel(ModelKind kind);

}  // namespace videoai
