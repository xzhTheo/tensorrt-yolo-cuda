#pragma once

#include <string>

#include "detector.hpp"

namespace videoai {

enum class Backend {
    OpenCV,
    Ffmpeg,
};

struct PipelineConfig {
    std::string enginePath;
    ModelKind   modelKind     = ModelKind::TensorRT;
    std::string inputPath;
    std::string outputPath;
    std::string labelPath;
    bool        showWindow    = false;
    float       confThreshold = 0.25f;
    int         deviceId      = 0;
    bool        isCamera      = false;
    Backend     backend          = Backend::OpenCV;
    std::string videoCodec       = "libx264";
    bool        hwaccel          = false;
    bool        rtspTcp          = true;
    bool        audioPassthrough = true;
    bool        isStream         = false;
};

int runVideoPipeline(const PipelineConfig& config);
bool isVideoFile(const std::string& path);
bool isCameraIndex(const std::string& s);
bool isStreamUrl(const std::string& s);

}  // namespace videoai
