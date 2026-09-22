#pragma once

#include <string>

#include "detector.hpp"

namespace videoai {

enum class LaunchAction {
    Quit,
    Camera,
    Video,
};

struct LaunchRequest {
    LaunchAction action    = LaunchAction::Quit;
    ModelKind    modelKind = ModelKind::TensorRT;
    std::string  videoPath;
};

LaunchRequest showLaunchWindow();
std::string   openVideoFileDialog();
void          showErrorDialog(const std::string& message);

}  // namespace videoai
