#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "app_ui.hpp"
#include "detector.hpp"
#include "video_processor.hpp"
#include "visualizer.hpp"

namespace fs = std::filesystem;

namespace {

void setupConsoleEncoding() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}

void printUsage(const char* prog) {
    std::cout <<
        "videoai\n\n"
        "  (no args)  open the UI\n"
        "  " << prog << " -e <engine> -i <input> [-o <output>] -l <labels> [opts]\n";
}

std::string defaultEngine() {
    return fs::exists("models/yolo11n.engine") ? "models/yolo11n.engine" : "";
}
std::string defaultOfficialOnnx() {
    if (fs::exists("models/yolo11n.onnx")) return "models/yolo11n.onnx";
    const char* alt = "D:/tendor/TensorRT-YOLO/examples/detect/models/yolo11n.onnx";
    return fs::exists(alt) ? alt : "";
}
std::string defaultLabels() {
    return fs::exists("labels/coco.txt") ? "labels/coco.txt" : "";
}
std::string modelPathFor(videoai::ModelKind kind) {
    return kind == videoai::ModelKind::OfficialOnnx ? defaultOfficialOnnx() : defaultEngine();
}

void runPipelineForSource(const std::string& enginePath, const std::string& labelPath,
                          const std::string& inputPath, const std::string& outputPath,
                          bool isCamera, bool isStream, bool showWindow,
                          float conf, int device,
                          videoai::Backend backend, const std::string& codec,
                          bool hwaccel, bool audioPassthrough,
                          videoai::ModelKind modelKind) {
    videoai::PipelineConfig cfg;
    cfg.enginePath = enginePath;
    cfg.modelKind = modelKind;
    cfg.inputPath = inputPath;
    cfg.outputPath = outputPath;
    cfg.labelPath = labelPath;
    cfg.showWindow = showWindow;
    cfg.confThreshold = conf;
    cfg.deviceId = device;
    cfg.isCamera = isCamera;
    cfg.isStream = isStream;
    cfg.backend = backend;
    cfg.videoCodec = codec;
    cfg.hwaccel = hwaccel;
    cfg.audioPassthrough = audioPassthrough;
    videoai::runVideoPipeline(cfg);
}

std::vector<std::string> listImages(const std::string& folder) {
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(folder)) {
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return (char)std::tolower(c); });
        if (fs::is_regular_file(entry) &&
            (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")) {
            files.push_back(entry.path().string());
        }
    }
    return files;
}

void processImage(const std::string& imagePath, const std::string& outPath,
                  videoai::Detector& detector,
                  const std::vector<std::string>& labels, float conf) {
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) throw std::runtime_error("cannot read image: " + imagePath);
    if (!image.isContinuous()) image = image.clone();
    auto dets = detector.detect(image.data, image.cols, image.rows);
    int n = videoai::drawDetections(image, dets, labels, conf);
    std::cout << "[videoai] " << fs::path(imagePath).filename().string()
              << " -> " << n << " objects" << std::endl;
    if (!outPath.empty()) cv::imwrite(outPath, image);
}

int runUiMode(const std::string& labelPath) {
    while (true) {
        videoai::LaunchRequest req = videoai::showLaunchWindow();
        if (req.action == videoai::LaunchAction::Quit) return 0;
        try {
            const std::string modelPath = modelPathFor(req.modelKind);
            if (modelPath.empty()) {
                throw std::runtime_error(req.modelKind == videoai::ModelKind::OfficialOnnx
                                             ? "missing models/yolo11n.onnx"
                                             : "missing models/yolo11n.engine");
            }
            if (req.action == videoai::LaunchAction::Camera) {
                runPipelineForSource(modelPath, labelPath, "0", "",
                                     true, false, true, 0.25f, 0,
                                     videoai::Backend::OpenCV, "libx264", false, true,
                                     req.modelKind);
            } else {
                fs::create_directories("output");
                const std::string out =
                    (fs::path("output") / (fs::path(req.videoPath).stem().string() + "_out.mp4"))
                        .string();
                runPipelineForSource(modelPath, labelPath, req.videoPath, out,
                                     false, false, true, 0.25f, 0,
                                     videoai::Backend::OpenCV, "libx264", false, true,
                                     req.modelKind);
            }
        } catch (const std::exception& e) {
            std::cerr << "error: " << e.what() << std::endl;
            videoai::showErrorDialog(e.what());
        }
    }
}

int runCli(int argc, char** argv) {
    std::string enginePath, inputPath, outputPath, labelPath;
    bool showWindow = false, noShow = false, hwaccel = false, audioOff = false;
    float conf = 0.25f;
    int device = 0;
    std::string backendStr = "opencv", codec = "libx264";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
            return argv[++i];
        };
        if (arg == "-e" || arg == "--engine") enginePath = need();
        else if (arg == "-i" || arg == "--input") inputPath = need();
        else if (arg == "-o" || arg == "--output") outputPath = need();
        else if (arg == "-l" || arg == "--labels") labelPath = need();
        else if (arg == "--show") showWindow = true;
        else if (arg == "--no-show") noShow = true;
        else if (arg == "--conf") conf = std::stof(need());
        else if (arg == "--device") device = std::stoi(need());
        else if (arg == "--backend") backendStr = need();
        else if (arg == "--codec") codec = need();
        else if (arg == "--hwaccel") hwaccel = true;
        else if (arg == "--no-audio") audioOff = true;
        else if (arg == "-h" || arg == "--help") { printUsage(argv[0]); return 0; }
        else if (!arg.empty() && arg[0] != '-' && inputPath.empty()) inputPath = arg;
        else {
            std::cerr << "unknown arg: " << arg << std::endl;
            printUsage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (enginePath.empty()) enginePath = defaultEngine();
    if (labelPath.empty()) labelPath = defaultLabels();
    if (enginePath.empty()) throw std::runtime_error("need -e engine file");
    if (!fs::exists(enginePath)) throw std::runtime_error("engine not found: " + enginePath);
    if (labelPath.empty()) throw std::runtime_error("need -l label file");
    if (inputPath.empty()) {
        inputPath = videoai::openVideoFileDialog();
        if (inputPath.empty()) { printUsage(argv[0]); return EXIT_FAILURE; }
    }

    videoai::Backend backend = (backendStr == "ffmpeg") ? videoai::Backend::Ffmpeg
                                                        : videoai::Backend::OpenCV;
    const bool isCamera = videoai::isCameraIndex(inputPath);
    const bool isStream = videoai::isStreamUrl(inputPath);
    const bool isVideo  = !isCamera && !isStream && fs::is_regular_file(inputPath) &&
                         videoai::isVideoFile(inputPath);

    if (isCamera || isVideo || isStream) {
        const bool autoShow  = isCamera || ((isVideo || isStream) && outputPath.empty());
        const bool finalShow = !noShow && (showWindow || autoShow);
        runPipelineForSource(enginePath, labelPath, inputPath, outputPath,
                             isCamera, isStream, finalShow, conf, device,
                             backend, codec, hwaccel, !audioOff,
                             videoai::modelKindFromPath(enginePath));
        return 0;
    }

    if (!fs::exists(inputPath)) throw std::runtime_error("input not found: " + inputPath);
    auto labels = videoai::loadLabels(labelPath);
    videoai::Detector detector(enginePath, videoai::modelKindFromPath(enginePath), device);
    if (fs::is_regular_file(inputPath)) {
        processImage(inputPath, outputPath, detector, labels, conf);
    } else if (fs::is_directory(inputPath)) {
        auto images = listImages(inputPath);
        if (images.empty()) throw std::runtime_error("no images in: " + inputPath);
        if (!outputPath.empty() && !fs::exists(outputPath)) fs::create_directories(outputPath);
        for (const auto& img : images) {
            std::string out = outputPath.empty()
                                  ? ""
                                  : (fs::path(outputPath) / fs::path(img).filename()).string();
            processImage(img, out, detector, labels, conf);
        }
    } else {
        throw std::runtime_error("unsupported input: " + inputPath);
    }
    std::cout << "[videoai] done." << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    setupConsoleEncoding();
    try {
        if (argc <= 1) {
            const std::string labels = defaultLabels();
            if (labels.empty()) throw std::runtime_error("missing labels/coco.txt");
            return runUiMode(labels);
        }
        return runCli(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        videoai::showErrorDialog(e.what());
        return EXIT_FAILURE;
    }
}
