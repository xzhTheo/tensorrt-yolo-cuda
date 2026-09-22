#include "video_processor.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>

#include <opencv2/opencv.hpp>

#include "detector.hpp"
#include "ffmpeg_io.hpp"
#include "visualizer.hpp"

namespace fs = std::filesystem;

namespace videoai {

namespace {
const char* kWindowName = "videoai - YOLO Video Detection";
struct UiState {
    int confSlider = 25;
};
}  // namespace

bool isVideoFile(const std::string& path) {
    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    static const std::vector<std::string> exts = {
        ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v", ".mpeg", ".mpg", ".webm"};
    return std::find(exts.begin(), exts.end(), ext) != exts.end();
}

bool isCameraIndex(const std::string& s) {
    return !s.empty() && std::all_of(s.begin(), s.end(),
                                     [](unsigned char c) { return std::isdigit(c); });
}

bool isStreamUrl(const std::string& s) {
    static const std::vector<std::string> p = {
        "rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://"};
    for (const auto& x : p) {
        if (s.size() >= x.size() && s.compare(0, x.size(), x) == 0) return true;
    }
    return false;
}

int runVideoPipeline(const PipelineConfig& config) {
    int    width = 0, height = 0, total = 0;
    double srcFps = 30.0;
    const bool useFfmpegInput = (config.backend == Backend::Ffmpeg) && !config.isCamera;

    cv::VideoCapture              cap;
    std::unique_ptr<FfmpegReader> ffReader;
    if (useFfmpegInput) {
        MediaInfo info;
        if (!ffprobeMedia(config.inputPath, info)) {
            throw std::runtime_error("ffprobe failed: " + config.inputPath);
        }
        width = info.width;
        height = info.height;
        srcFps = info.fps;
        total = (int)info.frames;
        ffReader = std::make_unique<FfmpegReader>(config.inputPath, width, height,
                                                  config.rtspTcp && config.isStream, config.hwaccel);
        std::cout << "[videoai] backend: FFmpeg  " << width << "x" << height << std::endl;
    } else {
        if (config.isCamera) cap.open(std::stoi(config.inputPath), cv::CAP_ANY);
        else cap.open(config.inputPath);
        if (!cap.isOpened()) throw std::runtime_error("cannot open: " + config.inputPath);
        srcFps = cap.get(cv::CAP_PROP_FPS);
        if (srcFps <= 1.0 || srcFps > 240.0) srcFps = 30.0;
        width  = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        total  = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
        std::cout << "[videoai] backend: OpenCV  " << width << "x" << height
                  << " @" << srcFps << "fps";
        if (total > 0) std::cout << "  frames=" << total;
        std::cout << std::endl;
    }

    auto readFrame = [&](cv::Mat& f) -> bool {
        if (ffReader) return ffReader->read(f);
        return cap.read(f) && !f.empty();
    };

    std::vector<std::string> labels = loadLabels(config.labelPath);
    Detector detector(config.enginePath, config.modelKind, config.deviceId);
    std::cout << "[videoai] model: " << detector.backendName() << "  "
              << config.enginePath << std::endl;

    cv::VideoWriter               writer;
    std::unique_ptr<FfmpegWriter> ffWriter;
    const bool useFfmpegOutput =
        (config.backend == Backend::Ffmpeg) && !config.outputPath.empty();
    if (!config.outputPath.empty()) {
        fs::path outDir = fs::path(config.outputPath).parent_path();
        if (!outDir.empty() && !fs::exists(outDir)) fs::create_directories(outDir);
        if (useFfmpegOutput) {
            std::string audioFrom;
            if (config.audioPassthrough && !config.isCamera && !config.isStream) {
                audioFrom = config.inputPath;
            }
            ffWriter = std::make_unique<FfmpegWriter>(config.outputPath, width, height, srcFps,
                                                      config.videoCodec, audioFrom);
        } else {
            writer.open(config.outputPath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                        srcFps, cv::Size(width, height));
            if (!writer.isOpened()) throw std::runtime_error("cannot write: " + config.outputPath);
        }
        std::cout << "[videoai] output: " << config.outputPath << std::endl;
    }

    auto hasWriter  = [&]() { return ffWriter || writer.isOpened(); };
    auto writeFrame = [&](const cv::Mat& f) {
        if (ffWriter) ffWriter->write(f);
        else if (writer.isOpened()) writer.write(f);
    };

    UiState ui;
    ui.confSlider = (int)(config.confThreshold * 100);
    if (config.showWindow) {
        cv::namedWindow(kWindowName, cv::WINDOW_NORMAL);
        cv::resizeWindow(kWindowName, std::min(width, 1280), std::min(height, 720));
        cv::createTrackbar("Conf %", kWindowName, &ui.confSlider, 100);
        std::cout << "[videoai] keys: space pause  s snapshot  +/- conf  q/ESC quit" << std::endl;
    }

    cv::Mat frame;
    int     frameIdx = 0;
    bool    paused   = false;
    double  emaFps   = 0.0;
    auto    lastReport = std::chrono::steady_clock::now();

    while (true) {
        cv::Mat display;
        if (!paused) {
            if (!readFrame(frame)) break;
            ++frameIdx;
            auto t0 = std::chrono::steady_clock::now();
            if (!frame.isContinuous()) frame = frame.clone();
            auto dets = detector.detect(frame.data, frame.cols, frame.rows);
            float confThreshold = ui.confSlider / 100.0f;
            std::map<std::string, int> counts;
            for (const auto& d : dets) {
                if (d.score < confThreshold) continue;
                std::string name = (d.classId >= 0 && d.classId < (int)labels.size())
                                       ? labels[d.classId]
                                       : ("id_" + std::to_string(d.classId));
                counts[name]++;
            }
            drawDetections(frame, dets, labels, confThreshold);
            auto   t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double inst = ms > 0 ? 1000.0 / ms : 0.0;
            emaFps = (emaFps == 0.0) ? inst : (emaFps * 0.9 + inst * 0.1);
            std::string info = (total > 0)
                                   ? (std::to_string(frameIdx) + " / " + std::to_string(total))
                                   : std::to_string(frameIdx);
            drawHud(frame, emaFps, info, counts, confThreshold, paused, detector.backendName());
            if (hasWriter()) writeFrame(frame);
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastReport).count() > 500) {
                std::cout << "\r[videoai] " << frameIdx;
                if (total > 0) std::cout << "/" << total;
                std::cout << "  " << cv::format("%.1f", emaFps) << " FPS   " << std::flush;
                lastReport = now;
            }
            display = frame;
        } else {
            display = frame.clone();
            drawHud(display, emaFps, std::to_string(frameIdx), {}, ui.confSlider / 100.0f, true,
                    detector.backendName());
        }
        if (config.showWindow) {
            cv::imshow(kWindowName, display);
            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) break;
            else if (key == ' ') paused = !paused;
            else if (key == 's') {
                std::string snap = "snapshot_" + std::to_string(frameIdx) + ".jpg";
                cv::imwrite(snap, display);
            } else if (key == '+' || key == '=') {
                ui.confSlider = std::min(100, ui.confSlider + 5);
                cv::setTrackbarPos("Conf %", kWindowName, ui.confSlider);
            } else if (key == '-' || key == '_') {
                ui.confSlider = std::max(0, ui.confSlider - 5);
                cv::setTrackbarPos("Conf %", kWindowName, ui.confSlider);
            }
            if (cv::getWindowProperty(kWindowName, cv::WND_PROP_VISIBLE) < 1) break;
        }
    }
    std::cout << std::endl;
    if (ffWriter) ffWriter->close();
    if (writer.isOpened()) writer.release();
    if (config.showWindow) cv::destroyAllWindows();
    std::cout << "[videoai] done, " << frameIdx << " frames." << std::endl;
    return frameIdx;
}

}  // namespace videoai
