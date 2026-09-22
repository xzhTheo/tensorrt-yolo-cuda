#include "ffmpeg_io.hpp"

#include <cstdlib>
#include <sstream>
#include <stdexcept>

namespace videoai {
namespace {

std::string ffmpegExe() {
    if (const char* e = std::getenv("VIDEOAI_FFMPEG")) return e;
    return "ffmpeg";
}
std::string ffprobeExe() {
    if (const char* e = std::getenv("VIDEOAI_FFPROBE")) return e;
    return "ffprobe";
}
std::string q(const std::string& s) { return "\"" + s + "\""; }

}  // namespace

bool ffprobeMedia(const std::string& input, MediaInfo& info) {
    {
        std::ostringstream cmd;
        cmd << ffprobeExe()
            << " -v error -select_streams v:0"
            << " -show_entries stream=width,height,avg_frame_rate,nb_frames"
            << " -of default=noprint_wrappers=1:nokey=0 " << q(input);
        FILE* p = _popen(cmd.str().c_str(), "r");
        if (!p) return false;
        char line[256];
        while (std::fgets(line, sizeof(line), p)) {
            std::string s(line);
            auto        eq = s.find('=');
            if (eq == std::string::npos) continue;
            std::string key = s.substr(0, eq);
            std::string val = s.substr(eq + 1);
            while (!val.empty() && (val.back() == '\n' || val.back() == '\r')) val.pop_back();
            if (key == "width") info.width = std::atoi(val.c_str());
            else if (key == "height") info.height = std::atoi(val.c_str());
            else if (key == "avg_frame_rate") {
                auto slash = val.find('/');
                if (slash != std::string::npos) {
                    double num = std::atof(val.substr(0, slash).c_str());
                    double den = std::atof(val.substr(slash + 1).c_str());
                    if (den > 0) info.fps = num / den;
                } else {
                    double f = std::atof(val.c_str());
                    if (f > 0) info.fps = f;
                }
            } else if (key == "nb_frames" && val != "N/A") {
                info.frames = std::atoll(val.c_str());
            }
        }
        _pclose(p);
    }
    if (info.width <= 0 || info.height <= 0) return false;
    if (info.fps <= 0.0 || info.fps > 1000.0) info.fps = 30.0;
    {
        std::ostringstream cmd;
        cmd << ffprobeExe() << " -v error -select_streams a:0 -show_entries stream=index"
            << " -of default=noprint_wrappers=1:nokey=1 " << q(input);
        FILE* p = _popen(cmd.str().c_str(), "r");
        if (p) {
            char line[64];
            if (std::fgets(line, sizeof(line), p)) {
                info.hasAudio = (line[0] != '\0' && line[0] != '\n');
            }
            _pclose(p);
        }
    }
    return true;
}

FfmpegReader::FfmpegReader(const std::string& input, int width, int height, bool rtspTcp,
                           bool hwaccel)
    : w_(width), h_(height) {
    frameBytes_ = static_cast<size_t>(w_) * h_ * 3;
    buffer_.resize(frameBytes_);
    std::ostringstream cmd;
    cmd << ffmpegExe() << " -hide_banner -loglevel error";
    if (rtspTcp) cmd << " -rtsp_transport tcp";
    if (hwaccel) cmd << " -hwaccel cuda";
    cmd << " -i " << q(input) << " -an -f rawvideo -pix_fmt bgr24 pipe:1";
    pipe_ = _popen(cmd.str().c_str(), "rb");
    if (!pipe_) throw std::runtime_error("failed to start ffmpeg decoder");
}

FfmpegReader::~FfmpegReader() {
    if (pipe_) {
        _pclose(pipe_);
        pipe_ = nullptr;
    }
}

bool FfmpegReader::read(cv::Mat& frame) {
    if (!pipe_) return false;
    size_t got = 0;
    while (got < frameBytes_) {
        size_t n = std::fread(buffer_.data() + got, 1, frameBytes_ - got, pipe_);
        if (n == 0) break;
        got += n;
    }
    if (got < frameBytes_) return false;
    cv::Mat m(h_, w_, CV_8UC3, buffer_.data());
    m.copyTo(frame);
    return true;
}

FfmpegWriter::FfmpegWriter(const std::string& output, int width, int height, double fps,
                           const std::string& codec, const std::string& audioFrom)
    : w_(width), h_(height) {
    std::ostringstream cmd;
    cmd << ffmpegExe() << " -hide_banner -loglevel error -y"
        << " -f rawvideo -pix_fmt bgr24 -s " << w_ << "x" << h_
        << " -r " << fps << " -i pipe:0";
    if (!audioFrom.empty()) {
        cmd << " -i " << q(audioFrom) << " -map 0:v:0 -map 1:a:0? -c:a aac -shortest";
    }
    cmd << " -c:v " << codec << " -pix_fmt yuv420p -movflags +faststart " << q(output);
    pipe_ = _popen(cmd.str().c_str(), "wb");
    if (!pipe_) throw std::runtime_error("failed to start ffmpeg encoder");
}

FfmpegWriter::~FfmpegWriter() { close(); }

bool FfmpegWriter::write(const cv::Mat& frame) {
    if (!pipe_) return false;
    cv::Mat cont  = frame.isContinuous() ? frame : frame.clone();
    size_t  bytes = static_cast<size_t>(w_) * h_ * 3;
    return std::fwrite(cont.data, 1, bytes, pipe_) == bytes;
}

void FfmpegWriter::close() {
    if (pipe_) {
        _pclose(pipe_);
        pipe_ = nullptr;
    }
}

}  // namespace videoai
