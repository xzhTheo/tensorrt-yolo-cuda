#pragma once

#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace videoai {

struct MediaInfo {
    int       width    = 0;
    int       height   = 0;
    double    fps      = 30.0;
    long long frames   = 0;
    bool      hasAudio = false;
};

bool ffprobeMedia(const std::string& input, MediaInfo& info);

class FfmpegReader {
public:
    FfmpegReader(const std::string& input, int width, int height, bool rtspTcp, bool hwaccel);
    ~FfmpegReader();
    FfmpegReader(const FfmpegReader&)            = delete;
    FfmpegReader& operator=(const FfmpegReader&) = delete;
    bool read(cv::Mat& frame);
    bool isOpen() const { return pipe_ != nullptr; }

private:
    FILE*              pipe_ = nullptr;
    int                w_;
    int                h_;
    size_t             frameBytes_;
    std::vector<uchar> buffer_;
};

class FfmpegWriter {
public:
    FfmpegWriter(const std::string& output, int width, int height, double fps,
                 const std::string& codec, const std::string& audioFrom);
    ~FfmpegWriter();
    FfmpegWriter(const FfmpegWriter&)            = delete;
    FfmpegWriter& operator=(const FfmpegWriter&) = delete;
    bool write(const cv::Mat& frame);
    bool isOpen() const { return pipe_ != nullptr; }
    void close();

private:
    FILE* pipe_ = nullptr;
    int   w_;
    int   h_;
};

}  // namespace videoai
