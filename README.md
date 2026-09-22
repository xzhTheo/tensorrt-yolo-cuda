# tensorrt-yolo-cuda

YOLO11 Detect 的两套工程，职责分开：

| 目录 | 角色 |
|------|------|
| **videoai** | Windows 界面：摄像头 / 上传视频 / 选模型 / 预览画框 |
| **cuda-tensorrt-detect** | CUDA 预处理 + TensorRT 推理引擎（本仓库重点） |

```
tensorrt-yolo-cuda/
  videoai/                  # UI 应用
  cuda-tensorrt-detect/     # 引擎库 + detect.exe
```

界面吃 TensorRT-YOLO 的编译产物（`trtyolo.dll` + `.engine`）。  
引擎工程自己用 CUDA + TensorRT 把「读图 → LetterBox → enqueue → 框映射回原图」走通。

---

## 1. 两套架构怎么分

```
                    ┌─────────────────────────┐
  摄像头 / 视频文件  │  videoai                 │
  官网 ONNX / engine │  读帧 → Detector → 画框  │
                    │  HighGUI 预览 / 导出 mp4 │
                    └────────────┬────────────┘
                                 │ TensorRT 路径调用
                                 ▼
                    ┌─────────────────────────┐
                    │  cuda-tensorrt-detect   │
                    │  CUDA LetterBox         │
                    │  TensorRT + EfficientNMS│
                    │  坐标反变换             │
                    └─────────────────────────┘
```

- **videoai**：应用层。OpenCV 读视频/摄像头，可选 OpenCV DNN 跑官网 `yolo11n.onnx`，或通过 `trtyolo::DetectModel` 跑 `.engine`。带启动窗、置信度滑条、HUD FPS。
- **cuda-tensorrt-detect**：推理层。从 TensorRT-YOLO 抽出的 Detect-only 实现：没有 classify / pose / segment / obb，没有自定义 plugin，NMS 用 TensorRT 自带 `EfficientNMS_TRT`。

---

## 2. cuda-tensorrt-detect（重点）

源码按调用顺序：

```
detect.cpp            入口：读图 → DetectModel::predict → 画框落盘
infer/trtyolo.hpp     公开 API：Image / Box / DetectRes / InferOption / DetectModel
infer/trtyolo.cpp     postProcessDetect：NMS 输出框从 letterbox 坐标映回原图
infer/backend.cpp     H2D → CUDA LetterBox → TensorRT enqueue → D2H
infer/letterbox.cu    等比缩放、pad 114、/255、可选 BGR→RGB（GPU 上做）
core/core.cpp         加载 .engine、注册官方 EfficientNMS、CUDA Graph
core/buffer.cpp       显存分配
utils/common.cpp      读 engine 文件、计时
labels.txt            COCO 80 类
models/yolo11n.engine FP16 引擎（与 GPU / TensorRT 版本绑定）
```

### 推理链路

```
BGR 图 (OpenCV)
  → InferOption.enableSwapRB()
  → CUDA LetterBox（保比例，灰边 114，归一化到 [0,1]）
  → TensorRT enqueueV3（FP16 + EfficientNMS）
  → 框从 640 画布坐标按 scale/pad 映回原图
  → 画框保存
```

### Engine I/O

| index | 角色 | shape |
|-------|------|--------|
| 0 | 输入图 | `[N, 3, H, W]` |
| 1 | num_detections | `[N, 1]` |
| 2 | boxes xyxy（letterbox 坐标） | `[N, max_det, 4]` |
| 3 | scores | `[N, max_det]` |
| 4 | classes | `[N, max_det]` |

预处理在 GPU（`letterbox.cu`），不是 CPU 上 `cv::resize`。  
NMS 在 TensorRT 图里，结果回传很小（框/分/类），所以 D2H 开销低。

### 编译与运行

本机依赖：MSVC（VS 2022 Build Tools）、CUDA 12.6、TensorRT 10.13.3.9、OpenCV 4.10、CMake + Ninja。  
默认架构 `sm_89`（RTX 4060 Laptop），换卡改 `CMakeLists.txt` 里 `CMAKE_CUDA_ARCHITECTURES`。

```bat
cd cuda-tensorrt-detect
build.bat
scripts\run.bat
```

或指定路径：

```bat
scripts\run.bat -e models\yolo11n.engine -i assets\bus.jpg -o output -l labels.txt
```

对目录推理会打印吞吐 / GPU 延迟 / CPU 延迟。

默认工具路径（`build.bat` / `CMakeLists.txt`）：

- TensorRT：`D:/tendor/tools/TensorRT-10.13.3.9`
- OpenCV：`D:/tendor/tools/opencv/build/x64/vc16/lib`
- CUDA：`C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6`

### 自己编引擎

仓库里的 `yolo11n.engine` 按 TensorRT 10.13 + FP16 编过。换 GPU 或 TensorRT 版本请重编：

```bat
yolo export model=yolo11n.pt format=onnx batch=1
trtyolo-export -i yolo11n.onnx -o yolo11n-trtyolo.onnx -s
trtexec --onnx=yolo11n-trtyolo.onnx --saveEngine=models\yolo11n.engine --fp16
```

---

## 3. videoai（界面）

```bat
cd videoai
scripts\build.bat
scripts\run.bat
```

不带参数弹出启动窗：打开摄像头或上传视频 → 选官网 ONNX / TensorRT 引擎 → 开始检测。  
预览热键：空格暂停、`s` 截图、`+/-` 调阈值、`q`/`ESC` 退出。

依赖额外包括 `D:\tendor\tools\trtyolo-install`（`trtyolo.dll`）和可选 FFmpeg。

---

## 4. 性能（RTX 4060 Laptop，同一段 1280×720 / 125 帧）

在 **GPU 空闲** 下测得。旁边有其它占 GPU 的程序时，数字会掉一截，对比如下。

| | GPU 有其它占用 | GPU 空闲 |
|--|--|--|
| 官网 ONNX（OpenCV DNN / CPU） | 8 FPS / 18 s | **11 FPS / 12.8 s** |
| TensorRT videoai 端到端 | 245 FPS / 3.1 s | **350～455 FPS / 1.9 s** |
| trtexec 纯引擎 | 271 qps，GPU ≈ 3.0 ms | **424 qps，GPU ≈ 1.7～1.8 ms** |
| trtexec 时延抖动 | 方差 64% | **方差 27%** |

怎么读：

- 官网 ONNX 走 CPU DNN，FPS 低，GPU 利用率接近空闲。
- TensorRT 引擎把预处理、推理、NMS 放 GPU，端到端大约 **30～40 倍**。
- `trtexec` 只测模型，不含读视频/画框/编码，所以 qps 高于 videoai HUD。
- 空闲时吞吐从 271 → 424 qps，说明引擎对 GPU 争用很敏感，测速时尽量关掉其它占卡程序。

复测：

```bat
:: 盯占用
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv -l 1

:: 纯引擎
trtexec --loadEngine=cuda-tensorrt-detect\models\yolo11n.engine --fp16 --warmUp=200 --duration=10

:: 端到端（在 videoai 目录）
scripts\run.bat -e models\yolo11n.onnx    -i assets\test.mp4 -o output\onnx.mp4 --no-show
scripts\run.bat -e models\yolo11n.engine  -i assets\test.mp4 -o output\trt.mp4  --no-show
```

---

## 5. 和完整 TensorRT-YOLO 的关系

引擎代码摘自 [laugh12321/TensorRT-YOLO](https://github.com/laugh12321/TensorRT-YOLO) 的 Detect 路径，版权仍归原作者。本仓库只保留学习/部署 Detect 所需部分。