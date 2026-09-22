# tensorrt-yolo-cuda

YOLO **Detect** inference with **CUDA preprocess** and a **TensorRT** engine.

This repo is a focused extract of [laugh12321/TensorRT-YOLO](https://github.com/laugh12321/TensorRT-YOLO): detect only. No custom plugin, no classify / pose / segment / obb, no Python bindings. NMS uses TensorRT's built-in `EfficientNMS_TRT`.

Repo path: `cuda-tensorrt-detect/` (also usable as `D:\tendor\tensorrt-yolo-cuda\cuda-tensorrt-detect`).

## Pipeline

```
image (BGR)
  -> InferOption.enableSwapRB
  -> CUDA LetterBox (keep aspect, pad 114, /255, BGR->RGB)
  -> TensorRT enqueue (FP16 engine + EfficientNMS)
  -> map boxes back to original image
  -> draw and save
```

Engine I/O:

| index | role | shape |
|-------|------|--------|
| 0 | input | `[N,3,H,W]` |
| 1 | num_detections | `[N,1]` |
| 2 | boxes xyxy (letterbox) | `[N,max_det,4]` |
| 3 | scores | `[N,max_det]` |
| 4 | classes | `[N,max_det]` |

## Layout

```
detect.cpp            CLI: image or folder
infer/trtyolo.*       public API (Image / DetectRes / DetectModel)
infer/backend.*       H2D -> LetterBox -> enqueue -> D2H
infer/letterbox.cu    CUDA preprocess
core/core.*           load engine, EfficientNMS, CUDA Graph
core/buffer.*         device buffers
utils/common.*        read engine, timing
labels.txt            COCO 80 classes
models/yolo11n.engine TensorRT engine (FP16, yolo11n)
assets/bus.jpg        sample image
build.bat             configure + build
scripts/run.bat       set DLL PATH and run
```

## Dependencies (this machine)

- Visual Studio 2022 Build Tools (MSVC x64)
- CUDA Toolkit 12.6
- TensorRT 10.13.3.9
- OpenCV 4.10
- CMake + Ninja
- NVIDIA GPU, sm_89 (RTX 4060 Laptop). Change `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt` for other GPUs.

Default paths in `build.bat` / `CMakeLists.txt`:

- TensorRT: `D:/tendor/tools/TensorRT-10.13.3.9`
- OpenCV: `D:/tendor/tools/opencv/build/x64/vc16/lib`
- CUDA: `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6`

## Build

```bat
cd cuda-tensorrt-detect
build.bat
```

Produces `bin\detect.exe` and copies TensorRT runtime DLLs into `bin\`.

## Run

```bat
:: sample image
scripts\run.bat

:: custom
scripts\run.bat -e models\yolo11n.engine -i assets\bus.jpg -o output -l labels.txt

:: folder (prints throughput / latency)
scripts\run.bat -e models\yolo11n.engine -i assets -o output -l labels.txt
```

Output: `output\<name>.jpg` with boxes.

Need these on PATH (handled by `scripts\run.bat`):

- `TensorRT\lib`
- `opencv\build\x64\vc16\bin`
- `CUDA\v12.6\bin`

## Build your own engine

The checked-in `models/yolo11n.engine` was built with TensorRT 10.13 + FP16 for this GPU. Other GPUs / TensorRT versions should rebuild:

```bat
:: 1) official weights -> ONNX
yolo export model=yolo11n.pt format=onnx batch=1

:: 2) TensorRT-YOLO export (adds EfficientNMS)
trtyolo-export -i yolo11n.onnx -o yolo11n-trtyolo.onnx -s

:: 3) build engine
trtexec --onnx=yolo11n-trtyolo.onnx --saveEngine=models\yolo11n.engine --fp16
```

## Notes

- Preprocess is CUDA (`letterbox.cu`), not OpenCV resize on CPU.
- Detect NMS is the official TensorRT plugin, not a custom plugin in this repo.
- Original library copyright: laugh12321 / TensorRT-YOLO.