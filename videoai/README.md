# videoai

Windows UI for YOLO video / camera detect.

- Open camera or upload a video
- Choose official ONNX (`yolo11n.onnx`) or TensorRT engine (`yolo11n.engine`)
- Start detect: preview window + optional export to `output\`

This folder is the **UI app**. The CUDA / TensorRT engine library lives in sibling `cuda-tensorrt-detect`.

## Run

```bat
cd D:\tendor\videoai
scripts\build.bat
scripts\run.bat
```

Depends on: MSVC, CUDA 12.6, TensorRT 10.13, OpenCV 4.10, `D:\tendor\tools\trtyolo-install`.