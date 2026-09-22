# tensorrt-yolo-cuda

Two top-level projects:

| folder | what |
|--------|------|
| **videoai** | Windows UI: camera / upload video / official ONNX vs TensorRT, preview window |
| **cuda-tensorrt-detect** | CUDA letterbox + TensorRT detect engine (`detect.exe`) |

## videoai (UI)

```bat
cd videoai
scripts\build.bat
scripts\run.bat
```

## cuda-tensorrt-detect (engine)

```bat
cd cuda-tensorrt-detect
build.bat
scripts\run.bat
```

See each folder README for dependencies and engine export.