@echo off
chcp 65001 >nul
setlocal
set "PROJ=D:\tendor\cuda-tensorrt-detect"
set "PATH=D:\tendor\tools\TensorRT-10.13.3.9\lib;D:\tendor\tools\opencv\build\x64\vc16\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;%PROJ%\bin;%PATH%"
cd /d "%PROJ%"

if not exist "bin\detect.exe" (
  echo detect.exe not found. Run build.bat first.
  exit /b 1
)

if "%~1"=="" (
  bin\detect.exe -e models\yolo11n.engine -i assets\bus.jpg -o output -l labels.txt
) else (
  bin\detect.exe %*
)
endlocal