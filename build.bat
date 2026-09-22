@echo off
REM Build tensorrt-yolo-cuda (CUDA letterbox + TensorRT detect)
call "D:\VS\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
set "PATH=%CUDA_PATH%\bin;%PATH%"

set "PROJ=D:\tendor\cuda-tensorrt-detect"
set "CMAKE=C:\Program Files\CMake\bin\cmake.exe"
set "NINJA=D:\VS\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
if not exist "%NINJA%" set "NINJA=D:\tendor\.venv\Scripts\ninja.exe"

cd /d "%PROJ%"

echo ============ CMake configure ============
"%CMAKE%" -S . -B build -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
  -DTRT_PATH=D:/tendor/tools/TensorRT-10.13.3.9 ^
  -DOpenCV_DIR=D:/tendor/tools/opencv/build/x64/vc16/lib ^
  -DCMAKE_CUDA_COMPILER="%CUDA_PATH%/bin/nvcc.exe"
if errorlevel 1 ( echo CONFIGURE_FAILED & exit /b 1 )

echo ============ CMake build ============
"%CMAKE%" --build build --config Release
if errorlevel 1 ( echo BUILD_FAILED & exit /b 2 )

echo BUILD_OK
echo Next: scripts\run.bat