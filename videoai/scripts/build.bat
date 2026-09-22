@echo off
call "D:\VS\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
set "PATH=%CUDA_PATH%\bin;%PATH%"
set "CMAKE=C:\Program Files\CMake\bin\cmake.exe"
set "NINJA=D:\VS\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
set "PROJ=D:\tendor\videoai"
cd /d "%PROJ%"
"%CMAKE%" -S . -B build -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
  -DOpenCV_DIR=D:/tendor/tools/opencv/build/x64/vc16/lib ^
  -DCMAKE_PREFIX_PATH=D:/tendor/tools/trtyolo-install
if errorlevel 1 ( echo CONFIGURE_FAILED & exit /b 1 )
"%CMAKE%" --build build --config Release
if errorlevel 1 ( echo BUILD_FAILED & exit /b 2 )
echo BUILD_OK