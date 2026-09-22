@echo off
chcp 65001 >nul
setlocal
set "PROJ=D:\tendor\videoai"
set "PATH=D:\tendor\tools\TensorRT-10.13.3.9\lib;D:\tendor\tools\opencv\build\x64\vc16\bin;D:\tendor\tools\trtyolo-install\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;D:\tendor\tools\ffmpeg\bin;%PATH%"
cd /d "%PROJ%"
"%PROJ%\bin\videoai.exe" %*
endlocal