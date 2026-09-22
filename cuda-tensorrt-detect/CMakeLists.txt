cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0091 NEW)

# Must be set before project() so nvcc does not default to sm_52
set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures")

# Detect-only: CUDA letterbox + TensorRT engine. No plugin / classify / pose / segment / obb.
project(tensorrt-yolo-cuda
    VERSION 1.0.0
    LANGUAGES CXX CUDA
    DESCRIPTION "YOLO Detect inference with CUDA preprocess and TensorRT"
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

if(NOT TRT_PATH)
    if(EXISTS "D:/tendor/tools/TensorRT-10.13.3.9")
        set(TRT_PATH "D:/tendor/tools/TensorRT-10.13.3.9")
    endif()
endif()
set(TRT_PATH "${TRT_PATH}" CACHE PATH "Path to TensorRT installation")
if(NOT TRT_PATH)
    message(FATAL_ERROR "Set TensorRT path: -DTRT_PATH=/path/to/tensorrt")
endif()

set(TRT_LIB_DIR "${TRT_PATH}/lib")
if(MSVC AND EXISTS "${TRT_LIB_DIR}/nvinfer_10.lib")
    set(TRT_LIBS nvinfer_10 nvinfer_plugin_10 nvonnxparser_10)
else()
    set(TRT_LIBS nvinfer nvinfer_plugin nvonnxparser)
endif()

function(set_target_compile_options target)
    if(MSVC)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/O2 /Oi /fp:fast>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:/Od /Ob0 /Zi /RTC1>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/utf-8,/O2 -O3>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/utf-8,/Od -g -G>
        )
        set_target_properties(${target} PROPERTIES MSVC_RUNTIME_LIBRARY
            "$<$<CONFIG:Debug>:MultiThreadedDebugDLL>$<$<CONFIG:Release>:MultiThreadedDLL>"
        )
    else()
        target_compile_options(${target} PRIVATE
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3 -march=native -DNDEBUG>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0 -g3 -fno-omit-frame-pointer>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-g -G>
        )
    endif()
    target_compile_definitions(${target} PRIVATE
        $<$<CONFIG:Debug>:DEBUG>
        $<$<NOT:$<CONFIG:Debug>>:NDEBUG>
    )
endfunction()

function(configure_cuda_trt_target target)
    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/infer
        ${TRT_PATH}/include
    )
    target_link_directories(${target} PUBLIC ${TRT_LIB_DIR})
    target_link_libraries(${target} PUBLIC CUDA::cudart ${TRT_LIBS})
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_compile_features(${target} PRIVATE cxx_std_17)
endfunction()

set(BIN_DIR "${CMAKE_SOURCE_DIR}/bin")

add_library(trtyolo STATIC
    core/core.cpp
    core/buffer.cpp
    utils/common.cpp
    infer/backend.cpp
    infer/letterbox.cu
    infer/trtyolo.cpp
)
configure_cuda_trt_target(trtyolo)
set_target_compile_options(trtyolo)

add_executable(detect detect.cpp)
target_include_directories(detect PRIVATE
    ${CMAKE_SOURCE_DIR}/infer
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(detect PRIVATE trtyolo ${OpenCV_LIBS})
set_target_compile_options(detect)

set_target_properties(detect PROPERTIES
    OUTPUT_NAME "detect"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    RUNTIME_OUTPUT_DIRECTORY "${BIN_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${BIN_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${BIN_DIR}"
)

if(WIN32)
    set(TRT_RUNTIME_DLLS
        "${TRT_LIB_DIR}/nvinfer_10.dll"
        "${TRT_LIB_DIR}/nvinfer_plugin_10.dll"
        "${TRT_LIB_DIR}/nvinfer_dispatch_10.dll"
    )
    add_custom_command(TARGET detect POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TRT_RUNTIME_DLLS} "${BIN_DIR}"
        COMMENT "Copy TensorRT runtime DLLs to bin/"
    )
endif()
