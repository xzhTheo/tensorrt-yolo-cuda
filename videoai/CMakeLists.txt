cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0091 NEW)

project(videoai LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)
find_package(TensorRT-YOLO REQUIRED)

add_executable(videoai
    src/main.cpp
    src/app_ui.cpp
    src/detector.cpp
    src/visualizer.cpp
    src/video_processor.cpp
    src/ffmpeg_io.cpp
)

target_include_directories(videoai PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT-YOLO_INCLUDE_DIRS}
)

target_link_libraries(videoai PRIVATE
    ${OpenCV_LIBS}
    ${TensorRT-YOLO_LIBs}
    comdlg32
    user32
    gdi32
)

if(MSVC)
    target_compile_options(videoai PRIVATE
        $<$<CONFIG:Release>:/O2 /Oi /fp:fast>
        /utf-8
    )
    set_target_properties(videoai PROPERTIES MSVC_RUNTIME_LIBRARY
        "$<$<CONFIG:Debug>:MultiThreadedDebugDLL>$<$<CONFIG:Release>:MultiThreadedDLL>"
    )
endif()

set_target_properties(videoai PROPERTIES
    OUTPUT_NAME "videoai"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/bin"
)
