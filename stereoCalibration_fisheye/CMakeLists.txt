cmake_minimum_required(VERSION 2.8.11)
project(FISHEYE_STEREO)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(OpenCV_DIR "/usr/local/opencv3_2/share/OpenCV")

find_package(OpenCV REQUIRED)
include_directories($(OpenCV_INCLUDE_DIRS))

add_executable(stereoCalibrate_fisheye stereo_calib_fisheye.cpp)
target_link_libraries(stereoCalibrate_fisheye ${OpenCV_LIBS} "-lpopt")
