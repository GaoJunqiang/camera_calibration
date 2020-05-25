cmake_minimum_required(VERSION 2.8.11)
project(CALIB_CAM)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
#set(OpenCV_DIR "/usr/local/opencv2411/share/OpenCV")

find_package(OpenCV REQUIRED)
include_directories($(OpenCV_INCLUDE_DIRS))

add_executable(stereoCalibration stereoCalibration.cpp)
target_link_libraries(stereoCalibration ${OpenCV_LIBS} "-lpopt")


ADD_EXECUTABLE(segmentImg segmentImg.cpp )
TARGET_LINK_LIBRARIES(segmentImg ${OpenCV_LIBS})