cmake_minimum_required(VERSION 2.8.11)
project(FISHEYE_STEREO)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(OpenCV_DIR "/usr/local/opencv3_2/share/OpenCV")

find_package(OpenCV REQUIRED)
include_directories($(OpenCV_INCLUDE_DIRS))

add_executable(get_remap get_remap.cpp)
target_link_libraries(get_remap ${OpenCV_LIBS})

add_executable(get_remap_fisheye get_remap_fisheye.cpp)
target_link_libraries(get_remap_fisheye ${OpenCV_LIBS})

