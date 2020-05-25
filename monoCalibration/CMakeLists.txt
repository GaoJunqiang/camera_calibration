cmake_minimum_required(VERSION 2.8)
project(mono_calibration)

SET(CMAKE_BUILD_TYPE Release)


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native -fPIC")

find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

ADD_EXECUTABLE(mono_calibration camera_mono.cpp)
TARGET_LINK_LIBRARIES(mono_calibration ${OpenCV_LIBS} "-lpopt")

