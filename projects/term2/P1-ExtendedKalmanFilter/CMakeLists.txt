project(ExtendedKF)

cmake_minimum_required (VERSION 3.5)

set(CMAKE_BUILD_TYPE Debug)

add_definitions(-std=c++0x)

set(sources
    src/FusionEKF.cpp
    src/kalman_filter.cpp
    src/main.cpp
    src/tools.cpp)

add_executable(ExtendedKF ${sources})
