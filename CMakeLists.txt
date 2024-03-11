# ----------------------------------------------------------------------------
# Root CMake file for nanoflann-benchmark
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.1)

project(nanoflann-benchmark)

find_package(Threads REQUIRED)  # TODO: Missing in mola_* packages ?

find_package(nanoflann REQUIRED)

# 3rdparty Libraries
include(3rdparty/CMakeLists-flann.txt)
include(3rdparty/CMakeLists-fastann.txt)
include(3rdparty/CMakeLists-libkdtree.txt)

# Benchmarks
add_subdirectory(benchmarkTool)
