# ----------------------------------------------------------------------------
# loopClosureTests: C5 application benchmark -- loop-closure candidate
# retrieval over real SE(3) keyframe poses (KITTI odometry ground truth).
# Buildable standalone; only needs the proposed library's header.
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16)
project(loopClosureTests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# ---- proposed library: nanoflann 2.0 dev header (manifold support) ----------
set(NANOFLANN_INCLUDE_DIR "$ENV{HOME}/code/nanoflann-manifold/include"
    CACHE PATH "Path to the proposed library's header directory (manifold branch)")

add_executable(bench_loopclosure bench_loopclosure.cpp)
target_include_directories(bench_loopclosure PRIVATE ${NANOFLANN_INCLUDE_DIR})
# manifold feature is C++17-gated inside nanoflann.hpp; enable explicitly.
target_compile_definitions(bench_loopclosure PRIVATE NANOFLANN_ENABLE_MANIFOLDS)
