# ----------------------------------------------------------------------------
# manifoldLibTests: head-to-head of the proposed manifold KD-tree vs the prior
# exact-manifold NN libraries (nigh = Ichnowski, MPNN = Yershova/LaValle).
# Buildable standalone.
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16)
project(manifoldLibTests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

include(FetchContent)

# ---- proposed library: nanoflann 2.0 dev header (manifold support) ----------
set(NANOFLANN_INCLUDE_DIR "$ENV{HOME}/code/nanoflann/include"
    CACHE PATH "Path to the proposed library's header directory (branch nanoflann2)")

find_package(Eigen3 REQUIRED)

add_executable(bench_manifold bench_manifold.cpp)
target_include_directories(bench_manifold PRIVATE ${NANOFLANN_INCLUDE_DIR})
target_link_libraries(bench_manifold PRIVATE Eigen3::Eigen)
# manifold feature is C++17-gated inside nanoflann.hpp; enable explicitly.
target_compile_definitions(bench_manifold PRIVATE NANOFLANN_ENABLE_MANIFOLDS)

# ---- nigh (UNC-Robotics): header-only, Ichnowski exact SO(3)/SE(3) ----------
option(WITH_NIGH "Benchmark against nigh" ON)
if(WITH_NIGH)
  FetchContent_Declare(nigh
    GIT_REPOSITORY https://github.com/UNC-Robotics/nigh.git
    GIT_TAG master)
  FetchContent_MakeAvailable(nigh)
  target_include_directories(bench_manifold PRIVATE ${nigh_SOURCE_DIR}/src)
  target_compile_definitions(bench_manifold PRIVATE HAVE_NIGH)
endif()

# ---- MPNN (Yershova & LaValle): builds a small static lib from its sources --
option(WITH_MPNN "Benchmark against MPNN" ON)
if(WITH_MPNN)
  FetchContent_Declare(mpnn
    GIT_REPOSITORY https://github.com/drewm1980/mpnn.git
    GIT_TAG master)
  FetchContent_Populate(mpnn)
  file(GLOB MPNN_SRC ${mpnn_SOURCE_DIR}/src/*.cpp)
  add_library(mpnn_DNN STATIC ${MPNN_SRC})
  target_include_directories(mpnn_DNN PUBLIC ${mpnn_SOURCE_DIR}/include)
  target_compile_options(mpnn_DNN PRIVATE -w)  # silence legacy warnings
  target_link_libraries(bench_manifold PRIVATE mpnn_DNN)
  target_compile_definitions(bench_manifold PRIVATE HAVE_MPNN)
endif()
