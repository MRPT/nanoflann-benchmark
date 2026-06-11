# ----------------------------------------------------------------------------
# overheadTests: paired build of the identical R^3 benchmark source with the
# manifold feature compiled in (default, C++17) vs out (NANOFLANN_NO_MANIFOLDS)
# to substantiate the "no-overhead drop-in" claim.
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16)
project(overheadTests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

set(NANOFLANN_INCLUDE_DIR "$ENV{HOME}/code/nanoflann-manifold/include"
    CACHE PATH "Path to the proposed library's header directory")

add_executable(bench_overhead_on bench_overhead.cpp)
target_include_directories(bench_overhead_on PRIVATE ${NANOFLANN_INCLUDE_DIR})

add_executable(bench_overhead_off bench_overhead.cpp)
target_include_directories(bench_overhead_off PRIVATE ${NANOFLANN_INCLUDE_DIR})
target_compile_definitions(bench_overhead_off PRIVATE NANOFLANN_NO_MANIFOLDS)
