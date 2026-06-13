# ----------------------------------------------------------------------------
# rrtTests: RRT*-style growing-tree NN benchmark on SE(2), comparing the
# incremental manifold KD-tree against rebuild-each-iteration, brute force, and
# the naive Euclidean-on-raw-coordinates workaround. Buildable standalone.
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16)
project(rrtTests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# Path to the proposed library's header (branch nanoflann2).
set(NANOFLANN_INCLUDE_DIR "$ENV{HOME}/code/nanoflann/include"
    CACHE PATH "Path to the proposed library's header directory")

find_package(Threads REQUIRED)

add_executable(bench_rrt_se2 bench_rrt_se2.cpp)
target_include_directories(bench_rrt_se2 PRIVATE ${NANOFLANN_INCLUDE_DIR})
target_link_libraries(bench_rrt_se2 PRIVATE Threads::Threads)
