cmake_minimum_required(VERSION 3.10)
project(nanoflann_incremental_benchmark CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3 -march=native)

find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common)

# nanoflann: use the local working-copy header (one dir up by default).
if(NOT NANOFLANN_INCLUDE_DIR)
  set(NANOFLANN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../nanoflann/include")
endif()
message(STATUS "Using nanoflann header from: ${NANOFLANN_INCLUDE_DIR}")

# ikd-Tree (external, GPLv2 — comparison only).
set(IKD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/ikd-Tree/ikd-Tree")

add_executable(benchmark_incremental
  benchmark_incremental.cpp
  kitti_loader.cpp
  ${IKD_DIR}/ikd_Tree.cpp
)
target_include_directories(benchmark_incremental PRIVATE
  ${NANOFLANN_INCLUDE_DIR}
  ${IKD_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(benchmark_incremental PRIVATE Threads::Threads ${PCL_LIBRARIES})

# Optional: KITTI data source via mola.
find_package(mola_input_kitti_dataset QUIET)
find_package(mola_yaml QUIET)
find_package(mrpt-maps QUIET)
if(mola_input_kitti_dataset_FOUND AND mola_yaml_FOUND AND mrpt-maps_FOUND)
  message(STATUS "KITTI (mola) support: ENABLED")
  target_compile_definitions(benchmark_incremental PRIVATE HAVE_MOLA_KITTI)
  target_link_libraries(benchmark_incremental PRIVATE
    mola::mola_input_kitti_dataset mola::mola_yaml mrpt::maps)
else()
  message(STATUS "KITTI (mola) support: DISABLED (synthetic data only)")
endif()
