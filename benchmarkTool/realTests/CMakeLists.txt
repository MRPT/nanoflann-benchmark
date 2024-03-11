


find_package(mola_input_kitti_dataset REQUIRED)
find_package(mola_yaml REQUIRED)
find_package(mrpt-maps REQUIRED)
message(STATUS "mola_input_kitti_dataset: ${mola_input_kitti_dataset_FOUND}")

# ===================
# NANOFLANN
# ===================
add_executable(benchmark_nanoflann_real benchmark_nanoflann_real.cpp)
target_link_libraries(benchmark_nanoflann_real
    nanoflann::nanoflann
    Threads::Threads
    mrpt::maps
    mola::mola_input_kitti_dataset
    mola::mola_yaml
)
add_dependencies(benchmarks benchmark_nanoflann_real)

# ===================
# FLANN
# ===================
# Since flann 1.8.5, LZ4 is vendored. Building against this version fails.
# This explicitly states to use the system's version of LZ4 instead.
# See: https://github.com/mariusmuja/flann/issues/384#issuecomment-668782051
message(STATUS "Finding LZ4 libraries")
find_library(LZ4_LIBRARIES NAMES liblz4.so HINTS "lz4/lib")
if(LZ4_LIBRARIES)
    message(STATUS "Found: ${LZ4_LIBRARIES}")
    else()
    message(STATUS "Not found: ${LZ4_LIBRARIES}")
    message(FATAL_ERROR "Cannot find required LZ4 libraries")
endif()

set(flann_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/3rdparty/flann/src/cpp")
include_directories(${flann_INCLUDE_DIRS})
set(flann_LIBRARIES "${CMAKE_BINARY_DIR}/flann-prefix/src/flann-build/lib/libflann_cpp.so")
add_executable(benchmark_flann_real benchmark_flann_real.cpp)
target_link_libraries(benchmark_flann_real nanoflann::nanoflann ${flann_LIBRARIES} ${LZ4_LIBRARIES})
add_dependencies(benchmark_flann_real flann)
add_dependencies(benchmarks benchmark_flann_real)

# ===================
# FASTANN
# ===================
set(fastann_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/3rdparty/fastann")
include_directories(${fastann_INCLUDE_DIRS})
set(fastann_LIBRARIES "${CMAKE_BINARY_DIR}/fastann-prefix/src/fastann-build/libfastann.so")
add_executable(benchmark_fastann_real benchmark_fastann_real.cpp)
target_link_libraries(benchmark_fastann_real nanoflann::nanoflann ${fastann_LIBRARIES})
add_dependencies(benchmark_fastann_real fastann)
add_dependencies(benchmarks benchmark_fastann_real)

# ===================
# LIBKDTREE
# ===================
set(libkdtree_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/3rdparty/libkdtree")
include_directories(${libkdtree_INCLUDE_DIRS})
add_executable(benchmark_libkdtree_real benchmark_libkdtree_real.cpp)
set_target_properties(benchmark_nanoflann_real benchmark_flann_real benchmark_fastann_real benchmark_libkdtree_real PROPERTIES FOLDER "RealTests")
add_dependencies(benchmark_libkdtree_real libkdtree)
add_dependencies(benchmarks benchmark_libkdtree_real)
