add_executable(benchmark_nanoflann_random benchmark_nanoflann_random.cpp)
target_link_libraries(benchmark_nanoflann_random nanoflann::nanoflann)
add_dependencies(benchmarks benchmark_nanoflann_random)


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
add_executable(benchmark_flann_random benchmark_flann_random.cpp)
add_dependencies(benchmark_flann_random flann)
target_link_libraries(benchmark_flann_random nanoflann::nanoflann ${flann_LIBRARIES} ${LZ4_LIBRARIES})
add_dependencies(benchmarks benchmark_flann_random)

set(fastann_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/3rdparty/fastann")
include_directories(${fastann_INCLUDE_DIRS})
set(fastann_LIBRARIES "${CMAKE_BINARY_DIR}/fastann-prefix/src/fastann-build/libfastann.so")
add_executable(benchmark_fastann_random benchmark_fastann_random.cpp)
add_dependencies(benchmark_fastann_random fastann)
target_link_libraries(benchmark_fastann_random nanoflann::nanoflann ${fastann_LIBRARIES})
add_dependencies(benchmarks benchmark_fastann_random)

set(libkdtree_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/3rdparty/libkdtree")
include_directories(${libkdtree_INCLUDE_DIRS})
add_executable(benchmark_libkdtree_random benchmark_libkdtree_random.cpp)
add_dependencies(benchmark_libkdtree_random libkdtree)
set_target_properties(benchmark_nanoflann_random benchmark_flann_random benchmark_fastann_random benchmark_libkdtree_random PROPERTIES FOLDER "RandomTests")
add_dependencies(benchmarks benchmark_nanoflann_random)
