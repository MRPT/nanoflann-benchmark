# ----------------------------------------------------------------------------
# multiLibTests: one binary comparing many C++ NN libraries on a shared
# dataset (single-tree build + query). Buildable standalone or from the root.
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16)
project(multiLibTests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

include(FetchContent)

# ---- the proposed library (anonymized): local nanoflann 2.0 dev header ------
set(NANOFLANN_INCLUDE_DIR "$ENV{HOME}/code/nanoflann/include"
    CACHE PATH "Path to the proposed library's header directory")

add_executable(bench bench.cpp)
target_include_directories(bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR} ${NANOFLANN_INCLUDE_DIR})

# ---- FLANN (system: libflann-dev) -------------------------------------------
option(WITH_FLANN "Benchmark against FLANN" ON)
if(WITH_FLANN)
  find_path(FLANN_INCLUDE_DIR flann/flann.hpp)
  find_library(LZ4_LIB lz4)
  if(FLANN_INCLUDE_DIR)
    message(STATUS "FLANN: ${FLANN_INCLUDE_DIR}")
    target_include_directories(bench PRIVATE ${FLANN_INCLUDE_DIR})
    target_compile_definitions(bench PRIVATE HAVE_FLANN)
    if(LZ4_LIB)
      target_link_libraries(bench PRIVATE ${LZ4_LIB})
    endif()
  endif()
endif()

# ---- PCL KdTreeFLANN (system: libpcl-dev) -----------------------------------
option(WITH_PCL "Benchmark against PCL KdTreeFLANN" ON)
if(WITH_PCL)
  find_package(PCL QUIET COMPONENTS common kdtree)
  if(PCL_FOUND)
    message(STATUS "PCL: ${PCL_VERSION}")
    target_include_directories(bench PRIVATE ${PCL_INCLUDE_DIRS})
    target_link_libraries(bench PRIVATE ${PCL_KDTREE_LIBRARIES} ${PCL_COMMON_LIBRARIES})
    target_compile_definitions(bench PRIVATE HAVE_PCL ${PCL_DEFINITIONS})
  endif()
endif()

# ---- libnabo (FetchContent) -------------------------------------------------
option(WITH_LIBNABO "Benchmark against libnabo" ON)
if(WITH_LIBNABO)
  find_package(Eigen3 QUIET)
  set(LIBNABO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(LIBNABO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(LIBNABO_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(libnabo
    GIT_REPOSITORY https://github.com/norlab-ulaval/libnabo.git
    GIT_TAG 1.1.2)
  FetchContent_MakeAvailable(libnabo)
  if(TARGET libnabo::nabo)
    target_link_libraries(bench PRIVATE libnabo::nabo)
    target_compile_definitions(bench PRIVATE HAVE_LIBNABO)
  elseif(TARGET nabo)
    target_link_libraries(bench PRIVATE nabo)
    target_include_directories(bench PRIVATE ${libnabo_SOURCE_DIR})
    target_compile_definitions(bench PRIVATE HAVE_LIBNABO)
  endif()
endif()

# ---- pico_tree (header-only, FetchContent) ----------------------------------
option(WITH_PICOTREE "Benchmark against pico_tree" ON)
if(WITH_PICOTREE)
  FetchContent_Declare(picotree
    GIT_REPOSITORY https://github.com/Jaybro/pico_tree.git
    GIT_TAG v0.8.3)
  FetchContent_MakeAvailable(picotree)
  target_include_directories(bench PRIVATE ${picotree_SOURCE_DIR}/src/pico_tree)
  target_compile_definitions(bench PRIVATE HAVE_PICOTREE)
endif()

# ---- fastann (FetchContent; Philbin's exact/approx ANN) ---------------------
option(WITH_FASTANN "Benchmark against fastann" ON)
if(WITH_FASTANN)
  FetchContent_Declare(fastann
    GIT_REPOSITORY https://github.com/philbinj/fastann.git
    GIT_TAG master)
  FetchContent_MakeAvailable(fastann)
  if(TARGET fastann)
    target_link_libraries(bench PRIVATE fastann)
    target_include_directories(bench PRIVATE ${fastann_SOURCE_DIR})
    target_compile_definitions(bench PRIVATE HAVE_FASTANN)
  endif()
endif()

# ---- libkdtree++ (header-only, FetchContent) --------------------------------
option(WITH_LIBKDTREE "Benchmark against libkdtree++" ON)
if(WITH_LIBKDTREE)
  FetchContent_Declare(libkdtree
    GIT_REPOSITORY https://github.com/nvmd/libkdtree.git
    GIT_TAG master)
  FetchContent_Populate(libkdtree)
  target_include_directories(bench PRIVATE ${libkdtree_SOURCE_DIR})
  target_compile_definitions(bench PRIVATE HAVE_LIBKDTREE)
endif()

# ---- hnswlib (header-only, approximate; FetchContent) -----------------------
option(WITH_HNSWLIB "Benchmark against hnswlib (approximate)" ON)
if(WITH_HNSWLIB)
  FetchContent_Declare(hnswlib
    GIT_REPOSITORY https://github.com/nmslib/hnswlib.git
    GIT_TAG v0.8.0)
  FetchContent_MakeAvailable(hnswlib)
  target_include_directories(bench PRIVATE ${hnswlib_SOURCE_DIR})
  target_compile_definitions(bench PRIVATE HAVE_HNSWLIB)
endif()
