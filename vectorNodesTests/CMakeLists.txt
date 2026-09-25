# ----------------------------------------------------------------------------
# vectorNodesTests: paired build of the same sources against two nanoflann
# working copies (baseline / candidate), to evaluate storing the kd-tree nodes
# in one contiguous std::vector instead of a pool of pointer-linked nodes.
#
# Compile flags match benchmarkTool/realTests (-O3 -DNDEBUG, no -march=native).
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16)
project(vectorNodesTests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

find_package(Threads REQUIRED)

set(NANOFLANN_BASELINE_INCLUDE_DIR "$ENV{HOME}/code/nanoflann/include"
    CACHE PATH "Header directory of the baseline nanoflann working copy")
set(NANOFLANN_CANDIDATE_INCLUDE_DIR "${NANOFLANN_BASELINE_INCLUDE_DIR}"
    CACHE PATH "Header directory of the candidate nanoflann working copy")
set(NANOFLANN_VARIANT_INCLUDE_DIR "" CACHE PATH
    "Optional header directory of a third nanoflann copy (KITTI benchmark only)")

function(add_paired_target name src)
  foreach(which baseline candidate)
    string(TOUPPER ${which} WHICH)
    add_executable(${name}_${which} ${src})
    target_include_directories(${name}_${which}
        PRIVATE ${NANOFLANN_${WHICH}_INCLUDE_DIR} ${ARGN})
    target_link_libraries(${name}_${which} PRIVATE Threads::Threads)
  endforeach()
endfunction()

add_paired_target(fingerprint fingerprint.cpp)
add_paired_target(exc_safety  exc_safety.cpp)

set(REAL_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarkTool/realTests")
find_package(mola_input_kitti_dataset QUIET)
find_package(mola_yaml QUIET)
find_package(mrpt-maps QUIET)
if(NOT mrpt-maps_FOUND)
  find_package(mrpt_maps QUIET)
endif()
if(TARGET mrpt::maps)
  set(MRPT_MAPS_TARGET mrpt::maps)
elseif(TARGET mrpt::mrpt_maps)
  set(MRPT_MAPS_TARGET mrpt::mrpt_maps)
endif()

if(mola_input_kitti_dataset_FOUND AND mola_yaml_FOUND AND MRPT_MAPS_TARGET)
  message(STATUS "KITTI (mola) support: ENABLED")
  add_paired_target(kitti_vn kitti_vn.cpp ${REAL_TESTS_DIR})
  set(kitti_variants baseline candidate)
  # Optional third header, e.g. a tweaked copy of the candidate.
  if(NANOFLANN_VARIANT_INCLUDE_DIR)
    add_executable(kitti_vn_variant kitti_vn.cpp)
    target_include_directories(kitti_vn_variant
        PRIVATE ${NANOFLANN_VARIANT_INCLUDE_DIR} ${REAL_TESTS_DIR})
    target_link_libraries(kitti_vn_variant PRIVATE Threads::Threads)
    list(APPEND kitti_variants variant)
  endif()
  foreach(which ${kitti_variants})
    target_link_libraries(kitti_vn_${which} PRIVATE
        ${MRPT_MAPS_TARGET} mola::mola_input_kitti_dataset mola::mola_yaml)
  endforeach()
else()
  message(STATUS "KITTI (mola) support: DISABLED")
endif()

# Build-time probe, one binary per header directory listed in PROBE_HEADERS
# (name=dir;name=dir;...).
if(TARGET kitti_vn_baseline)
  set(PROBE_HEADERS "" CACHE STRING "name=include_dir;... for build_probe_<name>")
  foreach(pair ${PROBE_HEADERS})
    string(REPLACE "=" ";" kv ${pair})
    list(GET kv 0 pname)
    list(GET kv 1 pdir)
    add_executable(build_probe_${pname} build_probe.cpp)
    target_include_directories(build_probe_${pname} PRIVATE ${pdir} ${REAL_TESTS_DIR})
    target_link_libraries(build_probe_${pname} PRIVATE Threads::Threads
        ${MRPT_MAPS_TARGET} mola::mola_input_kitti_dataset mola::mola_yaml)
  endforeach()
endif()

# Extra copies of the KITTI benchmark, one per header directory listed in
# KITTI_EXTRA_HEADERS (name=dir;name=dir;...).
if(TARGET kitti_vn_baseline)
  set(KITTI_EXTRA_HEADERS "" CACHE STRING "name=include_dir;... for kitti_vn_<name>")
  foreach(pair ${KITTI_EXTRA_HEADERS})
    string(REPLACE "=" ";" kv ${pair})
    list(GET kv 0 pname)
    list(GET kv 1 pdir)
    add_executable(kitti_vn_${pname} kitti_vn.cpp)
    target_include_directories(kitti_vn_${pname} PRIVATE ${pdir} ${REAL_TESTS_DIR})
    target_link_libraries(kitti_vn_${pname} PRIVATE Threads::Threads
        ${MRPT_MAPS_TARGET} mola::mola_input_kitti_dataset mola::mola_yaml)
  endforeach()
endif()
