# ----------------------------------------------------------------------------
# concurrentBuildTests: paired build of the same sources against two nanoflann
# working copies (a baseline header and a candidate header), to compare the
# concurrent KD-tree builder for correctness and for speed.
#
# Compile flags deliberately match benchmarkTool/realTests (-O3 -DNDEBUG, no
# -march=native), so the timings here are comparable with those.
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16)
project(concurrentBuildTests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

find_package(Threads REQUIRED)

set(NANOFLANN_BASELINE_INCLUDE_DIR "$ENV{HOME}/ros2_ws/src/nanoflann/include"
    CACHE PATH "Header directory of the baseline nanoflann working copy")
set(NANOFLANN_CANDIDATE_INCLUDE_DIR "${NANOFLANN_BASELINE_INCLUDE_DIR}"
    CACHE PATH "Header directory of the candidate nanoflann working copy")

message(STATUS "baseline  nanoflann: ${NANOFLANN_BASELINE_INCLUDE_DIR}")
message(STATUS "candidate nanoflann: ${NANOFLANN_CANDIDATE_INCLUDE_DIR}")

# One target per (source, header) pair, suffixed _baseline / _candidate.
function(add_paired_target name src)
  foreach(which baseline candidate)
    string(TOUPPER ${which} WHICH)
    add_executable(${name}_${which} ${src})
    target_include_directories(${name}_${which}
        PRIVATE ${NANOFLANN_${WHICH}_INCLUDE_DIR} ${ARGN})
    target_link_libraries(${name}_${which} PRIVATE Threads::Threads)
  endforeach()
endfunction()

# --- correctness harnesses (pure nanoflann, no external data) ---------------
add_paired_target(verify_mt_build      verify_mt_build.cpp)
add_paired_target(verify_dyn_mt_build  verify_dyn_mt_build.cpp)
add_paired_target(exc_safety           exc_safety.cpp)

# --- real-KITTI harnesses (need mola to read the dataset) ------------------
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
  # kitti_bench is benchmarkTool/realTests/benchmark_nanoflann_real.cpp verbatim,
  # compiled once per header, so its numbers stay comparable with that tool's.
  add_paired_target(kitti_bench
      "${REAL_TESTS_DIR}/benchmark_nanoflann_real.cpp" ${REAL_TESTS_DIR})
  foreach(t kitti_bench kitti_sweep kitti_pool_mem)
    if(NOT TARGET ${t}_baseline)
      add_paired_target(${t} ${t}.cpp ${REAL_TESTS_DIR})
    endif()
    foreach(which baseline candidate)
      target_link_libraries(${t}_${which} PRIVATE
          ${MRPT_MAPS_TARGET} mola::mola_input_kitti_dataset mola::mola_yaml)
    endforeach()
  endforeach()
else()
  message(STATUS "KITTI (mola) support: DISABLED (real-cloud targets skipped)")
endif()

# --- instrumented / retuned copies of the candidate header ------------------
# gen_variants.py rewrites the candidate header in place-safe copies under
# ${CMAKE_CURRENT_BINARY_DIR}/generated; targets below are added only for the
# variants it actually produced.
set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
execute_process(
  COMMAND ${CMAKE_COMMAND} -E env python3
          "${CMAKE_CURRENT_SOURCE_DIR}/gen_variants.py"
          "${NANOFLANN_CANDIDATE_INCLUDE_DIR}/nanoflann.hpp" "${GENERATED_DIR}"
  RESULT_VARIABLE gen_result
  OUTPUT_VARIABLE gen_output)
message(STATUS "${gen_output}")

if(EXISTS "${GENERATED_DIR}/peak/nanoflann.hpp")
  add_executable(peak_threads peak_threads.cpp)
  target_include_directories(peak_threads PRIVATE "${GENERATED_DIR}/peak")
  target_link_libraries(peak_threads PRIVATE Threads::Threads)
endif()

# Build-time sweep over the task-size cutoff, one binary per value.
if(TARGET kitti_sweep_candidate)
  file(GLOB cutoff_dirs RELATIVE "${GENERATED_DIR}" "${GENERATED_DIR}/cutoff*")
  foreach(d ${cutoff_dirs})
    string(REGEX REPLACE "^cutoff" "" c "${d}")
    add_executable(kitti_sweep_cutoff${c} kitti_sweep.cpp)
    target_include_directories(kitti_sweep_cutoff${c}
        PRIVATE "${GENERATED_DIR}/${d}" ${REAL_TESTS_DIR})
    target_link_libraries(kitti_sweep_cutoff${c} PRIVATE Threads::Threads
        ${MRPT_MAPS_TARGET} mola::mola_input_kitti_dataset mola::mola_yaml)
  endforeach()

  if(EXISTS "${GENERATED_DIR}/count/nanoflann.hpp")
    add_executable(kitti_sweep_spawncount kitti_sweep.cpp)
    target_include_directories(kitti_sweep_spawncount
        PRIVATE "${GENERATED_DIR}/count" ${REAL_TESTS_DIR})
    target_compile_definitions(kitti_sweep_spawncount PRIVATE NF_COUNT_SPAWNS)
    target_link_libraries(kitti_sweep_spawncount PRIVATE Threads::Threads
        ${MRPT_MAPS_TARGET} mola::mola_input_kitti_dataset mola::mola_yaml)
  endif()
endif()
