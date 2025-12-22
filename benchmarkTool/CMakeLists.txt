
add_custom_target(benchmarks)

add_subdirectory(realTests)
add_subdirectory(randomTests)

add_executable(alloc_benchmark alloc_benchmark.cpp)