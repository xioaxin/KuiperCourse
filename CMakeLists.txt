cmake_minimum_required(VERSION 3.16)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CUDACXX /usr/local/cuda/bin/nvcc)
project(kuiper_course CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

# DEBUG
option(DEBUG "Debug the project" OFF)
message("TEST_DEBUG Value: ${DEBUG}")
#CUDA
option(CUDA "Enable the project cuda" ON)
message("CUDA_ENABLE Value: ${CUDA}")
#OPENMP
option(OPENMP "OpenMP accelerate the project" ON)
message("OPENMP_ENABLE Value: ${OPENMP}")
#OPENCV
option(OPENCV "Opencv Module" ON)
message("OPENCV_ENABLE Value: ${OPENCV}")
#BENCHMARK
option(BENCHMARK "Benchmark Module" ON)
message("BENCHMARK_ENABLE Value: ${BENCHMARK}")

#FSANITIZE
option(FSANITIZE "Fsanitize Module" OFF)
message("FSANITIZE_ENABLE Value: ${FSANITIZE}")

find_package(glog REQUIRED)
find_package(magic_enum CONFIG REQUIRED)

include_directories(./include)
set(link_lib glog pthread gtest)
set(link_math_lib armadillo blas lapack)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
add_executable(kuiper_course main_cu.cu)
if (DEBUG)
    add_compile_options("-DDEBUG")
endif (DEBUG)
if (CUDA)
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif (CUDA)
if (OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif (OPENMP)
if (BENCHMARK)
    find_package(benchmark REQUIRED)
    target_link_libraries(${PROJECT_NAME} benchmark::benchmark)
    add_subdirectory(benchmark)
endif (BENCHMARK)
if (OPENCV)
    find_package(OpenCV REQUIRED)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
endif (OPENCV)

if (FSANITIZE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address ")
endif (FSANITIZE)

target_include_directories(kuiper_course PUBLIC /usr/include/armadillo_bits)
target_include_directories(kuiper_course PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_libraries(kuiper_course ${link_lib} ${link_math_lib} magic_enum::magic_enum)
enable_testing()
add_subdirectory(test)
