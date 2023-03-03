cmake_minimum_required(VERSION 3.16)
project(kuiper_course)
<<<<<<< HEAD
set(CMAKE_CXX_STANDARD 17)

# DEBUG
option(DEBUG "Debug the project" OFF)
message("TEST_DEBUG Value: ${DEBUG}")
if(DEBUG)
    add_compile_options("-DDEBUG")
endif(DEBUG)

find_package(glog REQUIRED)
find_package(benchmark REQUIRED)
include_directories(./include)
set(link_lib glog pthread gtest)
set(link_math_lib armadillo blas lapack)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_executable(kuiper_course main.cpp)

target_include_directories(kuiper_course PUBLIC /usr/include/armadillo_bits)
target_include_directories(kuiper_course PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_libraries(kuiper_course ${link_lib} ${link_math_lib} benchmark::benchmark)

enable_testing()
add_subdirectory(test)
add_subdirectory(benchmark)
=======

set(CMAKE_CXX_STANDARD 17)

find_package(glog REQUIRED)

set(link_lib glog pthread gtest)
set(link_math_lib armadillo blas lapack)
add_executable(kuiper_course main.cpp)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

target_include_directories(kuiper_course PUBLIC /usr/include/armadillo_bits)
target_include_directories(kuiper_course PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_libraries(kuiper_course ${link_lib} ${link_math_lib})

enable_testing()
add_subdirectory(test)
>>>>>>> cc18220129f36e4521c5895c1dab566b2107d767
