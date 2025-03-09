cmake_minimum_required(VERSION 3.30)
project(OpenCLTest)

set(CMAKE_CXX_STANDARD 14)
include_directories(inc)

list(APPEND CMAKE_INSTALL_PREFIX "$ENV{OpenCLRoot}")
find_package(OpenCL CONFIG REQUIRED)

add_executable(OpenCLTest
    inc/deep_network.hpp
    inc/matrix.hpp
    inc/square_matrix.hpp
    inc/vector.hpp
    src/deep_network.cpp
    src/main.cpp
    src/matrix.cpp
    src/square_matrix.cpp
    src/vector.cpp)

add_compile_options(/EHsc)

target_link_libraries(OpenCLTest OpenCL::OpenCL)
target_link_libraries(OpenCLTest OpenCL::UtilsCpp)
target_compile_definitions(OpenCLTest
    PRIVATE
        CL_HPP_ENABLE_EXCEPTIONS
        CL_HPP_TARGET_OPENCL_VERSION=210)
